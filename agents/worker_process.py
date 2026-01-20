"""执行者 Agent 进程

作为独立进程运行的执行者
"""
import asyncio
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Any, Optional

from loguru import logger

from core.config import DEFAULT_WORKER_MODEL, DEFAULT_WORKER_TIMEOUT
from cursor.client import CursorAgentClient, CursorAgentConfig
from process.message_queue import ProcessMessage, ProcessMessageType
from process.worker import AgentWorkerProcess

# 从共享模块导入知识库常量和工具函数（避免重复定义）
from core.knowledge import (
    MAX_CLI_ASK_CHARS_PER_DOC,
    MAX_CLI_ASK_DOCS,
    MAX_CHARS_PER_DOC,
    MAX_KNOWLEDGE_DOCS,
    MAX_TOTAL_KNOWLEDGE_CHARS,
    truncate_knowledge_docs,
)


@dataclass
class KnowledgeIntegrationStats:
    """知识库集成统计数据"""
    # 搜索统计
    total_searches: int = 0
    total_search_time_ms: float = 0.0
    total_docs_found: int = 0
    total_docs_injected: int = 0

    # Payload 统计
    total_payload_chars: int = 0
    payloads: list[int] = field(default_factory=list)

    # 任务统计
    tasks_with_knowledge: int = 0
    tasks_without_knowledge: int = 0

    @property
    def avg_search_time_ms(self) -> float:
        """平均搜索耗时（毫秒）"""
        if self.total_searches == 0:
            return 0.0
        return self.total_search_time_ms / self.total_searches

    @property
    def avg_payload_chars(self) -> float:
        """平均 payload 大小（字符数）"""
        if not self.payloads:
            return 0.0
        return sum(self.payloads) / len(self.payloads)

    @property
    def avg_docs_per_task(self) -> float:
        """每任务平均注入文档数"""
        if self.total_searches == 0:
            return 0.0
        return self.total_docs_injected / self.total_searches

    def record_search(self, duration_ms: float, docs_found: int, docs_injected: int, payload_chars: int) -> None:
        """记录一次搜索"""
        self.total_searches += 1
        self.total_search_time_ms += duration_ms
        self.total_docs_found += docs_found
        self.total_docs_injected += docs_injected
        self.total_payload_chars += payload_chars
        self.payloads.append(payload_chars)
        if docs_injected > 0:
            self.tasks_with_knowledge += 1
        else:
            self.tasks_without_knowledge += 1

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "total_searches": self.total_searches,
            "total_search_time_ms": round(self.total_search_time_ms, 2),
            "avg_search_time_ms": round(self.avg_search_time_ms, 2),
            "total_docs_found": self.total_docs_found,
            "total_docs_injected": self.total_docs_injected,
            "avg_docs_per_task": round(self.avg_docs_per_task, 2),
            "total_payload_chars": self.total_payload_chars,
            "avg_payload_chars": round(self.avg_payload_chars, 2),
            "tasks_with_knowledge": self.tasks_with_knowledge,
            "tasks_without_knowledge": self.tasks_without_knowledge,
        }

    def get_summary(self) -> str:
        """生成统计摘要"""
        if self.total_searches == 0:
            return "无知识库搜索记录"
        hit_rate = self.tasks_with_knowledge / self.total_searches * 100
        return (
            f"知识库集成统计:\n"
            f"  搜索次数: {self.total_searches}\n"
            f"  平均搜索耗时: {self.avg_search_time_ms:.2f}ms\n"
            f"  命中率: {hit_rate:.1f}% ({self.tasks_with_knowledge}/{self.total_searches})\n"
            f"  平均注入文档数: {self.avg_docs_per_task:.2f}\n"
            f"  平均 payload 大小: {self.avg_payload_chars:.0f} 字符"
        )


class WorkerAgentProcess(AgentWorkerProcess):
    """执行者 Agent 进程

    独立进程中运行的执行者，负责：
    1. 接收任务
    2. 调用 Cursor Agent 执行任务
    3. 返回执行结果
    4. [可选] 知识库集成：自动搜索并注入相关文档
    """

    SYSTEM_PROMPT = """你是一个代码执行者。你的职责是:

1. 专注完成分配给你的具体任务
2. 按照指令进行代码修改
3. 确保修改的正确性和完整性
4. 完成后报告执行结果

你可以使用的工具:
- 文件操作（读取、创建、编辑文件）
- 搜索工具（查找代码、文件）
- Shell 命令（运行测试、构建等）

Shell 命令限制（重要）:
- 命令会在 30 秒后超时
- 不要运行长时间驻留的服务器或进程
- 不要运行交互式命令（需要用户输入的）
- 每条命令独立执行，目录变更不持久
- 在其他目录运行时使用: cd <dir> && <command>
- 适合: 状态检查、快速构建、文件操作、测试

执行要求:
- 只关注当前任务，不考虑其他任务
- 严格按照指令执行
- 如遇到问题，记录并报告
- 完成后简要总结所做的修改

请执行以下任务:"""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        inbox: Queue,
        outbox: Queue,
        config: dict,
    ):
        super().__init__(agent_id, agent_type, inbox, outbox, config)
        self.cursor_client: Optional[CursorAgentClient] = None
        self.current_task_id: Optional[str] = None
        self.completed_tasks: list[str] = []
        self.failed_tasks: list[str] = []

        # 知识库集成相关
        self._knowledge_storage: Optional[Any] = None  # 延迟导入类型
        self._knowledge_enabled: bool = False
        self._knowledge_stats: KnowledgeIntegrationStats = KnowledgeIntegrationStats()
        self._knowledge_config: dict = {}

    def on_start(self) -> None:
        """进程启动初始化"""
        # 创建 agent CLI 客户端 - 使用 DEFAULT_WORKER_MODEL 进行编码
        # 使用 --force 允许直接修改文件
        stream_enabled = self.config.get("stream_events_enabled", False)
        cursor_config = CursorAgentConfig(
            working_directory=self.config.get("working_directory", "."),
            timeout=self.config.get("task_timeout", int(DEFAULT_WORKER_TIMEOUT)),
            model=self.config.get("model", DEFAULT_WORKER_MODEL),
            output_format="stream-json" if stream_enabled else "text",
            non_interactive=True,  # 非交互模式
            force_write=True,      # --force 允许直接修改文件
            stream_partial_output=stream_enabled,
            stream_events_enabled=stream_enabled,
            stream_log_console=self.config.get("stream_log_console", True),
            stream_log_detail_dir=self.config.get("stream_log_detail_dir", "logs/stream_json/detail/"),
            stream_log_raw_dir=self.config.get("stream_log_raw_dir", "logs/stream_json/raw/"),
            stream_agent_id=self.agent_id,
            stream_agent_role=self.agent_type,
            stream_agent_name=self.config.get("agent_name"),
        )
        self.cursor_client = CursorAgentClient(cursor_config)
        logger.info(f"[{self.agent_id}] 执行者初始化完成 (模型: {cursor_config.model}, force: True)")

        # 初始化知识库集成（如果配置启用）
        self._init_knowledge_integration()

    def _init_knowledge_integration(self) -> None:
        """初始化知识库集成（可选路径）

        当配置启用时，在 worker 进程启动时初始化 KnowledgeStorage。
        这是一个轻量级组合，仅用于搜索，不需要完整的 KnowledgeManager。
        """
        # 从配置获取知识库集成设置
        ki_config = self.config.get("knowledge_integration", {})
        if not ki_config.get("enabled", False):
            logger.debug(f"[{self.agent_id}] 知识库集成未启用")
            return

        self._knowledge_config = ki_config

        try:
            # 延迟导入，避免在不需要时加载依赖
            from knowledge.storage import KnowledgeStorage, StorageConfig

            storage_path = ki_config.get("storage_path", ".cursor/knowledge")
            enable_vector = ki_config.get("enable_vector_index", False)
            working_dir = self.config.get("working_directory", ".")

            storage_config = StorageConfig(
                storage_root=storage_path,
                enable_vector_index=enable_vector,
            )

            self._knowledge_storage = KnowledgeStorage(
                config=storage_config,
                workspace_root=working_dir,
            )

            # 同步初始化（进程启动时）
            asyncio.run(self._knowledge_storage.initialize())

            self._knowledge_enabled = True
            stats = asyncio.run(self._knowledge_storage.get_stats())
            doc_count = stats.get("document_count", 0)

            logger.info(
                f"[{self.agent_id}] 知识库集成已启用 "
                f"(文档数: {doc_count}, 搜索模式: {ki_config.get('search_mode', 'keyword')}, "
                f"向量索引: {enable_vector})"
            )

        except ImportError as e:
            logger.warning(f"[{self.agent_id}] 知识库模块导入失败，禁用知识库集成: {e}")
            self._knowledge_enabled = False
        except Exception as e:
            logger.warning(f"[{self.agent_id}] 知识库初始化失败，禁用知识库集成: {e}")
            self._knowledge_enabled = False

    async def _search_knowledge_for_task(self, task_data: dict) -> list[dict]:
        """为任务搜索知识库相关文档

        Args:
            task_data: 任务数据

        Returns:
            知识库文档列表（已格式化为 knowledge_docs 格式）
        """
        if not self._knowledge_enabled or not self._knowledge_storage:
            return []

        start_time = time.perf_counter()

        try:
            # 构建搜索查询：使用任务指令作为查询
            instruction = task_data.get("instruction", "")
            if not instruction:
                return []

            # 获取配置参数
            search_mode = self._knowledge_config.get("search_mode", "keyword")
            max_docs = self._knowledge_config.get("max_docs", 3)
            min_score = self._knowledge_config.get("min_score", 0.3)

            # 执行搜索
            results = await self._knowledge_storage.search(
                query=instruction[:500],  # 限制查询长度
                limit=max_docs,
                search_content=True,
                mode=search_mode,
            )

            # 过滤低分结果
            results = [r for r in results if r.score >= min_score]

            if not results:
                self._record_search_stats(start_time, 0, 0, 0)
                return []

            # 加载文档内容并转换为 knowledge_docs 格式
            knowledge_docs = []
            total_payload_chars = 0

            for result in results:
                doc = await self._knowledge_storage.load_document(result.doc_id)
                if doc:
                    # 限制单文档内容长度
                    content = doc.content[:MAX_CHARS_PER_DOC] if doc.content else ""
                    doc_entry = {
                        "title": doc.title or result.title,
                        "url": doc.url or result.url,
                        "content": content,
                        "score": result.score,
                        "source": "worker-knowledge",  # 标记来源
                    }
                    knowledge_docs.append(doc_entry)
                    total_payload_chars += len(content)

            self._record_search_stats(
                start_time,
                len(results),
                len(knowledge_docs),
                total_payload_chars,
            )

            return knowledge_docs

        except Exception as e:
            logger.warning(f"[{self.agent_id}] 知识库搜索失败: {e}")
            self._record_search_stats(start_time, 0, 0, 0)
            return []

    def _record_search_stats(
        self,
        start_time: float,
        docs_found: int,
        docs_injected: int,
        payload_chars: int,
    ) -> None:
        """记录搜索统计数据"""
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._knowledge_stats.record_search(duration_ms, docs_found, docs_injected, payload_chars)

        if self._knowledge_config.get("log_stats", True):
            logger.debug(
                f"[{self.agent_id}] 知识库搜索: "
                f"耗时={duration_ms:.2f}ms, "
                f"找到={docs_found}, 注入={docs_injected}, "
                f"payload={payload_chars}字符"
            )

    def on_stop(self) -> None:
        """进程停止清理"""
        logger.info(f"[{self.agent_id}] 执行者停止，完成 {len(self.completed_tasks)} 个任务")

        # 输出知识库集成统计（如果启用）
        if self._knowledge_enabled and self._knowledge_config.get("log_stats", True):
            self._output_knowledge_stats()

    def _output_knowledge_stats(self) -> None:
        """输出知识库集成统计数据和分析结论"""
        stats = self._knowledge_stats
        summary = stats.get_summary()
        logger.info(f"[{self.agent_id}] {summary}")

        # 分析结论
        if stats.total_searches > 0:
            hit_rate = stats.tasks_with_knowledge / stats.total_searches
            avg_time = stats.avg_search_time_ms
            avg_payload = stats.avg_payload_chars

            # 评估标准：
            # - 搜索耗时 < 100ms 为低开销
            # - 命中率 > 30% 为有效
            # - payload < 5000 字符为合理
            low_overhead = avg_time < 100
            good_hit_rate = hit_rate > 0.3
            reasonable_payload = avg_payload < 5000

            conclusion_parts = []
            if low_overhead and good_hit_rate:
                conclusion_parts.append("推荐保留: 搜索开销低且命中率良好")
            elif not low_overhead:
                conclusion_parts.append(f"注意: 搜索耗时较高 ({avg_time:.1f}ms)")
            elif not good_hit_rate:
                conclusion_parts.append(f"注意: 命中率较低 ({hit_rate*100:.1f}%)")

            if not reasonable_payload and stats.tasks_with_knowledge > 0:
                conclusion_parts.append(f"注意: 平均 payload 较大 ({avg_payload:.0f} 字符)")

            # 综合评估
            if low_overhead and good_hit_rate and reasonable_payload:
                verdict = "✓ 知识库集成方案值得保留"
            elif good_hit_rate and (low_overhead or reasonable_payload):
                verdict = "△ 知识库集成方案可选保留，建议优化"
            else:
                verdict = "✗ 知识库集成方案收益有限，建议禁用"

            conclusion = " | ".join(conclusion_parts) if conclusion_parts else "运行正常"
            logger.info(f"[{self.agent_id}] 知识库集成分析: {conclusion}")
            logger.info(f"[{self.agent_id}] 结论: {verdict}")

            # 发送统计消息到协调器（可选）
            self._send_message(ProcessMessageType.STATUS_RESPONSE, {
                "type": "knowledge_stats",
                "stats": stats.to_dict(),
                "verdict": verdict,
            })

    def handle_message(self, message: ProcessMessage) -> None:
        """处理业务消息

        注意：任务执行通过 asyncio.run 同步执行，但在基类中已标记 _is_busy=True。
        心跳响应由独立的心跳线程处理，不会被任务执行阻塞。
        """
        if message.type == ProcessMessageType.TASK_ASSIGN:
            # 任务执行会阻塞主线程，但心跳线程独立运行
            asyncio.run(self._handle_task(message))

    async def _handle_task(self, message: ProcessMessage) -> None:
        """处理任务"""
        task_data = message.payload.get("task", {})
        task_id = task_data.get("id", "unknown")

        self.current_task_id = task_id

        logger.info(f"[{self.agent_id}] 开始执行任务: {task_id}")

        try:
            # [知识库集成] 搜索相关文档并注入到上下文
            injected_knowledge_docs = []
            if self._knowledge_enabled:
                injected_knowledge_docs = await self._search_knowledge_for_task(task_data)

            # 合并知识库文档：原有的 + 本地搜索的
            task_context = task_data.get("context") or {}
            existing_knowledge_docs = task_context.get("knowledge_docs") or []

            # 如果本地搜索到了文档，合并到现有列表（去重）
            if injected_knowledge_docs:
                existing_urls = {d.get("url") for d in existing_knowledge_docs if d.get("url")}
                for doc in injected_knowledge_docs:
                    if doc.get("url") not in existing_urls:
                        existing_knowledge_docs.append(doc)
                        existing_urls.add(doc.get("url"))
                task_context["knowledge_docs"] = existing_knowledge_docs
                task_data["context"] = task_context

            # 构建执行 prompt（包含知识库文档渲染）
            prompt = self._build_execution_prompt(task_data)

            # 构建上下文（合并 task_data 中的 context）
            context = {
                "task_id": task_id,
                "task_type": task_data.get("type", "custom"),
                "target_files": task_data.get("target_files", []),
                **task_context,  # 合并任务上下文（包含 knowledge_docs 等）
            }

            # 发送进度消息
            self._send_message(ProcessMessageType.TASK_PROGRESS, {
                "task_id": task_id,
                "status": "executing",
                "progress": 0,
                "knowledge_docs_count": len(existing_knowledge_docs),  # 添加知识库文档计数
            })

            # 调用 Cursor Agent 执行
            result = await self.cursor_client.execute(
                instruction=prompt,
                context=context,
            )

            if result.success:
                self.completed_tasks.append(task_id)
                self._send_message(ProcessMessageType.TASK_RESULT, {
                    "task_id": task_id,
                    "success": True,
                    "output": result.output,
                    "duration": result.duration,
                    "files_modified": result.files_modified,
                }, correlation_id=message.id)
                logger.info(f"[{self.agent_id}] 任务完成: {task_id}")
            else:
                self.failed_tasks.append(task_id)
                self._send_message(ProcessMessageType.TASK_RESULT, {
                    "task_id": task_id,
                    "success": False,
                    "error": result.error,
                    "output": result.output,
                }, correlation_id=message.id)
                logger.error(f"[{self.agent_id}] 任务失败: {task_id} - {result.error}")

        except Exception as e:
            logger.exception(f"[{self.agent_id}] 任务执行异常: {e}")
            self.failed_tasks.append(task_id)
            self._send_message(ProcessMessageType.TASK_RESULT, {
                "task_id": task_id,
                "success": False,
                "error": str(e),
            }, correlation_id=message.id)
        finally:
            self.current_task_id = None

    def _build_execution_prompt(self, task_data: dict) -> str:
        """构建执行 prompt

        与协程版 WorkerAgent._build_execution_prompt 对齐，支持知识库文档渲染。

        展示字段说明：
            - task_data["instruction"]: 任务指令（必需）
            - task_data["target_files"]: 涉及的文件列表
            - context["knowledge_docs"]: 知识库文档列表
                - source="cli-ask": CLI Ask 模式智能回答（优先展示）
                - 其他 source: 普通知识库文档

        Payload 限制（使用模块级常量）：
            - CLI Ask 文档: MAX_CLI_ASK_DOCS 个，每个 MAX_CLI_ASK_CHARS_PER_DOC 字符
            - 普通文档: MAX_KNOWLEDGE_DOCS 个，总字符 MAX_TOTAL_KNOWLEDGE_CHARS
            - 降级策略: truncate_knowledge_docs() 函数处理超限情况
        """
        parts = [
            self.SYSTEM_PROMPT,
            f"\n## 任务指令\n{task_data.get('instruction', '')}",
        ]

        # 涉及文件
        target_files = task_data.get("target_files", [])
        if target_files:
            files = "\n".join(f"- {f}" for f in target_files)
            parts.append(f"\n## 涉及文件\n{files}")

        # 获取 context 中的知识库文档
        context = task_data.get("context") or {}

        # 添加知识库文档（Cursor 相关问题自动搜索结果）
        knowledge_docs = context.get("knowledge_docs") or []
        if knowledge_docs:
            # 分离 CLI ask 结果和普通文档
            cli_ask_docs = [d for d in knowledge_docs if d.get("source") == "cli-ask"]
            regular_docs = [d for d in knowledge_docs if d.get("source") != "cli-ask"]

            # 优先展示 CLI ask 查询结果（智能回答）- 使用常量限制
            if cli_ask_docs:
                # 应用 CLI Ask 文档限制
                limited_cli_docs = cli_ask_docs[:MAX_CLI_ASK_DOCS]
                parts.append("\n## 知识库智能回答（CLI Ask 模式）")
                for doc in limited_cli_docs:
                    context_used = doc.get("context_used", [])
                    context_info = f"（参考了 {len(context_used)} 个文档）" if context_used else ""
                    content = doc.get("content", "")[:MAX_CLI_ASK_CHARS_PER_DOC]
                    if len(doc.get("content", "")) > MAX_CLI_ASK_CHARS_PER_DOC:
                        content += "..."
                    parts.append(
                        f"\n### 查询: {doc.get('query', 'N/A')}"
                        f"\n{context_info}"
                        f"\n```\n{content}\n```"
                    )
                if len(cli_ask_docs) > MAX_CLI_ASK_DOCS:
                    parts.append(f"\n... 还有 {len(cli_ask_docs) - MAX_CLI_ASK_DOCS} 个智能回答未显示")

            # 展示普通文档上下文 - 使用降级策略
            if regular_docs:
                # 应用降级策略处理普通文档
                truncated_docs, total_chars = truncate_knowledge_docs(
                    regular_docs,
                    max_docs=MAX_KNOWLEDGE_DOCS,
                    max_chars_per_doc=MAX_CHARS_PER_DOC,
                    max_total_chars=MAX_TOTAL_KNOWLEDGE_CHARS,
                )
                original_count = len(regular_docs)
                shown_count = len(truncated_docs)

                parts.append(f"\n## 参考文档（来自 Cursor 知识库，共 {original_count} 个，展示 {shown_count} 个）")
                for i, doc in enumerate(truncated_docs, 1):
                    truncated_marker = " [已截断]" if doc.get("truncated") else ""
                    parts.append(
                        f"\n### {i}. {doc.get('title', '未命名')}{truncated_marker}"
                        f"\n来源: {doc.get('url', 'N/A')}"
                        f"\n相关度: {doc.get('score', 0):.2f}"
                        f"\n```\n{doc.get('content', '')}\n```"
                    )
                if original_count > shown_count:
                    parts.append(f"\n... 还有 {original_count - shown_count} 个参考文档未显示（总字符限制: {MAX_TOTAL_KNOWLEDGE_CHARS}）")

        # 添加其他上下文（排除已处理的字段）
        exclude_keys = ("task_id", "task_type", "target_files", "reference_code", "knowledge_docs")
        other_context = {k: v for k, v in context.items() if k not in exclude_keys}
        if other_context:
            import json
            parts.append(f"\n## 上下文信息\n```json\n{json.dumps(other_context, ensure_ascii=False, indent=2)}\n```")

        parts.append("\n请开始执行任务:")

        return "\n".join(parts)

    def get_status(self) -> dict:
        """获取当前状态"""
        status = {
            "agent_id": self.agent_id,
            "type": "worker",
            "busy": self.current_task_id is not None,
            "current_task": self.current_task_id,
            "completed_count": len(self.completed_tasks),
            "failed_count": len(self.failed_tasks),
        }

        # 添加知识库集成状态
        if self._knowledge_enabled:
            status["knowledge_integration"] = {
                "enabled": True,
                "stats": self._knowledge_stats.to_dict(),
            }

        return status
