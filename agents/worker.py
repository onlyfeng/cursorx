"""执行者 Agent

负责领取任务并专注完成，不关心全局
支持语义搜索增强的上下文获取
支持知识库自动搜索（Cursor 相关问题）
"""
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from core.base import AgentConfig, AgentRole, AgentStatus, BaseAgent
from cursor.client import CursorAgentConfig
from cursor.cloud_client import CloudAuthConfig
from cursor.executor import (
    AgentExecutor,
    AgentExecutorFactory,
    CLIAgentExecutor,
    ExecutionMode,
)
from tasks.task import Task

# 可选的语义搜索支持
if TYPE_CHECKING:
    from indexing import SemanticSearch
    from knowledge import KnowledgeManager

# 从共享模块导入 Cursor 关键词检测（避免重复定义）
from core.knowledge import is_cursor_related


class WorkerConfig(BaseModel):
    """执行者配置"""
    name: str = "worker"
    working_directory: str = "."
    max_concurrent_tasks: int = 1      # 同时处理的任务数（通常为1）
    task_timeout: int = 500            # 任务超时时间
    cursor_config: CursorAgentConfig = Field(default_factory=CursorAgentConfig)
    # 执行模式配置
    execution_mode: ExecutionMode = ExecutionMode.CLI  # 执行模式: cli, cloud, auto
    cloud_auth_config: Optional[CloudAuthConfig] = None  # Cloud 认证配置
    # 语义搜索配置（可选增强）
    enable_context_search: bool = False        # 是否启用上下文搜索
    context_search_top_k: int = 5              # 上下文搜索返回结果数
    context_search_min_score: float = 0.4      # 最低相似度阈值
    # 知识库配置（Cursor 相关问题自动搜索）
    enable_knowledge_search: bool = True       # 是否启用知识库搜索
    knowledge_search_top_k: int = 5            # 知识库搜索返回结果数
    # CLI ask 模式配置（用于知识库增强查询）
    enable_cli_ask_mode: bool = False          # 是否启用 CLI ask 模式查询
    cli_ask_timeout: int = 60                  # CLI ask 查询超时时间
    cli_ask_model: Optional[str] = None        # CLI ask 使用的模型（默认使用系统默认）
    # 中间提交配置
    enable_intermediate_commit: bool = False   # 是否启用中间提交建议
    intermediate_commit_threshold: int = 5     # 更改文件数阈值，超过则建议提交


class WorkerAgent(BaseAgent):
    """执行者 Agent

    职责:
    1. 从任务队列领取任务
    2. 专注执行分配的任务
    3. 完成后提交变更
    4. 不与其他执行者协调
    """

    # 执行者系统提示
    SYSTEM_PROMPT = """你是一个代码执行者。你的职责是:

1. 专注完成分配给你的具体任务
2. 按照指令进行代码修改
3. 确保修改的正确性和完整性
4. 完成后报告执行结果

执行要求:
- 只关注当前任务，不考虑其他任务
- 严格按照指令执行
- 如遇到问题，记录并报告
- 完成后简要总结所做的修改

请执行以下任务:"""

    def __init__(
        self,
        config: WorkerConfig,
        semantic_search: Optional["SemanticSearch"] = None,
        knowledge_manager: Optional["KnowledgeManager"] = None,
    ):
        agent_config = AgentConfig(
            role=AgentRole.WORKER,
            name=config.name,
            working_directory=config.working_directory,
        )
        super().__init__(agent_config)
        self.worker_config = config
        self._apply_stream_config(self.worker_config.cursor_config)

        # 使用 AgentExecutorFactory 创建执行器
        self._executor: AgentExecutor = AgentExecutorFactory.create(
            mode=config.execution_mode,
            cli_config=config.cursor_config,
            cloud_auth_config=config.cloud_auth_config,
        )

        # 保留 cursor_client 属性以保持向后兼容
        if isinstance(self._executor, CLIAgentExecutor):
            self.cursor_client = self._executor.client
        else:
            # 对于其他执行器类型，创建一个备用客户端（用于非执行操作）
            from cursor.client import CursorAgentClient
            self.cursor_client = CursorAgentClient(config.cursor_config)

        self.current_task: Optional[Task] = None
        self.completed_tasks: list[str] = []

        # 语义搜索增强（可选）
        self._semantic_search: Optional["SemanticSearch"] = semantic_search
        self._search_enabled = config.enable_context_search and semantic_search is not None
        if self._search_enabled:
            logger.info(f"[{config.name}] 上下文搜索已启用")

        # 知识库管理器（用于 Cursor 相关问题自动搜索）
        self._knowledge_manager: Optional["KnowledgeManager"] = knowledge_manager
        self._knowledge_search_enabled = config.enable_knowledge_search and knowledge_manager is not None
        if self._knowledge_search_enabled:
            logger.info(f"[{config.name}] 知识库搜索已启用")

        logger.debug(f"[{config.name}] 使用执行模式: {config.execution_mode.value}")

    def _apply_stream_config(self, cursor_config: CursorAgentConfig) -> None:
        """注入流式日志配置与 Agent 标识"""
        cursor_config.stream_agent_id = self.id
        cursor_config.stream_agent_role = self.role.value
        cursor_config.stream_agent_name = self.name
        if cursor_config.stream_events_enabled:
            cursor_config.output_format = "stream-json"
            cursor_config.stream_partial_output = True

    def set_semantic_search(self, search: "SemanticSearch") -> None:
        """设置语义搜索引擎（延迟初始化）

        Args:
            search: SemanticSearch 实例
        """
        self._semantic_search = search
        self._search_enabled = self.worker_config.enable_context_search
        if self._search_enabled:
            logger.info(f"[{self.id}] 上下文搜索已启用")

    def set_knowledge_manager(self, manager: "KnowledgeManager") -> None:
        """设置知识库管理器（延迟初始化）

        Args:
            manager: KnowledgeManager 实例
        """
        self._knowledge_manager = manager
        self._knowledge_search_enabled = self.worker_config.enable_knowledge_search
        if self._knowledge_search_enabled:
            logger.info(f"[{self.id}] 知识库搜索已启用")

    async def execute(self, instruction: str, context: Optional[dict] = None) -> dict[str, Any]:
        """执行任务指令

        Args:
            instruction: 任务指令
            context: 上下文信息

        Returns:
            执行结果
        """
        self.update_status(AgentStatus.RUNNING)
        logger.info(f"[{self.id}] 开始执行: {instruction[:100]}...")

        try:
            # 构建执行 prompt
            prompt = self._build_execution_prompt(instruction, context)

            # 调用执行器执行
            result = await self._executor.execute(
                prompt=prompt,
                working_directory=self.worker_config.working_directory,
                context=context,
                timeout=self.worker_config.task_timeout,
            )

            if result.success:
                self.update_status(AgentStatus.COMPLETED)
                logger.info(f"[{self.id}] 执行成功")
                return {
                    "success": True,
                    "output": result.output,
                    "duration": result.duration,
                }
            else:
                self.update_status(AgentStatus.FAILED)
                logger.error(f"[{self.id}] 执行失败: {result.error}")
                return {
                    "success": False,
                    "error": result.error,
                    "output": result.output,
                }

        except Exception as e:
            logger.exception(f"[{self.id}] 执行异常: {e}")
            self.update_status(AgentStatus.FAILED)
            return {"success": False, "error": str(e)}

    async def execute_task(self, task: Task) -> Task:
        """执行任务对象

        Args:
            task: 任务对象

        Returns:
            更新后的任务对象
        """
        self.current_task = task
        task.start()
        task.assigned_to = self.id

        logger.info(f"[{self.id}] 执行任务: {task.id} - {task.title}")

        # 构建任务上下文
        context = {
            "task_id": task.id,
            "task_type": task.type.value,
            "target_files": task.target_files,
            **task.context,
        }

        # 使用语义搜索获取相关上下文（可选增强）
        if self._search_enabled:
            search_context = await self._search_task_context(task)
            if search_context:
                context["reference_code"] = search_context
                logger.info(f"[{self.id}] 找到 {len(search_context)} 个参考代码片段")

        # 使用知识库搜索获取 Cursor 相关上下文（自动检测）
        if self._knowledge_search_enabled:
            knowledge_context = await self._search_knowledge_context(task)
            if knowledge_context:
                context["knowledge_docs"] = knowledge_context
                logger.info(f"[{self.id}] 从知识库补充 {len(knowledge_context)} 个文档上下文")

        # 执行任务
        result = await self.execute(task.instruction, context)

        if result.get("success"):
            # 检查是否需要建议中间提交
            suggest_commit, changed_files_count = await self._should_suggest_commit()

            task.complete({
                "output": result.get("output", ""),
                "duration": result.get("duration", 0),
                "worker_id": self.id,
                "suggest_commit": suggest_commit,
                "changed_files_count": changed_files_count,
            })
            self.completed_tasks.append(task.id)

            if suggest_commit:
                logger.info(f"[{self.id}] 任务完成: {task.id}，建议中间提交（{changed_files_count} 个文件更改）")
            else:
                logger.info(f"[{self.id}] 任务完成: {task.id}")
        else:
            task.fail(result.get("error", "未知错误"))
            logger.error(f"[{self.id}] 任务失败: {task.id} - {result.get('error')}")

        self.current_task = None
        return task

    async def _search_task_context(self, task: Task) -> list[dict[str, Any]]:
        """搜索任务相关的上下文代码

        根据任务描述和目标文件搜索相关代码作为参考

        Args:
            task: 任务对象

        Returns:
            相关代码片段列表
        """
        if not self._semantic_search:
            return []

        try:
            # 构建搜索查询：结合任务标题和描述
            query = f"{task.title}. {task.description}" if task.description else task.title

            # 优先在目标文件中搜索
            if task.target_files:
                results = await self._semantic_search.search_in_files(
                    query=query,
                    file_paths=task.target_files,
                    top_k=self.worker_config.context_search_top_k,
                    min_score=self.worker_config.context_search_min_score,
                )
            else:
                # 全局搜索
                results = await self._semantic_search.search(
                    query=query,
                    top_k=self.worker_config.context_search_top_k,
                    min_score=self.worker_config.context_search_min_score,
                )

            # 构建参考代码列表
            reference_code = []
            for result in results:
                chunk = result.chunk
                reference_code.append({
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "name": chunk.name,
                    "content": chunk.content,
                    "score": result.score,
                })

            return reference_code

        except Exception as e:
            logger.warning(f"[{self.id}] 上下文搜索失败: {e}")
            return []

    async def _search_knowledge_context(self, task: Task) -> list[dict[str, Any]]:
        """从知识库搜索任务相关的文档上下文

        当任务与 Cursor 相关时，自动搜索知识库补充上下文。
        支持两种模式：
        1. 关键词搜索模式（默认）：快速、轻量
        2. CLI ask 模式：通过 --mode=ask 执行只读查询，获取更智能的回答

        Args:
            task: 任务对象

        Returns:
            相关知识库文档列表
        """
        if not self._knowledge_manager:
            return []

        # 构建搜索查询
        query = f"{task.title}. {task.description}" if task.description else task.title

        # 检查是否与 Cursor 相关（使用共享模块的检测函数）
        if not is_cursor_related(query) and not is_cursor_related(task.instruction):
            return []

        try:
            logger.info(f"[{self.id}] 检测到 Cursor 相关任务，搜索知识库...")

            knowledge_docs = []

            # 如果启用了 CLI ask 模式，使用 ask_with_cli 进行增强查询
            if self.worker_config.enable_cli_ask_mode:
                ask_result = await self._query_knowledge_with_cli_ask(query)
                if ask_result:
                    knowledge_docs.append(ask_result)

            # 同时进行关键词搜索获取原始文档上下文
            results = self._knowledge_manager.search(
                query=query,
                max_results=self.worker_config.knowledge_search_top_k,
            )

            # 构建知识库上下文列表
            for result in results:
                doc = self._knowledge_manager.get_document(result.doc_id)
                if doc:
                    knowledge_docs.append({
                        "title": doc.title or result.url,
                        "url": doc.url,
                        "content": doc.content[:1500],  # 限制内容长度
                        "score": result.score,
                        "source": "cursor-docs",
                    })

            if knowledge_docs:
                logger.info(f"[{self.id}] 从知识库找到 {len(knowledge_docs)} 个相关文档/回答")

            return knowledge_docs

        except Exception as e:
            logger.warning(f"[{self.id}] 知识库搜索失败: {e}")
            return []

    async def _query_knowledge_with_cli_ask(self, query: str) -> Optional[dict[str, Any]]:
        """使用 CLI ask 模式查询知识库

        通过 Cursor CLI 的 --mode=ask 执行只读查询，获取基于知识库的智能回答。

        Args:
            query: 查询问题

        Returns:
            查询结果字典，失败返回 None
        """
        if not self._knowledge_manager:
            return None

        try:
            logger.info(f"[{self.id}] 使用 CLI ask 模式查询知识库...")

            result = await self._knowledge_manager.ask_with_cli(
                question=query,
                max_context_docs=self.worker_config.knowledge_search_top_k,
                timeout=self.worker_config.cli_ask_timeout,
                model=self.worker_config.cli_ask_model,
                working_directory=self.worker_config.working_directory,
            )

            if result.success and result.answer:
                logger.info(f"[{self.id}] CLI ask 查询成功，耗时 {result.duration:.2f}s")
                return {
                    "title": "CLI Ask 查询结果",
                    "url": "cli-ask://knowledge-query",
                    "content": result.answer,
                    "score": 1.0,  # CLI ask 回答优先级最高
                    "source": "cli-ask",
                    "context_used": result.context_used,
                    "query": result.query,
                }
            else:
                if result.error:
                    logger.warning(f"[{self.id}] CLI ask 查询失败: {result.error}")
                return None

        except Exception as e:
            logger.warning(f"[{self.id}] CLI ask 查询异常: {e}")
            return None

    async def _should_suggest_commit(self) -> tuple[bool, int]:
        """检查是否应该建议中间提交

        通过检查 git status 获取更改文件数，根据阈值判断是否建议提交。

        Returns:
            (should_commit, changed_files_count): 是否建议提交，更改文件数
        """
        if not self.worker_config.enable_intermediate_commit:
            return False, 0

        try:
            import subprocess

            # 执行 git status --porcelain 获取更改文件列表
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.worker_config.working_directory,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.warning(f"[{self.id}] git status 执行失败: {result.stderr}")
                return False, 0

            # 统计更改文件数（每行一个文件）
            lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            changed_files_count = len(lines)

            # 判断是否超过阈值
            threshold = self.worker_config.intermediate_commit_threshold
            should_commit = changed_files_count >= threshold

            if should_commit:
                logger.info(
                    f"[{self.id}] 更改文件数 ({changed_files_count}) 达到阈值 ({threshold})，建议提交"
                )

            return should_commit, changed_files_count

        except subprocess.TimeoutExpired:
            logger.warning(f"[{self.id}] git status 执行超时")
            return False, 0
        except FileNotFoundError:
            logger.warning(f"[{self.id}] git 命令未找到")
            return False, 0
        except Exception as e:
            logger.warning(f"[{self.id}] 检查提交建议失败: {e}")
            return False, 0

    def _build_execution_prompt(self, instruction: str, context: Optional[dict] = None) -> str:
        """构建执行 prompt"""
        parts = [
            self.SYSTEM_PROMPT,
            f"\n## 任务指令\n{instruction}",
        ]

        if context:
            if "target_files" in context and context["target_files"]:
                files = "\n".join(f"- {f}" for f in context["target_files"])
                parts.append(f"\n## 涉及文件\n{files}")

            # 添加参考代码（语义搜索结果）
            if "reference_code" in context and context["reference_code"]:
                reference_code = context["reference_code"]
                parts.append(f"\n## 参考代码（共 {len(reference_code)} 个相关片段）")
                for i, code in enumerate(reference_code[:3], 1):  # 只展示前3个
                    parts.append(
                        f"\n### {i}. {code.get('name', '未命名')} ({code['file_path']}:{code['start_line']}-{code['end_line']})"
                        f"\n相似度: {code.get('score', 0):.2f}"
                        f"\n```\n{code.get('content', '')[:500]}\n```"
                    )
                if len(reference_code) > 3:
                    parts.append(f"\n... 还有 {len(reference_code) - 3} 个参考代码片段未显示")

            # 添加知识库文档（Cursor 相关问题自动搜索结果）
            if "knowledge_docs" in context and context["knowledge_docs"]:
                knowledge_docs = context["knowledge_docs"]

                # 分离 CLI ask 结果和普通文档
                cli_ask_docs = [d for d in knowledge_docs if d.get("source") == "cli-ask"]
                regular_docs = [d for d in knowledge_docs if d.get("source") != "cli-ask"]

                # 优先展示 CLI ask 查询结果（智能回答）
                if cli_ask_docs:
                    parts.append("\n## 知识库智能回答（CLI Ask 模式）")
                    for doc in cli_ask_docs:
                        context_used = doc.get("context_used", [])
                        context_info = f"（参考了 {len(context_used)} 个文档）" if context_used else ""
                        parts.append(
                            f"\n### 查询: {doc.get('query', 'N/A')}"
                            f"\n{context_info}"
                            f"\n```\n{doc.get('content', '')[:1200]}\n```"
                        )

                # 展示普通文档上下文
                if regular_docs:
                    parts.append(f"\n## 参考文档（来自 Cursor 知识库，共 {len(regular_docs)} 个）")
                    for i, doc in enumerate(regular_docs[:3], 1):  # 只展示前3个
                        parts.append(
                            f"\n### {i}. {doc.get('title', '未命名')}"
                            f"\n来源: {doc.get('url', 'N/A')}"
                            f"\n相关度: {doc.get('score', 0):.2f}"
                            f"\n```\n{doc.get('content', '')[:800]}\n```"
                        )
                    if len(regular_docs) > 3:
                        parts.append(f"\n... 还有 {len(regular_docs) - 3} 个参考文档未显示")

            # 添加其他上下文（排除已处理的字段）
            exclude_keys = ("task_id", "task_type", "target_files", "reference_code", "knowledge_docs")
            other_context = {k: v for k, v in context.items() if k not in exclude_keys}
            if other_context:
                import json
                parts.append(f"\n## 上下文信息\n```json\n{json.dumps(other_context, ensure_ascii=False, indent=2)}\n```")

        parts.append("\n请开始执行任务:")

        return "\n".join(parts)

    async def reset(self) -> None:
        """重置执行者状态"""
        self.update_status(AgentStatus.IDLE)
        self.current_task = None
        self.clear_context()
        # 不清除 completed_tasks，保留历史记录
        logger.debug(f"[{self.id}] 状态已重置")

    def get_statistics(self) -> dict[str, Any]:
        """获取执行统计"""
        return {
            "worker_id": self.id,
            "status": self.status.value,
            "completed_tasks_count": len(self.completed_tasks),
            "current_task": self.current_task.id if self.current_task else None,
            "context_search_enabled": self._search_enabled,
            "knowledge_search_enabled": self._knowledge_search_enabled,
        }
