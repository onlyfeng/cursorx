"""规划者 Agent

负责探索代码库、分解任务、派生子规划者
支持语义搜索增强的代码库探索
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
from tasks.task import Task, TaskPriority, TaskType

# 可选的语义搜索支持
if TYPE_CHECKING:
    from indexing import SemanticSearch


class PlannerConfig(BaseModel):
    """规划者配置"""
    name: str = "planner"
    working_directory: str = "."
    max_sub_planners: int = 3          # 最大子规划者数量
    max_tasks_per_plan: int = 10       # 单次规划最大任务数
    exploration_depth: int = 3          # 探索深度
    cursor_config: CursorAgentConfig = Field(default_factory=CursorAgentConfig)
    # 执行模式配置
    execution_mode: ExecutionMode = ExecutionMode.CLI  # 执行模式: cli, cloud, auto
    cloud_auth_config: Optional[CloudAuthConfig] = None  # Cloud 认证配置
    # Plan 模式配置
    use_plan_mode: bool = True         # 是否使用 --mode=plan（仅分析不修改文件）
    # 语义搜索配置（可选增强）
    enable_semantic_search: bool = False       # 是否启用语义搜索
    semantic_search_top_k: int = 10            # 搜索返回结果数
    semantic_search_min_score: float = 0.3     # 最低相似度阈值


class PlannerAgent(BaseAgent):
    """规划者 Agent

    职责:
    1. 探索代码库结构
    2. 分析用户目标
    3. 分解为可执行的子任务
    4. 必要时派生子规划者处理特定区域
    """

    # 规划者系统提示
    SYSTEM_PROMPT = """你是一个代码项目规划者。你的职责是:

1. 分析用户的目标需求
2. 探索代码库结构，了解项目组织
3. 将大目标分解为具体的、可执行的子任务
4. 为每个任务提供清晰的指令

输出格式要求:
- 每个任务必须是独立可执行的
- 任务描述要具体明确
- 包含涉及的文件路径
- 指定任务类型和优先级

请以 JSON 格式输出任务列表:
```json
{
  "analysis": "对代码库的分析总结",
  "tasks": [
    {
      "title": "任务标题",
      "description": "详细描述",
      "instruction": "具体执行指令",
      "type": "implement|refactor|fix|test|document",
      "priority": "low|normal|high|critical",
      "target_files": ["file1.py", "file2.py"],
      "depends_on": []
    }
  ],
  "sub_planners_needed": [
    {
      "area": "需要深入规划的区域",
      "reason": "为什么需要子规划者"
    }
  ]
}
```"""

    def __init__(self, config: PlannerConfig, semantic_search: Optional["SemanticSearch"] = None):
        agent_config = AgentConfig(
            role=AgentRole.PLANNER,
            name=config.name,
            working_directory=config.working_directory,
        )
        super().__init__(agent_config)
        self.planner_config = config

        # 应用 plan 模式配置（--mode=plan 仅分析不修改文件）
        self._apply_plan_mode_config(self.planner_config.cursor_config)
        self._apply_stream_config(self.planner_config.cursor_config)

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

        self.sub_planners: list["PlannerAgent"] = []

        # 语义搜索增强（可选）
        self._semantic_search: Optional["SemanticSearch"] = semantic_search
        self._search_enabled = config.enable_semantic_search and semantic_search is not None
        if self._search_enabled:
            logger.info(f"[{config.name}] 语义搜索已启用")

        logger.debug(f"[{config.name}] 使用执行模式: {config.execution_mode.value}")
        if config.use_plan_mode:
            logger.debug(f"[{config.name}] 已启用 plan 模式（--mode=plan）")

    def _apply_plan_mode_config(self, cursor_config: CursorAgentConfig) -> None:
        """应用 plan 模式配置

        plan 模式特点:
        - 使用 --mode=plan 参数
        - 仅分析和规划，不修改文件
        - 使用 JSON 输出格式便于解析结构化结果
        - 禁用 force_write 以确保不会意外修改文件
        """
        if self.planner_config.use_plan_mode:
            cursor_config.mode = "plan"
            # plan 模式下推荐使用 JSON 输出格式，便于解析结构化的规划结果
            if cursor_config.output_format == "text":
                cursor_config.output_format = "json"
            # 确保不会强制写入文件
            cursor_config.force_write = False

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
        self._search_enabled = self.planner_config.enable_semantic_search
        if self._search_enabled:
            logger.info(f"[{self.id}] 语义搜索已启用")

    async def execute(self, instruction: str, context: Optional[dict] = None) -> dict[str, Any]:
        """执行规划任务

        Args:
            instruction: 用户目标或规划指令
            context: 上下文（包含已有分析结果等）

        Returns:
            规划结果，包含任务列表和子规划者建议
        """
        self.update_status(AgentStatus.RUNNING)
        logger.info(f"[{self.id}] 开始规划: {instruction[:100]}...")

        try:
            # 使用语义搜索探索代码库（可选增强）
            search_context = None
            if self._search_enabled:
                search_context = await self._explore_with_semantic_search(instruction)
                logger.info(f"[{self.id}] 语义搜索找到 {len(search_context.get('related_code', []))} 个相关代码片段")

            # 合并上下文
            merged_context = context.copy() if context else {}
            if search_context:
                merged_context["semantic_search_results"] = search_context

            # 构建规划 prompt
            prompt = self._build_planning_prompt(instruction, merged_context)

            # 调用执行器执行规划
            result = await self._executor.execute(
                prompt=prompt,
                working_directory=self.planner_config.working_directory,
                context=merged_context,
            )

            if not result.success:
                error_detail = result.error or f"exit_code={result.exit_code}, output={result.output[:200] if result.output else 'empty'}"
                logger.error(f"[{self.id}] 规划失败: {error_detail}")
                self.update_status(AgentStatus.FAILED)
                return {"success": False, "error": error_detail, "tasks": []}

            # 解析规划结果
            plan_result = self._parse_planning_result(result.output)

            self.update_status(AgentStatus.COMPLETED)
            logger.info(f"[{self.id}] 规划完成, 生成 {len(plan_result.get('tasks', []))} 个任务")

            return {
                "success": True,
                "analysis": plan_result.get("analysis", ""),
                "tasks": plan_result.get("tasks", []),
                "sub_planners_needed": plan_result.get("sub_planners_needed", []),
                "search_context": search_context,  # 返回搜索结果供后续使用
            }

        except Exception as e:
            logger.exception(f"[{self.id}] 规划异常: {e}")
            self.update_status(AgentStatus.FAILED)
            return {"success": False, "error": str(e), "tasks": []}

    async def _explore_with_semantic_search(self, instruction: str) -> dict[str, Any]:
        """使用语义搜索探索代码库

        Args:
            instruction: 用户目标指令

        Returns:
            搜索结果上下文
        """
        if not self._semantic_search:
            return {}

        try:
            # 执行语义搜索
            results = await self._semantic_search.search(
                query=instruction,
                top_k=self.planner_config.semantic_search_top_k,
                min_score=self.planner_config.semantic_search_min_score,
            )

            # 构建相关代码片段列表
            related_code = []
            for result in results:
                chunk = result.chunk
                related_code.append({
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "name": chunk.name,
                    "chunk_type": chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type),
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "score": result.score,
                })

            return {
                "related_code": related_code,
                "total_found": len(results),
            }

        except Exception as e:
            logger.warning(f"[{self.id}] 语义搜索失败: {e}")
            return {}

    def _build_planning_prompt(self, instruction: str, context: Optional[dict] = None) -> str:
        """构建规划 prompt"""
        parts = [
            self.SYSTEM_PROMPT,
            f"\n## 用户目标\n{instruction}",
        ]

        if context:
            if "codebase_info" in context:
                parts.append(f"\n## 代码库信息\n{context['codebase_info']}")
            if "previous_analysis" in context:
                parts.append(f"\n## 前序分析\n{context['previous_analysis']}")
            if "constraints" in context:
                parts.append(f"\n## 约束条件\n{context['constraints']}")

            # 添加语义搜索结果
            if "semantic_search_results" in context:
                search_results = context["semantic_search_results"]
                related_code = search_results.get("related_code", [])
                if related_code:
                    parts.append(f"\n## 相关代码（语义搜索结果，共 {len(related_code)} 个）")
                    for i, code in enumerate(related_code[:5], 1):  # 只展示前5个
                        parts.append(
                            f"\n### {i}. {code.get('name', '未命名')} ({code['file_path']}:{code['start_line']}-{code['end_line']})"
                            f"\n类型: {code.get('chunk_type', 'unknown')} | 相似度: {code.get('score', 0):.2f}"
                            f"\n```\n{code.get('content_preview', '')}\n```"
                        )
                    if len(related_code) > 5:
                        parts.append(f"\n... 还有 {len(related_code) - 5} 个相关代码片段未显示")

        parts.append("\n请分析并输出任务规划:")

        return "\n".join(parts)

    def _parse_planning_result(self, output: str) -> dict[str, Any]:
        """解析规划结果

        支持多种输出格式:
        1. plan 模式 JSON 输出: {"type": "result", "result": "..."}
        2. 包含 ```json``` 代码块的文本
        3. 直接 JSON 字符串
        4. 纯文本（回退）
        """
        import json
        import re

        content = output

        # 处理 plan 模式的 JSON 输出格式
        # {"type": "result", "subtype": "success", "result": "<规划内容>", ...}
        try:
            outer_json = json.loads(output)
            if isinstance(outer_json, dict) and outer_json.get("type") == "result":
                # 提取 result 字段作为实际规划内容
                content = outer_json.get("result", "")
                logger.debug(f"[{self.id}] 从 plan 模式 JSON 输出中提取规划内容")
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块（Markdown 代码块格式）
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试直接解析为 JSON
        try:
            parsed = json.loads(content)
            # 验证是规划结果格式
            if isinstance(parsed, dict) and ("tasks" in parsed or "analysis" in parsed):
                return parsed
        except json.JSONDecodeError:
            pass

        # 解析失败，返回原始输出作为分析
        logger.warning(f"[{self.id}] 无法解析规划结果为 JSON，使用原始文本")
        return {
            "analysis": content if content else output,
            "tasks": [],
            "sub_planners_needed": [],
        }

    async def spawn_sub_planner(self, area: str, context: Optional[dict] = None) -> "PlannerAgent":
        """派生子规划者

        Args:
            area: 需要深入规划的区域
            context: 上下文信息

        Returns:
            子规划者实例
        """
        if len(self.sub_planners) >= self.planner_config.max_sub_planners:
            logger.warning(f"[{self.id}] 已达到最大子规划者数量限制")
            raise ValueError("已达到最大子规划者数量限制")

        sub_config = PlannerConfig(
            name=f"{self.config.name}-sub-{len(self.sub_planners)}",
            working_directory=self.planner_config.working_directory,
            max_sub_planners=0,  # 子规划者不能再派生
            max_tasks_per_plan=self.planner_config.max_tasks_per_plan,
            cursor_config=self.planner_config.cursor_config,
            # 继承父规划者的执行模式配置
            execution_mode=self.planner_config.execution_mode,
            cloud_auth_config=self.planner_config.cloud_auth_config,
            # 继承父规划者的 plan 模式配置
            use_plan_mode=self.planner_config.use_plan_mode,
            # 继承父规划者的语义搜索配置
            enable_semantic_search=self.planner_config.enable_semantic_search,
            semantic_search_top_k=self.planner_config.semantic_search_top_k,
            semantic_search_min_score=self.planner_config.semantic_search_min_score,
        )

        # 创建子规划者并传递语义搜索实例
        sub_planner = PlannerAgent(sub_config, semantic_search=self._semantic_search)
        sub_planner.set_context("parent_planner", self.id)
        sub_planner.set_context("focus_area", area)
        if context:
            for k, v in context.items():
                sub_planner.set_context(k, v)

        self.sub_planners.append(sub_planner)
        logger.info(f"[{self.id}] 派生子规划者: {sub_planner.id} 负责区域: {area}")

        return sub_planner

    def create_task_from_plan(self, task_data: dict, iteration_id: int) -> Task:
        """从规划结果创建任务"""
        task_type_map = {
            "explore": TaskType.EXPLORE,
            "analyze": TaskType.ANALYZE,
            "implement": TaskType.IMPLEMENT,
            "refactor": TaskType.REFACTOR,
            "fix": TaskType.FIX,
            "test": TaskType.TEST,
            "document": TaskType.DOCUMENT,
        }

        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
        }

        return Task(
            title=task_data.get("title", "未命名任务"),
            description=task_data.get("description", ""),
            instruction=task_data.get("instruction", task_data.get("description", "")),
            type=task_type_map.get(task_data.get("type", ""), TaskType.CUSTOM),
            priority=priority_map.get(task_data.get("priority", "normal"), TaskPriority.NORMAL),
            target_files=task_data.get("target_files", []),
            context=task_data.get("context", {}),
            created_by=self.id,
            iteration_id=iteration_id,
        )

    async def reset(self) -> None:
        """重置规划者状态"""
        self.update_status(AgentStatus.IDLE)
        self.clear_context()
        # 清理子规划者
        for sub in self.sub_planners:
            await sub.reset()
        self.sub_planners.clear()
        logger.debug(f"[{self.id}] 状态已重置")
