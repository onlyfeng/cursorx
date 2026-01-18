"""规划者 Agent 进程

作为独立进程运行的规划者
"""
import asyncio
import json
import re
from multiprocessing import Queue
from typing import Any, Optional

from loguru import logger

from cursor.client import CursorAgentClient, CursorAgentConfig
from process.message_queue import ProcessMessage, ProcessMessageType
from process.worker import AgentWorkerProcess
from tasks.task import Task, TaskPriority, TaskType


class PlannerAgentProcess(AgentWorkerProcess):
    """规划者 Agent 进程

    独立进程中运行的规划者，负责：
    1. 接收规划请求
    2. 调用 Cursor Agent 进行规划
    3. 返回任务列表
    """

    SYSTEM_PROMPT = """你是一个代码项目规划者。

重要：不要编写任何代码，不要编辑任何文件。你只负责分析和规划。

你的职责是:
1. 使用搜索工具探索代码库结构
2. 分析用户的目标需求
3. 将大目标分解为具体的、可执行的子任务
4. 为每个任务提供清晰的指令

你可以使用的工具:
- 文件搜索和读取（了解代码结构）
- Shell 命令（如 ls, find, grep 等探索项目）

Shell 命令限制:
- 命令会在 30 秒后超时
- 每条命令独立执行，目录变更不持久
- 在其他目录运行时使用: cd <dir> && <command>

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
      "target_files": ["file1.py", "file2.py"]
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
        self.current_request_id: Optional[str] = None

    def on_start(self) -> None:
        """进程启动初始化"""
        # 创建 agent CLI 客户端 - 使用 GPT 5.2-high 模型进行规划
        # 不使用 --force，只提议更改而不应用
        stream_enabled = self.config.get("stream_events_enabled", False)
        cursor_config = CursorAgentConfig(
            working_directory=self.config.get("working_directory", "."),
            timeout=self.config.get("timeout", 180),
            model=self.config.get("model", "gpt-5.2-high"),
            output_format="stream-json" if stream_enabled else "json",
            non_interactive=True,
            force_write=False,     # 不修改文件，只分析
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
        logger.info(f"[{self.agent_id}] 规划者初始化完成 (模型: {cursor_config.model}, force: False)")

    def on_stop(self) -> None:
        """进程停止清理"""
        logger.info(f"[{self.agent_id}] 规划者停止")

    def handle_message(self, message: ProcessMessage) -> None:
        """处理业务消息"""
        if message.type == ProcessMessageType.PLAN_REQUEST:
            # 在新的事件循环中执行异步任务
            asyncio.run(self._handle_plan_request(message))

    async def _handle_plan_request(self, message: ProcessMessage) -> None:
        """处理规划请求"""
        self.current_request_id = message.id

        goal = message.payload.get("goal", "")
        context = message.payload.get("context", {})
        iteration_id = message.payload.get("iteration_id", 0)

        logger.info(f"[{self.agent_id}] 收到规划请求: {goal[:50]}...")

        try:
            # 构建规划 prompt
            prompt = self._build_planning_prompt(goal, context)

            # 调用 Cursor Agent
            result = await self.cursor_client.execute(
                instruction=prompt,
                context=context,
            )

            if not result.success:
                self._send_plan_result(
                    success=False,
                    error=result.error,
                    correlation_id=message.id,
                )
                return

            # 解析规划结果
            plan_data = self._parse_planning_result(result.output)

            # 转换为任务列表
            tasks = []
            for task_data in plan_data.get("tasks", []):
                task = self._create_task(task_data, iteration_id)
                tasks.append(task.model_dump())

            self._send_plan_result(
                success=True,
                analysis=plan_data.get("analysis", ""),
                tasks=tasks,
                sub_planners_needed=plan_data.get("sub_planners_needed", []),
                correlation_id=message.id,
            )

            logger.info(f"[{self.agent_id}] 规划完成，生成 {len(tasks)} 个任务")

        except Exception as e:
            logger.exception(f"[{self.agent_id}] 规划异常: {e}")
            self._send_plan_result(
                success=False,
                error=str(e),
                correlation_id=message.id,
            )
        finally:
            self.current_request_id = None

    def _build_planning_prompt(self, goal: str, context: dict) -> str:
        """构建规划 prompt"""
        parts = [
            self.SYSTEM_PROMPT,
            f"\n## 用户目标\n{goal}",
        ]

        if context.get("codebase_info"):
            parts.append(f"\n## 代码库信息\n{context['codebase_info']}")

        if context.get("previous_review"):
            review = context["previous_review"]
            parts.append("\n## 上次评审反馈")
            if review.get("suggestions"):
                parts.append("\n".join(f"- {s}" for s in review["suggestions"]))
            if review.get("next_focus"):
                parts.append(f"\n重点关注: {review['next_focus']}")

        parts.append("\n请分析并输出任务规划:")

        return "\n".join(parts)

    def _parse_planning_result(self, output: str) -> dict[str, Any]:
        """解析规划结果"""
        # 尝试提取 JSON 块
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试直接解析
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

        # 解析失败
        logger.warning(f"[{self.agent_id}] 无法解析规划结果为 JSON")
        return {
            "analysis": output,
            "tasks": [],
            "sub_planners_needed": [],
        }

    def _create_task(self, task_data: dict, iteration_id: int) -> Task:
        """创建任务对象"""
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
            created_by=self.agent_id,
            iteration_id=iteration_id,
        )

    def _send_plan_result(
        self,
        success: bool,
        analysis: str = "",
        tasks: list = None,
        sub_planners_needed: list = None,
        error: str = None,
        correlation_id: str = None,
    ) -> None:
        """发送规划结果"""
        self._send_message(
            ProcessMessageType.PLAN_RESULT,
            {
                "success": success,
                "analysis": analysis,
                "tasks": tasks or [],
                "sub_planners_needed": sub_planners_needed or [],
                "error": error,
            },
            correlation_id=correlation_id,
        )

    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "agent_id": self.agent_id,
            "type": "planner",
            "busy": self.current_request_id is not None,
            "current_request": self.current_request_id,
        }
