"""评审者 Agent 进程

作为独立进程运行的评审者
"""

import asyncio
import json
import re
from enum import Enum
from multiprocessing import Queue
from typing import Any, Optional

from loguru import logger

from core.config import DEFAULT_REVIEW_TIMEOUT, DEFAULT_REVIEWER_MODEL
from cursor.client import CursorAgentClient, CursorAgentConfig
from process.message_queue import ProcessMessage, ProcessMessageType
from process.worker import AgentWorkerProcess


class ReviewDecision(str, Enum):
    """评审决策"""

    CONTINUE = "continue"
    COMPLETE = "complete"
    ADJUST = "adjust"
    ABORT = "abort"


class ReviewerAgentProcess(AgentWorkerProcess):
    """评审者 Agent 进程

    独立进程中运行的评审者，负责：
    1. 评估迭代完成情况
    2. 判断是否达成目标
    3. 决定下一步行动
    """

    SYSTEM_PROMPT = """你是一个代码评审者。

重要：不要编写任何代码，不要编辑任何文件。你只负责评审和提供建议。

你的职责是:
1. 使用文件读取工具检查代码变更
2. 评估当前迭代的完成情况
3. 判断是否达成用户目标
4. 决定下一步行动
5. 提供具体的改进建议

你可以使用的工具:
- 文件搜索和读取（检查代码）
- Shell 命令（如 git diff, git status 等查看变更）

评审维度:
- 功能完整性: 是否完成了所有要求的功能
- 代码质量: 代码是否规范、可维护
- 测试覆盖: 是否有适当的测试
- 文档完善: 是否有必要的注释和文档

请以 JSON 格式输出评审结果:
```json
{
  "decision": "continue|complete|adjust|abort",
  "score": 0-100,
  "summary": "评审总结",
  "completed_items": ["已完成项1", "已完成项2"],
  "pending_items": ["待完成项1", "待完成项2"],
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "next_iteration_focus": "下一轮迭代的重点"
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
        self.review_history: list[dict] = []
        self.current_request_id: Optional[str] = None

    def on_start(self) -> None:
        """进程启动初始化"""
        # 创建 agent CLI 客户端 - 使用 DEFAULT_REVIEWER_MODEL 进行评审
        # 不使用 --force，只评审而不修改
        stream_enabled = self.config.get("stream_events_enabled", False)
        cursor_config = CursorAgentConfig(
            working_directory=self.config.get("working_directory", "."),
            timeout=self.config.get("timeout", int(DEFAULT_REVIEW_TIMEOUT)),
            model=self.config.get("model", DEFAULT_REVIEWER_MODEL),
            output_format="stream-json" if stream_enabled else "json",
            non_interactive=True,
            force_write=False,  # 不修改文件，只评审
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
        logger.info(f"[{self.agent_id}] 评审者初始化完成 (模型: {cursor_config.model}, force: False)")

    def on_stop(self) -> None:
        """进程停止清理"""
        logger.info(f"[{self.agent_id}] 评审者停止，完成 {len(self.review_history)} 次评审")

    def handle_message(self, message: ProcessMessage) -> None:
        """处理业务消息"""
        if message.type == ProcessMessageType.REVIEW_REQUEST:
            asyncio.run(self._handle_review_request(message))

    async def _handle_review_request(self, message: ProcessMessage) -> None:
        """处理评审请求"""
        self.current_request_id = message.id

        goal = message.payload.get("goal", "")
        iteration_id = message.payload.get("iteration_id", 0)
        tasks_completed = message.payload.get("tasks_completed", [])
        tasks_failed = message.payload.get("tasks_failed", [])
        extra_context = message.payload.get("context") or {}

        logger.info(f"[{self.agent_id}] 收到评审请求: 迭代 {iteration_id}")

        try:
            # 构建评审 prompt
            prompt = self._build_review_prompt(goal, iteration_id, tasks_completed, tasks_failed, extra_context)

            # 调用 Cursor Agent
            assert self.cursor_client is not None
            result = await self.cursor_client.execute(instruction=prompt)

            if not result.success:
                self._send_review_result(
                    success=False,
                    error=str(result.error) if result.error else None,
                    decision=ReviewDecision.CONTINUE,
                    correlation_id=message.id,
                )
                return

            # 解析评审结果
            review_data = self._parse_review_result(result.output)

            # 记录历史
            self.review_history.append(review_data)

            self._send_review_result(
                success=True,
                **review_data,
                correlation_id=message.id,
            )

            logger.info(f"[{self.agent_id}] 评审完成: {review_data.get('decision', 'unknown')}")

        except Exception as e:
            logger.exception(f"[{self.agent_id}] 评审异常: {e}")
            self._send_review_result(
                success=False,
                error=str(e),
                decision=ReviewDecision.CONTINUE,
                correlation_id=message.id,
            )
        finally:
            self.current_request_id = None

    def _build_review_prompt(
        self,
        goal: str,
        iteration_id: int,
        tasks_completed: list,
        tasks_failed: list,
        extra_context: Optional[dict] = None,
    ) -> str:
        """构建评审 prompt"""
        parts = [
            self.SYSTEM_PROMPT,
            f"\n## 用户目标\n{goal}",
            f"\n## 当前迭代\n第 {iteration_id} 轮迭代",
        ]

        if tasks_completed:
            tasks_str = "\n".join(f"- {t.get('title', t.get('id', 'unknown'))}" for t in tasks_completed)
            parts.append(f"\n## 已完成任务 ({len(tasks_completed)} 个)\n{tasks_str}")
        else:
            parts.append("\n## 已完成任务\n无")

        if tasks_failed:
            tasks_str = "\n".join(
                f"- {t.get('title', t.get('id', 'unknown'))}: {t.get('error', '未知错误')}" for t in tasks_failed
            )
            parts.append(f"\n## 失败任务 ({len(tasks_failed)} 个)\n{tasks_str}")

        total = len(tasks_completed) + len(tasks_failed)
        if total > 0:
            rate = len(tasks_completed) / total
            parts.append(f"\n## 完成率\n{rate:.1%}")

        if self.review_history:
            last = self.review_history[-1]
            parts.append(f"\n## 上次评审\n- 决策: {last.get('decision', 'N/A')}\n- 得分: {last.get('score', 'N/A')}")

        if extra_context and extra_context.get("iteration_assistant"):
            parts.append(
                "\n## 迭代上下文（.iteration / Engram / 规则）\n"
                f"```json\n{json.dumps(extra_context['iteration_assistant'], ensure_ascii=False, indent=2)}\n```"
            )

        if self.config.get("strict_mode"):
            parts.append("\n## 评审模式\n严格模式：请使用更高的标准进行评审")

        parts.append("\n请进行评审并输出结果:")

        return "\n".join(parts)

    def _parse_review_result(self, output: str) -> dict[str, Any]:
        """解析评审结果"""
        # 尝试提取 JSON 块
        json_match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if "decision" in result:
                    result["decision"] = ReviewDecision(result["decision"])
                return result
            except (json.JSONDecodeError, ValueError):
                pass

        # 尝试直接解析
        try:
            result = json.loads(output)
            if "decision" in result:
                result["decision"] = ReviewDecision(result["decision"])
            return result
        except (json.JSONDecodeError, ValueError):
            pass

        # 解析失败
        logger.warning(f"[{self.agent_id}] 无法解析评审结果")
        return {
            "decision": ReviewDecision.CONTINUE,
            "score": 50,
            "summary": output,
            "suggestions": ["无法解析评审结果，建议继续迭代"],
        }

    def _send_review_result(
        self,
        success: bool,
        decision: ReviewDecision = ReviewDecision.CONTINUE,
        score: int = 0,
        summary: str = "",
        completed_items: list[Any] | None = None,
        pending_items: list[Any] | None = None,
        issues: list[Any] | None = None,
        suggestions: list[Any] | None = None,
        next_iteration_focus: str = "",
        error: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """发送评审结果"""
        self._send_message(
            ProcessMessageType.REVIEW_RESULT,
            {
                "success": success,
                "decision": decision.value if isinstance(decision, ReviewDecision) else decision,
                "score": score,
                "summary": summary,
                "completed_items": completed_items or [],
                "pending_items": pending_items or [],
                "issues": issues or [],
                "suggestions": suggestions or [],
                "next_iteration_focus": next_iteration_focus,
                "error": error,
            },
            correlation_id=correlation_id,
        )

    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "agent_id": self.agent_id,
            "type": "reviewer",
            "busy": self.current_request_id is not None,
            "total_reviews": len(self.review_history),
        }
