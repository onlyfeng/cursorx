"""评审者 Agent

负责评估迭代完成度，决定是否继续
"""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from loguru import logger

from core.base import BaseAgent, AgentConfig, AgentRole, AgentStatus
from cursor.client import CursorAgentClient, CursorAgentConfig


class ReviewDecision(str, Enum):
    """评审决策"""
    CONTINUE = "continue"        # 继续下一轮迭代
    COMPLETE = "complete"        # 目标已完成
    ADJUST = "adjust"            # 需要调整方向
    ABORT = "abort"              # 终止（无法完成）
    COMMIT = "commit"            # 建议提交当前更改
    ROLLBACK = "rollback"        # 建议回退更改


class ReviewerConfig(BaseModel):
    """评审者配置"""
    name: str = "reviewer"
    working_directory: str = "."
    strict_mode: bool = False          # 严格模式（更高的完成标准）
    cursor_config: CursorAgentConfig = Field(default_factory=CursorAgentConfig)


class ReviewerAgent(BaseAgent):
    """评审者 Agent
    
    职责:
    1. 评估迭代完成情况
    2. 判断是否达成目标
    3. 决定是否继续迭代
    4. 提供改进建议
    """
    
    # 评审者系统提示
    SYSTEM_PROMPT = """你是一个代码评审者。你的职责是:

1. 评估当前迭代的完成情况
2. 判断是否达成用户目标
3. 决定下一步行动
4. 提供具体的改进建议
5. 评估是否应该提交或回退更改

评审维度:
- 功能完整性: 是否完成了所有要求的功能
- 代码质量: 代码是否规范、可维护
- 测试覆盖: 是否有适当的测试
- 文档完善: 是否有必要的注释和文档

决策类型:
- continue: 继续下一轮迭代
- complete: 目标已完成
- adjust: 需要调整方向
- abort: 终止（无法完成）
- commit: 建议提交当前更改（代码质量良好，可以提交）
- rollback: 建议回退更改（代码存在严重问题，需要回退）

请以 JSON 格式输出评审结果:
```json
{
  "decision": "continue|complete|adjust|abort|commit|rollback",
  "score": 0-100,
  "summary": "评审总结",
  "completed_items": ["已完成项1", "已完成项2"],
  "pending_items": ["待完成项1", "待完成项2"],
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "next_iteration_focus": "下一轮迭代的重点（如果 decision 是 continue）",
  "commit_suggestion": {
    "should_commit": true|false,
    "commit_message": "建议的提交信息",
    "files_to_commit": ["file1.py", "file2.py"],
    "reason": "提交或不提交的原因"
  }
}
```"""
    
    def __init__(self, config: ReviewerConfig):
        agent_config = AgentConfig(
            role=AgentRole.REVIEWER,
            name=config.name,
            working_directory=config.working_directory,
        )
        super().__init__(agent_config)
        self.reviewer_config = config
        self._apply_stream_config(self.reviewer_config.cursor_config)
        self.cursor_client = CursorAgentClient(config.cursor_config)
        self.review_history: list[dict] = []
    
    async def execute(self, instruction: str, context: Optional[dict] = None) -> dict[str, Any]:
        """执行评审
        
        Args:
            instruction: 评审指令（通常是用户目标）
            context: 上下文（包含迭代信息、任务完成情况等）
            
        Returns:
            评审结果
        """
        self.update_status(AgentStatus.RUNNING)
        logger.info(f"[{self.id}] 开始评审...")
        
        try:
            # 构建评审 prompt
            prompt = self._build_review_prompt(instruction, context)
            
            # 调用 Cursor Agent 执行评审
            result = await self.cursor_client.execute(
                instruction=prompt,
                working_directory=self.reviewer_config.working_directory,
                context=context,
            )
            
            if not result.success:
                logger.error(f"[{self.id}] 评审执行失败: {result.error}")
                self.update_status(AgentStatus.FAILED)
                return {
                    "success": False, 
                    "error": result.error,
                    "decision": ReviewDecision.CONTINUE,
                }
            
            # 解析评审结果
            review_result = self._parse_review_result(result.output)
            
            # 记录评审历史
            self.review_history.append(review_result)
            
            self.update_status(AgentStatus.COMPLETED)
            logger.info(f"[{self.id}] 评审完成: {review_result.get('decision', 'unknown')}")
            
            return {
                "success": True,
                **review_result,
            }
            
        except Exception as e:
            logger.exception(f"[{self.id}] 评审异常: {e}")
            self.update_status(AgentStatus.FAILED)
            return {
                "success": False, 
                "error": str(e),
                "decision": ReviewDecision.CONTINUE,
            }
    
    async def review_iteration(
        self,
        goal: str,
        iteration_id: int,
        tasks_completed: list[dict],
        tasks_failed: list[dict],
        previous_reviews: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """评审一次迭代
        
        Args:
            goal: 用户目标
            iteration_id: 迭代 ID
            tasks_completed: 已完成的任务列表
            tasks_failed: 失败的任务列表
            previous_reviews: 之前的评审结果
            
        Returns:
            评审结果
        """
        context = {
            "iteration_id": iteration_id,
            "tasks_completed": tasks_completed,
            "tasks_failed": tasks_failed,
            "completion_rate": len(tasks_completed) / max(len(tasks_completed) + len(tasks_failed), 1),
            "previous_reviews": previous_reviews or self.review_history,
        }
        
        return await self.execute(goal, context)
    
    def _build_review_prompt(self, instruction: str, context: Optional[dict] = None) -> str:
        """构建评审 prompt"""
        parts = [
            self.SYSTEM_PROMPT,
            f"\n## 用户目标\n{instruction}",
        ]
        
        if context:
            if "iteration_id" in context:
                parts.append(f"\n## 当前迭代\n第 {context['iteration_id']} 轮迭代")
            
            if "tasks_completed" in context:
                completed = context["tasks_completed"]
                if completed:
                    tasks_str = "\n".join(f"- {t.get('title', t.get('id', 'unknown'))}: {t.get('result', {}).get('output', '')[:100]}" 
                                         for t in completed)
                    parts.append(f"\n## 已完成任务 ({len(completed)} 个)\n{tasks_str}")
                else:
                    parts.append("\n## 已完成任务\n无")
            
            if "tasks_failed" in context:
                failed = context["tasks_failed"]
                if failed:
                    tasks_str = "\n".join(f"- {t.get('title', t.get('id', 'unknown'))}: {t.get('error', '未知错误')}" 
                                         for t in failed)
                    parts.append(f"\n## 失败任务 ({len(failed)} 个)\n{tasks_str}")
            
            if "completion_rate" in context:
                parts.append(f"\n## 完成率\n{context['completion_rate']:.1%}")
            
            if "previous_reviews" in context and context["previous_reviews"]:
                prev = context["previous_reviews"][-1]  # 只看上一次评审
                parts.append(f"\n## 上次评审\n- 决策: {prev.get('decision', 'N/A')}\n- 得分: {prev.get('score', 'N/A')}")
        
        if self.reviewer_config.strict_mode:
            parts.append("\n## 评审模式\n严格模式：请使用更高的标准进行评审")
        
        parts.append("\n请进行评审并输出结果:")
        
        return "\n".join(parts)
    
    def _parse_review_result(self, output: str) -> dict[str, Any]:
        """解析评审结果"""
        import json
        import re
        
        # 尝试提取 JSON 块
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                # 转换 decision 为枚举
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
        
        # 解析失败，返回默认继续
        logger.warning(f"[{self.id}] 无法解析评审结果，默认继续迭代")
        return {
            "decision": ReviewDecision.CONTINUE,
            "score": 50,
            "summary": output,
            "suggestions": ["无法解析评审结果，建议继续迭代"],
        }

    def _apply_stream_config(self, cursor_config: CursorAgentConfig) -> None:
        """注入流式日志配置与 Agent 标识"""
        cursor_config.stream_agent_id = self.id
        cursor_config.stream_agent_role = self.role.value
        cursor_config.stream_agent_name = self.name
        if cursor_config.stream_events_enabled:
            cursor_config.output_format = "stream-json"
            cursor_config.stream_partial_output = True
    
    async def reset(self) -> None:
        """重置评审者状态"""
        self.update_status(AgentStatus.IDLE)
        self.clear_context()
        # 不清除 review_history，保留评审历史
        logger.debug(f"[{self.id}] 状态已重置")
    
    def get_review_summary(self) -> dict[str, Any]:
        """获取评审总结"""
        if not self.review_history:
            return {"total_reviews": 0}
        
        scores = [r.get("score", 0) for r in self.review_history if "score" in r]
        decisions = [r.get("decision") for r in self.review_history if "decision" in r]
        
        return {
            "total_reviews": len(self.review_history),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "latest_decision": decisions[-1] if decisions else None,
            "decision_history": [d.value if isinstance(d, ReviewDecision) else d for d in decisions],
        }
