"""测试 Agent 行为方法

测试各 Agent 的核心异步方法：
- PlannerAgent.execute() - 规划任务
- WorkerAgent.execute() - 执行任务
- ReviewerAgent.execute() / review_iteration() - 评审任务

使用 mock 模拟 Cursor CLI 调用，避免实际执行 CLI 命令
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from agents.planner import PlannerAgent, PlannerConfig
from agents.reviewer import ReviewDecision, ReviewerAgent, ReviewerConfig
from agents.worker import WorkerAgent, WorkerConfig
from core.base import AgentStatus
from cursor.executor import AgentResult

# ========== 测试辅助函数 ==========


def create_mock_agent_result(
    success: bool = True,
    output: str = "",
    error: str | None = None,
    exit_code: int = 0,
    duration: float = 1.0,
) -> AgentResult:
    """创建模拟的 AgentResult"""
    return AgentResult(
        success=success,
        output=output,
        error=error,
        exit_code=exit_code,
        duration=duration,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        executor_type="cli",
    )


def create_planner_json_output() -> str:
    """创建规划者的模拟 JSON 输出"""
    plan_data = {
        "analysis": "代码库分析完成，发现需要重构的模块",
        "tasks": [
            {
                "title": "重构用户模块",
                "description": "将用户相关逻辑移至独立模块",
                "instruction": "创建 user 模块并迁移相关代码",
                "type": "refactor",
                "priority": "high",
                "target_files": ["src/user.py", "src/auth.py"],
                "depends_on": [],
            },
            {
                "title": "添加单元测试",
                "description": "为新模块添加测试",
                "instruction": "编写 user 模块的单元测试",
                "type": "test",
                "priority": "normal",
                "target_files": ["tests/test_user.py"],
                "depends_on": ["重构用户模块"],
            },
        ],
        "sub_planners_needed": [],
    }
    return f"```json\n{json.dumps(plan_data, ensure_ascii=False)}\n```"


def create_reviewer_json_output(decision: str = "continue", score: int = 75) -> str:
    """创建评审者的模拟 JSON 输出"""
    review_data = {
        "decision": decision,
        "score": score,
        "summary": "本轮迭代完成了主要功能，但仍有待改进的地方",
        "completed_items": ["用户模块重构", "基本测试覆盖"],
        "pending_items": ["性能优化", "文档更新"],
        "issues": ["部分代码缺少注释"],
        "suggestions": ["建议添加更多边界测试"],
        "next_iteration_focus": "补充测试用例和文档",
        "commit_suggestion": {
            "should_commit": True,
            "commit_message": "feat: 重构用户模块",
            "files_to_commit": ["src/user.py"],
            "reason": "代码质量良好，可以提交",
        },
    }
    return f"```json\n{json.dumps(review_data, ensure_ascii=False)}\n```"


# ========== PlannerAgent 测试 ==========


class TestPlannerAgentExecute:
    """测试 PlannerAgent.execute() 方法"""

    @pytest.fixture
    def planner(self) -> PlannerAgent:
        """创建 PlannerAgent 实例"""
        config = PlannerConfig(name="test-planner")
        return PlannerAgent(config)

    @pytest.mark.asyncio
    async def test_execute_success(self, planner: PlannerAgent):
        """测试成功执行规划任务"""
        # Mock 执行器返回成功结果
        mock_result = create_mock_agent_result(
            success=True,
            output=create_planner_json_output(),
        )

        with patch.object(planner._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await planner.execute("重构用户模块")

            # 验证执行器被调用
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args

            # 验证 prompt 包含指令
            assert "重构用户模块" in call_args.kwargs["prompt"]

            # 验证结果
            assert result["success"] is True
            assert len(result["tasks"]) == 2
            assert result["tasks"][0]["title"] == "重构用户模块"
            assert result["tasks"][1]["title"] == "添加单元测试"
            assert "分析" in result["analysis"] or "analysis" in result

    @pytest.mark.asyncio
    async def test_execute_with_context(self, planner: PlannerAgent):
        """测试带上下文的规划任务执行"""
        mock_result = create_mock_agent_result(
            success=True,
            output=create_planner_json_output(),
        )

        context = {
            "codebase_info": "Python 项目，使用 FastAPI 框架",
            "constraints": "保持向后兼容",
        }

        with patch.object(planner._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await planner.execute("优化 API 性能", context=context)

            # 验证上下文被传递
            call_args = mock_execute.call_args
            prompt = call_args.kwargs["prompt"]

            # prompt 应包含上下文信息
            assert "Python 项目" in prompt or "FastAPI" in prompt
            assert "向后兼容" in prompt

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_failure(self, planner: PlannerAgent):
        """测试规划任务执行失败"""
        mock_result = create_mock_agent_result(
            success=False,
            error="CLI 执行超时",
            exit_code=1,
        )

        with patch.object(planner._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await planner.execute("复杂的重构任务")

            # 验证失败结果
            assert result["success"] is False
            assert "error" in result
            assert result["tasks"] == []

    @pytest.mark.asyncio
    async def test_execute_invalid_json_output(self, planner: PlannerAgent):
        """测试规划结果 JSON 解析失败时的回退处理"""
        # 返回无法解析的输出
        mock_result = create_mock_agent_result(
            success=True,
            output="这是一段普通文本，不是 JSON 格式的规划结果。",
        )

        with patch.object(planner._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await planner.execute("生成规划")

            # 验证回退处理
            assert result["success"] is True
            assert result["tasks"] == []  # 无法解析，任务列表为空
            # 原始文本应作为 analysis 返回
            assert "普通文本" in result.get("analysis", "")

    @pytest.mark.asyncio
    async def test_execute_updates_status(self, planner: PlannerAgent):
        """测试执行过程中状态更新"""
        mock_result = create_mock_agent_result(
            success=True,
            output=create_planner_json_output(),
        )

        with patch.object(planner._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            # 执行前应为 IDLE
            assert planner.status == AgentStatus.IDLE

            await planner.execute("规划任务")

            # 成功后应为 COMPLETED
            assert planner.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, planner: PlannerAgent):
        """测试执行过程中异常处理"""
        with patch.object(planner._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("网络连接失败")

            result = await planner.execute("规划任务")

            # 验证异常被捕获
            assert result["success"] is False
            assert "网络连接失败" in result["error"]
            assert planner.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_plan_mode_json_output(self, planner: PlannerAgent):
        """测试 plan 模式下的 JSON 包装输出解析"""
        # plan 模式返回的是包装后的 JSON
        plan_content = {
            "analysis": "代码分析",
            "tasks": [{"title": "任务1", "type": "implement"}],
            "sub_planners_needed": [],
        }
        plan_output = {
            "type": "result",
            "subtype": "success",
            "result": f"```json\n{json.dumps(plan_content, ensure_ascii=False)}\n```",
        }

        mock_result = create_mock_agent_result(
            success=True,
            output=json.dumps(plan_output),
        )

        with patch.object(planner._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await planner.execute("分析代码")

            assert result["success"] is True
            assert len(result["tasks"]) == 1
            assert result["tasks"][0]["title"] == "任务1"


# ========== WorkerAgent 测试 ==========


class TestWorkerAgentExecute:
    """测试 WorkerAgent.execute() 方法"""

    @pytest.fixture
    def worker(self) -> WorkerAgent:
        """创建 WorkerAgent 实例"""
        config = WorkerConfig(name="test-worker")
        return WorkerAgent(config)

    @pytest.mark.asyncio
    async def test_execute_success(self, worker: WorkerAgent):
        """测试成功执行任务"""
        mock_result = create_mock_agent_result(
            success=True,
            output="任务执行成功，已修改 3 个文件",
            duration=5.5,
        )

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("修复登录 bug")

            # 验证执行器被调用
            mock_execute.assert_called_once()

            # 验证结果
            assert result["success"] is True
            assert "修改" in result["output"]
            assert result["duration"] == 5.5

    @pytest.mark.asyncio
    async def test_execute_with_context(self, worker: WorkerAgent):
        """测试带上下文的任务执行"""
        mock_result = create_mock_agent_result(
            success=True,
            output="已完成代码修改",
        )

        context = {
            "task_id": "task-001",
            "task_type": "fix",
            "target_files": ["src/auth.py", "src/login.py"],
        }

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("修复认证问题", context=context)

            # 验证 prompt 包含上下文
            call_args = mock_execute.call_args
            prompt = call_args.kwargs["prompt"]

            assert "修复认证问题" in prompt
            # target_files 应该被格式化到 prompt 中
            assert "src/auth.py" in prompt or "auth.py" in prompt

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_failure(self, worker: WorkerAgent):
        """测试任务执行失败"""
        mock_result = create_mock_agent_result(
            success=False,
            error="编译错误：语法错误",
            exit_code=1,
        )

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("重构代码")

            # 验证失败结果
            assert result["success"] is False
            assert "error" in result
            assert worker.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_updates_status(self, worker: WorkerAgent):
        """测试执行过程中状态更新"""
        mock_result = create_mock_agent_result(success=True, output="完成")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            assert worker.status == AgentStatus.IDLE

            await worker.execute("执行任务")

            assert worker.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, worker: WorkerAgent):
        """测试执行过程中异常处理"""
        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("执行器内部错误")

            result = await worker.execute("任务")

            assert result["success"] is False
            assert "执行器内部错误" in result["error"]
            assert worker.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, worker: WorkerAgent):
        """测试带超时的任务执行"""
        mock_result = create_mock_agent_result(success=True, output="完成")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            await worker.execute("长时间任务")

            # 验证超时参数被传递
            call_args = mock_execute.call_args
            assert call_args.kwargs["timeout"] == worker.worker_config.task_timeout

    # ========== 边界条件测试 ==========

    @pytest.mark.asyncio
    async def test_execute_empty_instruction(self, worker: WorkerAgent):
        """测试空指令执行"""
        mock_result = create_mock_agent_result(success=True, output="无操作")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("")

            # 空指令也应该正常处理
            mock_execute.assert_called_once()
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_very_long_instruction(self, worker: WorkerAgent):
        """测试超长指令执行"""
        long_instruction = "重构代码" * 1000  # 超长指令
        mock_result = create_mock_agent_result(success=True, output="完成超长任务")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute(long_instruction)

            # 超长指令应正常处理
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert long_instruction in call_args.kwargs["prompt"]
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_empty_context(self, worker: WorkerAgent):
        """测试空上下文字典执行"""
        mock_result = create_mock_agent_result(success=True, output="完成")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("任务", context={})

            mock_execute.assert_called_once()
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_none_context(self, worker: WorkerAgent):
        """测试 None 上下文执行"""
        mock_result = create_mock_agent_result(success=True, output="完成")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("任务", context=None)

            mock_execute.assert_called_once()
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_empty_target_files(self, worker: WorkerAgent):
        """测试目标文件为空列表"""
        mock_result = create_mock_agent_result(success=True, output="完成")

        context = {
            "task_id": "task-001",
            "target_files": [],  # 空文件列表
        }

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("修改代码", context=context)

            # 空文件列表应正常处理
            mock_execute.assert_called_once()
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_special_characters_in_instruction(self, worker: WorkerAgent):
        """测试指令包含特殊字符"""
        special_instruction = "修复 bug: $HOME/path && rm -rf / | grep 'test' < input > output"
        mock_result = create_mock_agent_result(success=True, output="完成")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute(special_instruction)

            # 特殊字符应正常传递
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert special_instruction in call_args.kwargs["prompt"]
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_unicode_content(self, worker: WorkerAgent):
        """测试包含 Unicode 字符的指令"""
        unicode_instruction = "处理多语言文件: 你好世界 🌍 日本語 العربية"
        mock_result = create_mock_agent_result(success=True, output="处理完成 ✓")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute(unicode_instruction)

            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert unicode_instruction in call_args.kwargs["prompt"]
            assert result["success"] is True
            assert "处理完成" in result["output"]

    @pytest.mark.asyncio
    async def test_execute_with_empty_output(self, worker: WorkerAgent):
        """测试执行成功但输出为空"""
        mock_result = create_mock_agent_result(success=True, output="")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("静默任务")

            assert result["success"] is True
            assert result["output"] == ""

    @pytest.mark.asyncio
    async def test_execute_with_none_error(self, worker: WorkerAgent):
        """测试失败但 error 为 None"""
        mock_result = create_mock_agent_result(
            success=False,
            error=None,
            exit_code=1,
        )

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("失败任务")

            assert result["success"] is False
            assert worker.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_multiple_sequential(self, worker: WorkerAgent):
        """测试连续多次执行任务"""
        results = [
            create_mock_agent_result(success=True, output="第一次完成"),
            create_mock_agent_result(success=True, output="第二次完成"),
            create_mock_agent_result(success=False, error="第三次失败"),
        ]

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = results

            result1 = await worker.execute("任务1")
            assert result1["success"] is True
            assert worker.status == AgentStatus.COMPLETED

            result2 = await worker.execute("任务2")
            assert result2["success"] is True
            assert worker.status == AgentStatus.COMPLETED

            result3 = await worker.execute("任务3")
            assert result3["success"] is False
            assert worker.status == AgentStatus.FAILED

            assert mock_execute.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_nested_context(self, worker: WorkerAgent):
        """测试嵌套的复杂上下文"""
        complex_context = {
            "task_id": "task-001",
            "target_files": ["src/main.py"],
            "metadata": {
                "nested": {
                    "deep": {"value": 123},
                },
                "list": [1, 2, {"inner": "data"}],
            },
        }
        mock_result = create_mock_agent_result(success=True, output="完成")

        with patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await worker.execute("复杂任务", context=complex_context)

            mock_execute.assert_called_once()
            # 验证上下文被传递
            call_args = mock_execute.call_args
            assert call_args.kwargs["context"] == complex_context
            assert result["success"] is True


# ========== ReviewerAgent 测试 ==========


class TestReviewerAgentExecute:
    """测试 ReviewerAgent.execute() 方法"""

    @pytest.fixture
    def reviewer(self) -> ReviewerAgent:
        """创建 ReviewerAgent 实例"""
        config = ReviewerConfig(name="test-reviewer")
        return ReviewerAgent(config)

    @pytest.mark.asyncio
    async def test_execute_success(self, reviewer: ReviewerAgent):
        """测试成功执行评审"""
        mock_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(decision="continue", score=75),
        )

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await reviewer.execute("评审重构进度")

            # 验证执行器被调用
            mock_execute.assert_called_once()

            # 验证结果
            assert result["success"] is True
            assert result["decision"] == ReviewDecision.CONTINUE
            assert result["score"] == 75
            assert "summary" in result

    @pytest.mark.asyncio
    async def test_execute_complete_decision(self, reviewer: ReviewerAgent):
        """测试评审决策为完成"""
        mock_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(decision="complete", score=95),
        )

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await reviewer.execute("评审最终结果")

            assert result["decision"] == ReviewDecision.COMPLETE
            assert result["score"] == 95

    @pytest.mark.asyncio
    async def test_execute_with_context(self, reviewer: ReviewerAgent):
        """测试带上下文的评审执行"""
        mock_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(),
        )

        context = {
            "iteration_id": 3,
            "tasks_completed": [{"id": "task-1", "title": "重构完成"}],
            "tasks_failed": [],
            "completion_rate": 1.0,
        }

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await reviewer.execute("评审第三轮迭代", context=context)

            # 验证 prompt 包含上下文
            call_args = mock_execute.call_args
            prompt = call_args.kwargs["prompt"]

            assert "第 3 轮" in prompt or "iteration" in prompt.lower()

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_failure(self, reviewer: ReviewerAgent):
        """测试评审执行失败"""
        mock_result = create_mock_agent_result(
            success=False,
            error="API 调用失败",
            exit_code=1,
        )

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await reviewer.execute("评审任务")

            assert result["success"] is False
            assert "error" in result
            # 失败时应有默认决策
            assert result["decision"] == ReviewDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_execute_invalid_json_fallback(self, reviewer: ReviewerAgent):
        """测试评审结果 JSON 解析失败时的回退处理"""
        mock_result = create_mock_agent_result(
            success=True,
            output="评审完成，代码质量良好，建议继续。",
        )

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await reviewer.execute("评审")

            # 回退处理：默认继续迭代
            assert result["success"] is True
            assert result["decision"] == ReviewDecision.CONTINUE
            assert "summary" in result

    @pytest.mark.asyncio
    async def test_execute_updates_review_history(self, reviewer: ReviewerAgent):
        """测试评审结果被记录到历史"""
        mock_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(decision="continue", score=80),
        )

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            assert len(reviewer.review_history) == 0

            await reviewer.execute("评审")

            assert len(reviewer.review_history) == 1
            assert reviewer.review_history[0]["score"] == 80

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, reviewer: ReviewerAgent):
        """测试执行过程中异常处理"""
        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("评审器异常")

            result = await reviewer.execute("评审")

            assert result["success"] is False
            assert "评审器异常" in result["error"]
            assert reviewer.status == AgentStatus.FAILED


class TestReviewerAgentReviewIteration:
    """测试 ReviewerAgent.review_iteration() 方法"""

    @pytest.fixture
    def reviewer(self) -> ReviewerAgent:
        """创建 ReviewerAgent 实例"""
        config = ReviewerConfig(name="test-reviewer")
        return ReviewerAgent(config)

    @pytest.mark.asyncio
    async def test_review_iteration_success(self, reviewer: ReviewerAgent):
        """测试成功评审迭代"""
        mock_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(decision="continue", score=70),
        )

        tasks_completed = [
            {"id": "task-1", "title": "实现功能 A", "result": {"output": "完成"}},
            {"id": "task-2", "title": "修复 bug B", "result": {"output": "完成"}},
        ]
        tasks_failed = [
            {"id": "task-3", "title": "优化性能", "error": "超时"},
        ]

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await reviewer.review_iteration(
                goal="完成用户模块重构",
                iteration_id=2,
                tasks_completed=tasks_completed,
                tasks_failed=tasks_failed,
            )

            # 验证 prompt 包含迭代信息
            call_args = mock_execute.call_args
            prompt = call_args.kwargs["prompt"]

            assert "完成用户模块重构" in prompt
            assert "第 2 轮" in prompt or "iteration" in prompt.lower()
            assert "2 个" in prompt or "已完成" in prompt  # 完成任务数
            assert "1 个" in prompt or "失败" in prompt  # 失败任务数

            assert result["success"] is True
            assert result["decision"] == ReviewDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_review_iteration_with_previous_reviews(self, reviewer: ReviewerAgent):
        """测试带历史评审记录的迭代评审"""
        mock_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(decision="complete", score=90),
        )

        previous_reviews = [
            {"decision": ReviewDecision.CONTINUE, "score": 50, "summary": "初步完成"},
            {"decision": ReviewDecision.CONTINUE, "score": 70, "summary": "进展良好"},
        ]

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            result = await reviewer.review_iteration(
                goal="完成项目",
                iteration_id=3,
                tasks_completed=[{"id": "task-1", "title": "最终任务"}],
                tasks_failed=[],
                previous_reviews=previous_reviews,
            )

            # 验证包含历史评审信息
            call_args = mock_execute.call_args
            prompt = call_args.kwargs["prompt"]

            # 应包含上次评审的决策和得分
            assert "上次评审" in prompt or "previous" in prompt.lower()

            assert result["decision"] == ReviewDecision.COMPLETE


class TestReviewerAgentStrict:
    """测试 ReviewerAgent 严格模式"""

    @pytest.mark.asyncio
    async def test_strict_mode_in_prompt(self):
        """测试严格模式在 prompt 中体现"""
        config = ReviewerConfig(name="strict-reviewer", strict_mode=True)
        reviewer = ReviewerAgent(config)

        mock_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(),
        )

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_result

            await reviewer.execute("评审代码")

            call_args = mock_execute.call_args
            prompt = call_args.kwargs["prompt"]

            # 严格模式应在 prompt 中体现
            assert "严格" in prompt or "strict" in prompt.lower()


# ========== 集成场景测试 ==========


class TestAgentBehaviorIntegration:
    """测试多个 Agent 协作场景"""

    @pytest.mark.asyncio
    async def test_planner_worker_reviewer_flow(self):
        """测试规划者 -> 执行者 -> 评审者完整流程"""
        # 创建 Agents
        planner_config = PlannerConfig(name="flow-planner")
        worker_config = WorkerConfig(name="flow-worker")
        reviewer_config = ReviewerConfig(name="flow-reviewer")

        planner = PlannerAgent(planner_config)
        worker = WorkerAgent(worker_config)
        reviewer = ReviewerAgent(reviewer_config)

        # Mock 执行器
        planner_result = create_mock_agent_result(
            success=True,
            output=create_planner_json_output(),
        )
        worker_result = create_mock_agent_result(
            success=True,
            output="已完成重构任务",
        )
        reviewer_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(decision="complete", score=90),
        )

        with (
            patch.object(planner._executor, "execute", new_callable=AsyncMock) as mock_planner_exec,
            patch.object(worker._executor, "execute", new_callable=AsyncMock) as mock_worker_exec,
            patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_reviewer_exec,
        ):
            mock_planner_exec.return_value = planner_result
            mock_worker_exec.return_value = worker_result
            mock_reviewer_exec.return_value = reviewer_result

            # 1. 规划者生成任务
            plan_result = await planner.execute("重构代码库")
            assert plan_result["success"] is True
            tasks = plan_result["tasks"]
            assert len(tasks) > 0

            # 2. 执行者执行第一个任务
            first_task = tasks[0]
            exec_result = await worker.execute(
                first_task["instruction"],
                context={"target_files": first_task["target_files"]},
            )
            assert exec_result["success"] is True

            # 3. 评审者评审结果
            review_result = await reviewer.review_iteration(
                goal="重构代码库",
                iteration_id=1,
                tasks_completed=[{"id": "task-1", "title": first_task["title"], "result": exec_result}],
                tasks_failed=[],
            )
            assert review_result["success"] is True
            assert review_result["decision"] == ReviewDecision.COMPLETE

    @pytest.mark.asyncio
    async def test_multiple_iterations(self):
        """测试多轮迭代场景"""
        config = ReviewerConfig(name="iter-reviewer")
        reviewer = ReviewerAgent(config)

        # 第一轮：继续
        first_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(decision="continue", score=60),
        )
        # 第二轮：继续
        second_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(decision="continue", score=80),
        )
        # 第三轮：完成
        third_result = create_mock_agent_result(
            success=True,
            output=create_reviewer_json_output(decision="complete", score=95),
        )

        with patch.object(reviewer._executor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = [first_result, second_result, third_result]

            # 第一轮迭代
            result1 = await reviewer.review_iteration(
                goal="完成项目",
                iteration_id=1,
                tasks_completed=[],
                tasks_failed=[],
            )
            assert result1["decision"] == ReviewDecision.CONTINUE

            # 第二轮迭代
            result2 = await reviewer.review_iteration(
                goal="完成项目",
                iteration_id=2,
                tasks_completed=[{"id": "t1"}],
                tasks_failed=[],
            )
            assert result2["decision"] == ReviewDecision.CONTINUE

            # 第三轮迭代
            result3 = await reviewer.review_iteration(
                goal="完成项目",
                iteration_id=3,
                tasks_completed=[{"id": "t1"}, {"id": "t2"}],
                tasks_failed=[],
            )
            assert result3["decision"] == ReviewDecision.COMPLETE

            # 验证评审历史
            assert len(reviewer.review_history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
