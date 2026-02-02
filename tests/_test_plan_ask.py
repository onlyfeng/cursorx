#!/usr/bin/env python3
"""临时测试脚本 - 验证 TestRunPlanAskModes"""

import asyncio
import sys

sys.path.insert(0, "/mnt/e/QianFeng/ai/cursorx")

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

from run import Runner

# 创建 mock_args
mock_args = argparse.Namespace(
    task="测试任务",
    mode="auto",
    directory=".",
    workers=3,
    max_iterations="10",
    strict=False,
    verbose=False,
    skip_online=False,
    dry_run=False,
    force_update=False,
    use_knowledge=False,
    search_knowledge=None,
    self_update=False,
    planner_model="gpt-5.2-high",
    worker_model="gpt-5.2-codex-high",
    stream_log=True,
    no_auto_analyze=False,
    auto_commit=True,
    auto_push=False,
    commit_per_iteration=False,
)

runner = Runner(mock_args)


async def test_run_plan_success():
    """测试规划模式成功返回计划"""
    with patch("run.subprocess.run") as mock_subprocess:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "计划内容"
        mock_subprocess.return_value = mock_result

        options = runner._merge_options({})
        result = await runner._run_plan("测试规划", options)

        assert result["success"] is True
        assert result["goal"] == "测试规划"
        assert result["mode"] == "plan"
        assert "plan" in result
        assert result["dry_run"] is True
        print("test_run_plan_success: PASSED")


async def test_run_plan_timeout():
    """测试规划模式超时处理"""
    import subprocess as sp

    with patch("run.subprocess.run") as mock_subprocess:
        mock_subprocess.side_effect = sp.TimeoutExpired(cmd="agent", timeout=300)

        result = await runner._run_plan("超时任务", runner._merge_options({}))

        assert result["success"] is False
        assert result["error"] == "timeout"
        print("test_run_plan_timeout: PASSED")


async def test_run_plan_error():
    """测试规划模式错误处理"""
    with patch("run.subprocess.run") as mock_subprocess:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "错误信息"
        mock_subprocess.return_value = mock_result

        result = await runner._run_plan("失败任务", runner._merge_options({}))

        assert result["success"] is False
        assert "error" in result
        print("test_run_plan_error: PASSED")


async def test_run_ask_success():
    """测试问答模式成功返回答案（使用 AskAgentExecutor）"""
    from cursor.executor import AgentResult

    mock_agent_result = AgentResult(
        success=True,
        output="回答内容",
        error=None,
        executor_type="ask",
    )

    with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
        mock_instance = MagicMock()
        mock_instance.execute = AsyncMock(return_value=mock_agent_result)
        MockExecutor.return_value = mock_instance

        result = await runner._run_ask("测试问题", runner._merge_options({}))

        assert result["success"] is True
        assert result["goal"] == "测试问题"
        assert result["mode"] == "ask"
        assert "answer" in result
        print("test_run_ask_success: PASSED")


async def test_run_ask_timeout():
    """测试问答模式超时处理"""
    with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
        mock_instance = MagicMock()
        mock_instance.execute = AsyncMock(side_effect=asyncio.TimeoutError())
        MockExecutor.return_value = mock_instance

        result = await runner._run_ask("超时问题", runner._merge_options({}))

        assert result["success"] is False
        assert result["error"] == "timeout"
        print("test_run_ask_timeout: PASSED")


async def test_result_structure():
    """验证返回的 dict 结构正确"""
    from cursor.executor import AgentResult

    # Plan 成功结构
    with patch("run.subprocess.run") as mock_subprocess:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "内容"
        mock_subprocess.return_value = mock_result

        plan_result = await runner._run_plan("任务", runner._merge_options({}))
        assert all(k in plan_result for k in ["success", "goal", "mode", "plan", "dry_run"])

    # Ask 成功结构（使用 AskAgentExecutor）
    mock_agent_result = AgentResult(
        success=True,
        output="回答",
        error=None,
        executor_type="ask",
    )

    with patch("cursor.executor.AskAgentExecutor") as MockExecutor:
        mock_instance = MagicMock()
        mock_instance.execute = AsyncMock(return_value=mock_agent_result)
        MockExecutor.return_value = mock_instance

        ask_result = await runner._run_ask("问题", runner._merge_options({}))
        assert all(k in ask_result for k in ["success", "goal", "mode", "answer"])

    print("test_result_structure: PASSED")


async def test_ask_readonly_guarantee():
    """测试问答模式的只读保证"""
    from cursor.executor import AskAgentExecutor

    # 验证 AskAgentExecutor 强制设置 force_write=False
    executor = AskAgentExecutor()
    assert executor.config.force_write is False, "AskAgentExecutor 应强制 force_write=False"
    assert executor.config.mode == "ask", "AskAgentExecutor 应强制 mode=ask"
    print("test_ask_readonly_guarantee: PASSED")


async def main():
    print("运行 TestRunPlanAskModes 测试...\n")
    await test_run_plan_success()
    await test_run_plan_timeout()
    await test_run_plan_error()
    await test_run_ask_success()
    await test_run_ask_timeout()
    await test_result_structure()
    await test_ask_readonly_guarantee()
    print("\n所有测试通过!")


if __name__ == "__main__":
    asyncio.run(main())
