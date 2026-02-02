"""端到端流式处理测试

验证流式输出处理的完整工作流，包括：
- 启用流式的完整工作流
- ProgressTracker 集成
- 事件顺序验证
- 流式事件处理
- 流式配置传播

使用 Mock 替代真实 Cursor CLI 调用
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agents.reviewer import ReviewDecision
from coordinator.orchestrator import Orchestrator, OrchestratorConfig
from cursor.client import CursorAgentConfig
from cursor.streaming import (
    DiffInfo,
    ProgressTracker,
    StreamEvent,
    StreamEventLogger,
    StreamEventType,
    ToolCallInfo,
    parse_stream_event,
)
from scripts.run_basic import resolve_stream_log_config as resolve_stream_log_config_single
from scripts.run_mp import resolve_stream_log_config as resolve_stream_log_config_multi


class TestStreamingWorkflow:
    """流式工作流测试"""

    @pytest.fixture
    def streaming_orchestrator(self) -> Orchestrator:
        """创建启用流式的 Orchestrator"""
        cursor_config = CursorAgentConfig(
            working_directory=".",
            stream_events_enabled=True,
            stream_log_console=False,  # 测试时禁用控制台输出
            stream_log_detail_dir="logs/test_stream/detail/",
            stream_log_raw_dir="logs/test_stream/raw/",
        )
        config = OrchestratorConfig(
            working_directory=".",
            max_iterations=2,
            worker_pool_size=1,
            enable_auto_commit=False,
            cursor_config=cursor_config,
            stream_events_enabled=True,
            stream_log_console=False,
            stream_log_detail_dir="logs/test_stream/detail/",
            stream_log_raw_dir="logs/test_stream/raw/",
        )
        return Orchestrator(config)

    @pytest.mark.asyncio
    async def test_streaming_enabled_workflow(self, streaming_orchestrator: Orchestrator) -> None:
        """启用流式的完整工作流"""
        orchestrator = streaming_orchestrator

        # Mock 规划阶段
        mock_plan_result = {
            "success": True,
            "tasks": [
                {
                    "type": "implement",
                    "title": "流式测试任务",
                    "description": "验证流式处理",
                    "instruction": "创建测试文件",
                    "target_files": ["stream_test.py"],
                }
            ],
        }

        # Mock 评审阶段
        mock_review_result = {
            "success": True,
            "decision": ReviewDecision.COMPLETE,
            "score": 100,
            "summary": "流式处理测试完成",
        }

        with (
            patch.object(orchestrator.planner, "execute", new_callable=AsyncMock) as mock_planner,
            patch.object(orchestrator.worker_pool, "start", new_callable=AsyncMock) as mock_workers,
            patch.object(orchestrator.reviewer, "review_iteration", new_callable=AsyncMock) as mock_reviewer,
        ):
            mock_planner.return_value = mock_plan_result
            mock_reviewer.return_value = mock_review_result

            async def simulate_complete(queue: Any, iteration_id: int) -> None:
                tasks = queue.get_tasks_by_iteration(iteration_id)
                for task in tasks:
                    task.complete({"output": "流式执行完成"})

            mock_workers.side_effect = simulate_complete

            result = await orchestrator.run("流式工作流测试")

            # 验证结果
            assert result["success"] is True
            assert result["iterations_completed"] == 1
            assert result["total_tasks_created"] == 1
            assert result["total_tasks_completed"] == 1

            # 验证流式配置已应用
            assert orchestrator.config.stream_events_enabled is True

    @pytest.mark.asyncio
    async def test_progress_tracker_integration(self) -> None:
        """ProgressTracker 集成测试"""
        tracker = ProgressTracker(verbose=False, show_diff=True)

        # 模拟系统初始化事件
        init_event = StreamEvent(
            type=StreamEventType.SYSTEM_INIT,
            subtype="init",
            model="gpt-5.2-high",
        )
        tracker.on_event(init_event)

        assert tracker.model == "gpt-5.2-high"
        assert len(tracker.events) == 1

        # 模拟助手消息事件
        assistant_event = StreamEvent(
            type=StreamEventType.ASSISTANT,
            content="正在分析代码结构...",
        )
        tracker.on_event(assistant_event)

        assert "正在分析代码结构" in tracker.accumulated_text
        assert len(tracker.events) == 2

        # 模拟工具调用开始事件
        tool_started = StreamEvent(
            type=StreamEventType.TOOL_STARTED,
            subtype="started",
            tool_call=ToolCallInfo(
                tool_type="read",
                path="src/main.py",
            ),
        )
        tracker.on_event(tool_started)

        assert tracker.tool_count == 1
        assert len(tracker.events) == 3

        # 模拟工具调用完成事件
        tool_completed = StreamEvent(
            type=StreamEventType.TOOL_COMPLETED,
            subtype="completed",
            tool_call=ToolCallInfo(
                tool_type="read",
                path="src/main.py",
                success=True,
                result={"totalLines": 100},
            ),
        )
        tracker.on_event(tool_completed)

        assert "src/main.py" in tracker.files_read
        assert len(tracker.events) == 4  # init + assistant + tool_started + tool_completed

        # 模拟编辑事件 - 先发送 DIFF_STARTED
        diff_started_event = StreamEvent(
            type=StreamEventType.DIFF_STARTED,
            subtype="started",
            tool_call=ToolCallInfo(
                tool_type="str_replace",
                path="src/utils.py",
                old_string="old_code",
                new_string="new_code",
                is_diff=True,
            ),
        )
        tracker.on_event(diff_started_event)

        assert tracker.diff_count == 1  # DIFF_STARTED 时递增

        # 再发送 DIFF_COMPLETED
        diff_completed_event = StreamEvent(
            type=StreamEventType.DIFF_COMPLETED,
            subtype="completed",
            tool_call=ToolCallInfo(
                tool_type="str_replace",
                path="src/utils.py",
                old_string="old_code",
                new_string="new_code",
                is_diff=True,
                success=True,
            ),
            diff_info=DiffInfo(
                path="src/utils.py",
                old_string="old_code",
                new_string="new_code",
            ),
        )
        tracker.on_event(diff_completed_event)

        assert "src/utils.py" in tracker.files_edited
        assert tracker.diff_count == 1  # DIFF_COMPLETED 不再递增
        assert len(tracker.events) == 6  # 现在有 6 个事件

        # 模拟结果事件
        result_event = StreamEvent(
            type=StreamEventType.RESULT,
            duration_ms=1500,
        )
        tracker.on_event(result_event)

        assert tracker.duration_ms == 1500
        assert tracker.is_complete is True
        assert len(tracker.events) == 7  # 共 7 个事件

        # 获取摘要
        summary = tracker.get_summary()
        assert summary["model"] == "gpt-5.2-high"
        assert summary["tool_count"] == 1
        assert summary["diff_count"] == 1
        assert summary["is_complete"] is True
        assert summary["duration_ms"] == 1500
        assert summary["total_events"] == 7

    @pytest.mark.asyncio
    async def test_stream_event_sequence(self) -> None:
        """事件顺序验证"""
        from core.config import DEFAULT_WORKER_MODEL

        tracker = ProgressTracker(verbose=False)

        # 定义期望的事件序列
        events = [
            StreamEvent(type=StreamEventType.SYSTEM_INIT, model=DEFAULT_WORKER_MODEL),
            StreamEvent(type=StreamEventType.ASSISTANT, content="开始执行..."),
            StreamEvent(
                type=StreamEventType.TOOL_STARTED,
                tool_call=ToolCallInfo(tool_type="read", path="file.py"),
            ),
            StreamEvent(
                type=StreamEventType.TOOL_COMPLETED,
                tool_call=ToolCallInfo(tool_type="read", path="file.py", success=True),
            ),
            StreamEvent(
                type=StreamEventType.DIFF_STARTED,
                tool_call=ToolCallInfo(
                    tool_type="str_replace",
                    path="file.py",
                    is_diff=True,
                ),
            ),
            StreamEvent(
                type=StreamEventType.DIFF_COMPLETED,
                tool_call=ToolCallInfo(
                    tool_type="str_replace",
                    path="file.py",
                    is_diff=True,
                    success=True,
                ),
            ),
            StreamEvent(type=StreamEventType.ASSISTANT, content="任务完成"),
            StreamEvent(type=StreamEventType.RESULT, duration_ms=2000),
        ]

        # 依次处理事件
        for event in events:
            tracker.on_event(event)

        # 验证事件顺序
        assert len(tracker.events) == 8

        # 验证事件类型顺序
        event_types = [e.type for e in tracker.events]
        assert event_types[0] == StreamEventType.SYSTEM_INIT
        assert event_types[-1] == StreamEventType.RESULT

        # 验证统计
        assert tracker.tool_count == 1
        assert tracker.diff_count == 1
        assert tracker.is_complete is True


class TestStreamEventProcessing:
    """流式事件处理测试"""

    def test_system_init_event(self) -> None:
        """系统初始化事件解析"""
        line = json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "gpt-5.2-high",
                "apiKeySource": "env",
                "cwd": "/test/path",
                "session_id": "test-session-123",
                "permissionMode": "default",
            }
        )

        event = parse_stream_event(line)

        assert event is not None
        assert event.type == StreamEventType.SYSTEM_INIT
        assert event.model == "gpt-5.2-high"
        assert event.subtype == "init"
        assert event.data.get("session_id") == "test-session-123"
        assert event.data.get("permissionMode") == "default"

    def test_tool_call_events(self) -> None:
        """工具调用事件序列测试"""
        # 测试 read 工具调用开始
        read_started = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "call-1",
                "tool_call": {
                    "readToolCall": {
                        "args": {"path": "src/main.py"},
                    }
                },
            }
        )

        event = parse_stream_event(read_started)
        assert event is not None
        assert event.type == StreamEventType.TOOL_STARTED
        assert event.tool_call is not None
        assert event.tool_call.tool_type == "read"
        assert event.tool_call.path == "src/main.py"

        # 测试 read 工具调用完成
        read_completed = json.dumps(
            {
                "type": "tool_call",
                "subtype": "completed",
                "call_id": "call-1",
                "tool_call": {
                    "readToolCall": {
                        "args": {"path": "src/main.py"},
                        "result": {"success": {"totalLines": 150, "content": "..."}},
                    }
                },
            }
        )

        event = parse_stream_event(read_completed)
        assert event is not None
        assert event.type == StreamEventType.TOOL_COMPLETED
        assert event.tool_call is not None
        assert event.tool_call.success is True

        # 测试 write 工具调用
        write_completed = json.dumps(
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {
                    "writeToolCall": {
                        "args": {"path": "output.py"},
                        "result": {"success": {"linesCreated": 50, "fileSize": 1024}},
                    }
                },
            }
        )

        event = parse_stream_event(write_completed)
        assert event is not None
        assert event.tool_call is not None
        assert event.tool_call.tool_type == "write"
        assert event.tool_call.success is True
        assert event.tool_call.result.get("linesCreated") == 50

        # 测试 shell 工具调用
        shell_started = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {
                    "shellToolCall": {
                        "args": {"command": "pytest tests/"},
                    }
                },
            }
        )

        event = parse_stream_event(shell_started)
        assert event is not None
        assert event.tool_call is not None
        assert event.tool_call.tool_type == "shell"
        assert event.tool_call.args.get("command") == "pytest tests/"

        # 测试 str_replace 工具调用（差异操作）
        str_replace_started = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {
                    "strReplaceToolCall": {
                        "args": {
                            "path": "config.py",
                            "old_string": "DEBUG = False",
                            "new_string": "DEBUG = True",
                        },
                    }
                },
            }
        )

        event = parse_stream_event(str_replace_started)
        assert event is not None
        # str_replace 被识别为差异操作
        assert event.type == StreamEventType.DIFF_STARTED
        assert event.tool_call is not None
        assert event.tool_call.is_diff is True
        assert event.tool_call.old_string == "DEBUG = False"
        assert event.tool_call.new_string == "DEBUG = True"

    def test_assistant_message_accumulation(self) -> None:
        """消息累积测试"""
        tracker = ProgressTracker(verbose=False)

        # 模拟多个助手消息事件（流式累积）
        messages = [
            "正在",
            "分析",
            "代码",
            "结构",
            "...",
        ]

        for msg in messages:
            event = StreamEvent(type=StreamEventType.ASSISTANT, content=msg)
            tracker.on_event(event)

        # 验证消息正确累积
        assert tracker.accumulated_text == "正在分析代码结构..."
        assert len(tracker.events) == 5

        # 测试带有 list 内容的消息解析
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"text": "第一部分 "},
                        {"text": "第二部分 "},
                        {"text": "第三部分"},
                    ]
                },
            }
        )

        parsed_event = parse_stream_event(line)
        assert parsed_event is not None
        assert parsed_event.type == StreamEventType.ASSISTANT
        assert parsed_event.content == "第一部分 第二部分 第三部分"

    def test_result_event_processing(self) -> None:
        """结果事件处理测试"""
        # 测试成功结果
        success_result = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "duration_ms": 3456,
                "duration_api_ms": 3000,
                "result": "任务完成",
                "session_id": "session-abc",
            }
        )

        event = parse_stream_event(success_result)
        assert event is not None
        assert event.type == StreamEventType.RESULT
        assert event.duration_ms == 3456
        assert event.data.get("is_error") is False

        # 测试 ProgressTracker 处理结果事件
        tracker = ProgressTracker(verbose=False)
        tracker.on_event(event)

        assert tracker.duration_ms == 3456
        assert tracker.is_complete is True
        assert len(tracker.errors) == 0

        # 测试错误事件
        error_event = StreamEvent(
            type=StreamEventType.ERROR,
            data={"error": "执行超时"},
        )
        tracker.on_event(error_event)

        assert "执行超时" in tracker.errors


class TestStreamingConfiguration:
    """流式配置测试"""

    def test_stream_log_directories(self, tmp_path: Path) -> None:
        """日志目录配置测试"""
        detail_dir = tmp_path / "detail"
        raw_dir = tmp_path / "raw"

        # 创建日志器
        logger = StreamEventLogger(
            agent_id="test-agent",
            agent_role="worker",
            agent_name="test-worker",
            console=False,
            detail_dir=str(detail_dir),
            raw_dir=str(raw_dir),
        )

        # 写入测试数据
        logger.handle_raw_line('{"type":"system","subtype":"init","model":"test"}')
        logger.handle_event(
            StreamEvent(
                type=StreamEventType.SYSTEM_INIT,
                model="test-model",
            )
        )
        logger.handle_event(
            StreamEvent(
                type=StreamEventType.ASSISTANT,
                content="测试消息",
            )
        )
        logger.close()

        # 验证目录和文件创建
        assert detail_dir.exists()
        assert raw_dir.exists()

        raw_files = list(raw_dir.glob("*.jsonl"))
        detail_files = list(detail_dir.glob("*.log"))

        assert len(raw_files) == 1
        assert len(detail_files) == 1

        # 验证文件内容
        raw_content = raw_files[0].read_text(encoding="utf-8")
        assert '"type":"system"' in raw_content

        detail_content = detail_files[0].read_text(encoding="utf-8")
        assert "worker:test-agent" in detail_content
        assert "test-model" in detail_content or "测试消息" in detail_content

    def test_console_output_toggle(self, tmp_path: Path, capsys) -> None:
        """控制台输出开关测试"""
        detail_dir = tmp_path / "detail"

        # 测试启用控制台输出
        logger_with_console = StreamEventLogger(
            agent_id="console-test",
            agent_role="planner",
            agent_name="planner",
            console=True,
            detail_dir=str(detail_dir),
            raw_dir="",
        )

        logger_with_console.handle_event(
            StreamEvent(
                type=StreamEventType.ASSISTANT,
                content="控制台输出测试",
            )
        )
        logger_with_console.close()

        captured = capsys.readouterr()
        assert "控制台输出测试" in captured.out

        # 测试禁用控制台输出
        logger_no_console = StreamEventLogger(
            agent_id="no-console-test",
            agent_role="worker",
            agent_name="worker",
            console=False,
            detail_dir=str(detail_dir),
            raw_dir="",
        )

        logger_no_console.handle_event(
            StreamEvent(
                type=StreamEventType.ASSISTANT,
                content="不应该输出到控制台",
            )
        )
        logger_no_console.close()

        captured = capsys.readouterr()
        assert "不应该输出到控制台" not in captured.out

    def test_stream_config_propagation(self) -> None:
        """配置传播到 Agent 测试"""
        # 测试 CLI 参数覆盖配置文件
        args = SimpleNamespace(
            stream_log_enabled=True,
            stream_log_console=False,
            stream_log_detail_dir="/custom/detail/",
            stream_log_raw_dir="/custom/raw/",
        )

        config_data = {
            "logging": {
                "stream_json": {
                    "enabled": False,  # 被 CLI 覆盖
                    "console": True,  # 被 CLI 覆盖
                    "detail_dir": "logs/default/detail/",
                    "raw_dir": "logs/default/raw/",
                }
            }
        }

        # 使用单进程版本的配置解析
        resolved = resolve_stream_log_config_single(
            cli_enabled=args.stream_log_enabled,
            cli_console=args.stream_log_console,
            cli_detail_dir=args.stream_log_detail_dir,
            cli_raw_dir=args.stream_log_raw_dir,
            config_data=config_data,
        )

        assert resolved["enabled"] is True
        assert resolved["console"] is False
        assert resolved["detail_dir"] == "/custom/detail/"
        assert resolved["raw_dir"] == "/custom/raw/"

        # 使用多进程版本的配置解析
        resolved_mp = resolve_stream_log_config_multi(
            cli_enabled=args.stream_log_enabled,
            cli_console=args.stream_log_console,
            cli_detail_dir=args.stream_log_detail_dir,
            cli_raw_dir=args.stream_log_raw_dir,
            config_data=config_data,
        )

        assert resolved_mp["enabled"] is True
        assert resolved_mp["console"] is False
        assert resolved_mp["detail_dir"] == "/custom/detail/"
        assert resolved_mp["raw_dir"] == "/custom/raw/"

    def test_stream_config_defaults(self) -> None:
        """默认配置测试"""
        args = SimpleNamespace(
            stream_log_enabled=None,
            stream_log_console=None,
            stream_log_detail_dir=None,
            stream_log_raw_dir=None,
        )

        # 空配置
        config_data: dict[str, Any] = {}

        resolved = resolve_stream_log_config_single(
            cli_enabled=args.stream_log_enabled,
            cli_console=args.stream_log_console,
            cli_detail_dir=args.stream_log_detail_dir,
            cli_raw_dir=args.stream_log_raw_dir,
            config_data=config_data,
        )

        # 验证默认值
        assert resolved["enabled"] is False
        assert resolved["console"] is True
        assert resolved["detail_dir"] == "logs/stream_json/detail/"
        assert resolved["raw_dir"] == "logs/stream_json/raw/"

    def test_orchestrator_config_stream_fields(self) -> None:
        """Orchestrator 配置中的流式字段测试"""
        cursor_config = CursorAgentConfig(
            stream_events_enabled=True,
            stream_log_console=True,
            stream_log_detail_dir="custom/detail/",
            stream_log_raw_dir="custom/raw/",
        )

        config = OrchestratorConfig(
            working_directory=".",
            stream_events_enabled=True,
            stream_log_console=True,
            stream_log_detail_dir="orchestrator/detail/",
            stream_log_raw_dir="orchestrator/raw/",
            cursor_config=cursor_config,
        )

        # 验证配置正确设置
        assert config.stream_events_enabled is True
        assert config.stream_log_console is True
        assert config.stream_log_detail_dir == "orchestrator/detail/"
        assert config.stream_log_raw_dir == "orchestrator/raw/"

        # 验证嵌套的 cursor_config
        assert config.cursor_config.stream_events_enabled is True
        assert config.cursor_config.stream_log_detail_dir == "custom/detail/"

    def test_config_partial_override(self) -> None:
        """部分配置覆盖测试"""
        args = SimpleNamespace(
            stream_log_enabled=True,  # 仅覆盖 enabled
            stream_log_console=None,  # 使用配置文件值
            stream_log_detail_dir=None,
            stream_log_raw_dir="/override/raw/",  # 仅覆盖 raw_dir
        )

        config_data: dict[str, Any] = {
            "logging": {
                "stream_json": {
                    "enabled": False,
                    "console": False,
                    "detail_dir": "config/detail/",
                    "raw_dir": "config/raw/",
                }
            }
        }

        resolved = resolve_stream_log_config_single(
            cli_enabled=args.stream_log_enabled,
            cli_console=args.stream_log_console,
            cli_detail_dir=args.stream_log_detail_dir,
            cli_raw_dir=args.stream_log_raw_dir,
            config_data=config_data,
        )

        # enabled 被 CLI 覆盖
        assert resolved["enabled"] is True
        # console 使用配置文件值
        assert resolved["console"] is False
        # detail_dir 使用配置文件值
        assert resolved["detail_dir"] == "config/detail/"
        # raw_dir 被 CLI 覆盖
        assert resolved["raw_dir"] == "/override/raw/"
