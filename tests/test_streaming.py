import json
from types import SimpleNamespace
from pathlib import Path

from cursor.streaming import (
    StreamEvent,
    StreamEventLogger,
    StreamEventType,
    ToolCallInfo,
    parse_stream_event,
)
from main import resolve_stream_log_config as resolve_stream_log_config_single
from main_mp import resolve_stream_log_config as resolve_stream_log_config_multi


def test_parse_stream_event_system_init() -> None:
    line = json.dumps({"type": "system", "subtype": "init", "model": "gpt-5.2-high"})
    event = parse_stream_event(line)
    assert event is not None
    assert event.type == StreamEventType.SYSTEM_INIT
    assert event.model == "gpt-5.2-high"


def test_parse_stream_event_assistant_content_join() -> None:
    line = json.dumps(
        {
            "type": "assistant",
            "message": {"content": [{"text": "hello "}, {"text": "world"}]},
        }
    )
    event = parse_stream_event(line)
    assert event is not None
    assert event.type == StreamEventType.ASSISTANT
    assert event.content == "hello world"


def test_parse_stream_event_tool_call() -> None:
    line = json.dumps(
        {
            "type": "tool_call",
            "subtype": "started",
            "tool_call": {
                "writeToolCall": {
                    "args": {"path": "foo.py"},
                    "result": {"success": {"linesCreated": 1, "fileSize": 2}},
                }
            },
        }
    )
    event = parse_stream_event(line)
    assert event is not None
    assert event.type == StreamEventType.TOOL_STARTED
    assert event.tool_call is not None
    assert event.tool_call.tool_type == "write"
    assert event.tool_call.path == "foo.py"
    assert event.tool_call.success is True


def test_parse_stream_event_tool_call_completed() -> None:
    line = json.dumps(
        {
            "type": "tool_call",
            "subtype": "completed",
            "tool_call": {"readToolCall": {"args": {"path": "bar.py"}}},
        }
    )
    event = parse_stream_event(line)
    assert event is not None
    assert event.type == StreamEventType.TOOL_COMPLETED
    assert event.tool_call is not None
    assert event.tool_call.tool_type == "read"
    assert event.tool_call.path == "bar.py"


def test_parse_stream_event_tool_call_shell() -> None:
    line = json.dumps(
        {
            "type": "tool_call",
            "subtype": "started",
            "tool_call": {"shellToolCall": {"args": {"command": "ls"}}},
        }
    )
    event = parse_stream_event(line)
    assert event is not None
    assert event.type == StreamEventType.TOOL_STARTED
    assert event.tool_call is not None
    assert event.tool_call.tool_type == "shell"
    assert event.tool_call.args.get("command") == "ls"


def test_parse_stream_event_assistant_string_content() -> None:
    line = json.dumps({"type": "assistant", "message": {"content": "plain"}})
    event = parse_stream_event(line)
    assert event is not None
    assert event.type == StreamEventType.ASSISTANT
    assert event.content == "plain"


def test_parse_stream_event_result_duration() -> None:
    line = json.dumps({"type": "result", "duration_ms": 456})
    event = parse_stream_event(line)
    assert event is not None
    assert event.type == StreamEventType.RESULT
    assert event.duration_ms == 456


def test_parse_stream_event_unknown_type() -> None:
    line = json.dumps({"type": "custom", "value": 1})
    event = parse_stream_event(line)
    assert event is not None
    assert event.type == StreamEventType.MESSAGE
    assert event.data.get("value") == 1


def test_parse_stream_event_invalid_json() -> None:
    line = "not-json"
    event = parse_stream_event(line)
    assert event is not None
    assert event.type == StreamEventType.MESSAGE
    assert event.content == "not-json"


def test_stream_event_logger_writes_files(tmp_path: Path) -> None:
    detail_dir = tmp_path / "detail"
    raw_dir = tmp_path / "raw"
    logger = StreamEventLogger(
        agent_id="agent-1",
        agent_role="planner",
        agent_name="planner",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir=str(raw_dir),
    )

    logger.handle_raw_line('{"type":"assistant","message":{"content":[{"text":"hi"}]}}')
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="hi"))
    logger.handle_event(
        StreamEvent(
            type=StreamEventType.TOOL_COMPLETED,
            tool_call=ToolCallInfo(tool_type="read", path="foo.py", success=True),
        )
    )
    logger.close()

    raw_files = list(raw_dir.glob("*.jsonl"))
    detail_files = list(detail_dir.glob("*.log"))
    assert len(raw_files) == 1
    assert len(detail_files) == 1

    raw_content = raw_files[0].read_text(encoding="utf-8")
    assert '"type":"assistant"' in raw_content

    detail_content = detail_files[0].read_text(encoding="utf-8")
    assert "planner:agent-1" in detail_content
    assert "hi" in detail_content


def test_stream_event_logger_error_event(tmp_path: Path) -> None:
    detail_dir = tmp_path / "detail"
    logger = StreamEventLogger(
        agent_id="agent-2",
        agent_role="worker",
        agent_name="worker",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir="",
    )
    logger.handle_event(StreamEvent(type=StreamEventType.ERROR, data={"error": "boom"}))
    logger.close()

    detail_files = list(detail_dir.glob("*.log"))
    assert len(detail_files) == 1
    detail_content = detail_files[0].read_text(encoding="utf-8")
    assert "错误: boom" in detail_content


def test_resolve_stream_log_config_cli_overrides() -> None:
    args = SimpleNamespace(
        stream_log_enabled=True,
        stream_log_console=False,
        stream_log_detail_dir="/tmp/detail",
        stream_log_raw_dir="/tmp/raw",
    )
    config_data = {
        "logging": {
            "stream_json": {
                "enabled": False,
                "console": True,
                "detail_dir": "logs/stream_json/detail/",
                "raw_dir": "logs/stream_json/raw/",
            }
        }
    }

    resolved = resolve_stream_log_config_single(args, config_data)
    assert resolved["enabled"] is True
    assert resolved["console"] is False
    assert resolved["detail_dir"] == "/tmp/detail"
    assert resolved["raw_dir"] == "/tmp/raw"


def test_resolve_stream_log_config_defaults() -> None:
    args = SimpleNamespace(
        stream_log_enabled=None,
        stream_log_console=None,
        stream_log_detail_dir=None,
        stream_log_raw_dir=None,
    )
    config_data = {}
    resolved = resolve_stream_log_config_multi(args, config_data)
    assert resolved["enabled"] is False
    assert resolved["console"] is True
    assert resolved["detail_dir"] == "logs/stream_json/detail/"
    assert resolved["raw_dir"] == "logs/stream_json/raw/"
