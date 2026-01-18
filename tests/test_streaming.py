import json
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cursor.streaming import (
    AdvancedTerminalRenderer,
    DiffInfo,
    ProgressTracker,
    StreamEvent,
    StreamEventLogger,
    StreamEventType,
    StreamRenderer,
    TerminalStreamRenderer,
    ToolCallInfo,
    format_colored_diff,
    format_diff,
    format_inline_diff,
    get_diff_stats,
    parse_stream_event,
)
from scripts.run_basic import resolve_stream_log_config as resolve_stream_log_config_single
from scripts.run_mp import resolve_stream_log_config as resolve_stream_log_config_multi


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


# ============== StreamEventLogger 消息聚合功能测试 ==============


def test_stream_event_logger_aggregation_enabled(tmp_path: Path) -> None:
    """测试 ASSISTANT 消息聚合功能开启时的行为"""
    detail_dir = tmp_path / "detail"
    logger = StreamEventLogger(
        agent_id="agent-agg",
        agent_role="worker",
        agent_name="test",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir="",
        aggregate_assistant_messages=True,  # 默认开启
    )

    # 发送多个 ASSISTANT 事件（模拟增量输出）
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Hello "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="World "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="!"))

    # 发送非 ASSISTANT 事件，触发缓冲区刷新
    logger.handle_event(
        StreamEvent(
            type=StreamEventType.TOOL_STARTED,
            tool_call=ToolCallInfo(tool_type="read", path="test.py"),
        )
    )

    logger.close()

    # 验证 detail 日志
    detail_files = list(detail_dir.glob("*.log"))
    assert len(detail_files) == 1
    detail_content = detail_files[0].read_text(encoding="utf-8")

    # 聚合模式下，应该只有一条完整的 ASSISTANT 消息
    assert "Hello World !" in detail_content
    # 工具事件应该单独记录
    assert "工具开始" in detail_content


def test_stream_event_logger_aggregation_flush_on_close(tmp_path: Path) -> None:
    """测试关闭时刷新 ASSISTANT 消息缓冲区"""
    detail_dir = tmp_path / "detail"
    logger = StreamEventLogger(
        agent_id="agent-close",
        agent_role="planner",
        agent_name="",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir="",
        aggregate_assistant_messages=True,
    )

    # 只发送 ASSISTANT 事件，不发送其他事件
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Test "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="message"))

    # 关闭时应该刷新缓冲区
    logger.close()

    detail_files = list(detail_dir.glob("*.log"))
    assert len(detail_files) == 1
    detail_content = detail_files[0].read_text(encoding="utf-8")

    # 验证聚合后的消息被写入
    assert "Test message" in detail_content


def test_stream_event_logger_aggregation_disabled(tmp_path: Path) -> None:
    """测试禁用消息聚合时的行为"""
    detail_dir = tmp_path / "detail"
    logger = StreamEventLogger(
        agent_id="agent-no-agg",
        agent_role="reviewer",
        agent_name="",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir="",
        aggregate_assistant_messages=False,  # 禁用聚合
    )

    # 发送多个 ASSISTANT 事件
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Line 1"))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Line 2"))

    logger.close()

    detail_files = list(detail_dir.glob("*.log"))
    assert len(detail_files) == 1
    detail_content = detail_files[0].read_text(encoding="utf-8")

    # 非聚合模式下，每个事件都应该单独记录
    lines = detail_content.strip().split("\n")
    # 应该有多行（至少 2 行，分别对应两个 ASSISTANT 事件）
    assert len(lines) >= 2


# ============== TerminalStreamRenderer 输出格式测试 ==============


def test_terminal_stream_renderer_verbose_mode() -> None:
    """测试基础 TerminalStreamRenderer 详细模式"""
    renderer = TerminalStreamRenderer(verbose=True)
    assert renderer.verbose is True


def test_terminal_stream_renderer_non_verbose_mode() -> None:
    """测试基础 TerminalStreamRenderer 精简模式"""
    renderer = TerminalStreamRenderer(verbose=False)
    assert renderer.verbose is False


def test_terminal_stream_renderer_render_init(capsys) -> None:
    """测试基础 TerminalStreamRenderer 初始化渲染"""
    renderer = TerminalStreamRenderer(verbose=False)
    renderer.render_init("test-model")

    captured = capsys.readouterr()
    assert "test-model" in captured.out


def test_terminal_stream_renderer_render_result(capsys) -> None:
    """测试基础 TerminalStreamRenderer 结果渲染"""
    renderer = TerminalStreamRenderer(verbose=False)
    renderer.render_result(duration_ms=1000, tool_count=5, text_length=100)

    captured = capsys.readouterr()
    assert "1000ms" in captured.out


# ============== AdvancedTerminalRenderer 高级渲染器测试 ==============


def test_advanced_terminal_renderer_init() -> None:
    """测试高级渲染器初始化"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=False,
        typing_delay=0,
        show_status_bar=False,
        output=output,
    )

    renderer.start()
    renderer.render_event(StreamEvent(type=StreamEventType.SYSTEM_INIT, model="gpt-4"))
    renderer.finish()

    result = output.getvalue()
    assert "gpt-4" in result


def test_advanced_terminal_renderer_tool_icons() -> None:
    """测试高级渲染器工具图标映射"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=False,
        typing_delay=0,
        show_status_bar=False,
        output=output,
    )

    renderer.start()

    # 测试各种工具类型的图标
    renderer.render_event(
        StreamEvent(
            type=StreamEventType.TOOL_STARTED,
            tool_call=ToolCallInfo(tool_type="read", path="test.py"),
        )
    )

    result = output.getvalue()
    assert "📖" in result or "read" in result


def test_advanced_terminal_renderer_diff_display() -> None:
    """测试高级渲染器差异显示"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=False,
        typing_delay=0,
        show_status_bar=False,
        output=output,
    )

    renderer.start()

    diff_info = DiffInfo(
        path="test.py",
        old_string="old line",
        new_string="new line",
    )

    renderer.render_event(
        StreamEvent(
            type=StreamEventType.DIFF,
            diff_info=diff_info,
        )
    )

    renderer.finish()

    result = output.getvalue()
    assert "test.py" in result


def test_advanced_terminal_renderer_color_disabled() -> None:
    """测试高级渲染器禁用颜色时的输出"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=False,
        typing_delay=0,
        show_status_bar=False,
        output=output,
    )

    colored_text = renderer._color("test", "red", "bold")
    assert colored_text == "test"  # 无颜色码

    ctrl = renderer._ctrl("clear_line")
    assert ctrl == ""  # 无控制序列


def test_advanced_terminal_renderer_color_enabled() -> None:
    """测试高级渲染器启用颜色时的输出"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=True,
        typing_delay=0,
        show_status_bar=False,
        output=output,
    )

    colored_text = renderer._color("test", "red")
    assert "\033[31m" in colored_text  # 红色 ANSI 码
    assert "\033[0m" in colored_text  # 重置码


def test_advanced_terminal_renderer_text_rendering() -> None:
    """测试高级渲染器文本渲染功能"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=False,
        typing_delay=0,
        show_status_bar=False,
        output=output,
    )

    renderer.start()
    renderer.render_text("Hello World")

    result = output.getvalue()
    assert "Hello World" in result
    assert renderer.char_count == 11  # "Hello World" 长度


# ============== ProgressTracker 集成渲染器测试 ==============


def test_progress_tracker_with_custom_renderer() -> None:
    """测试 ProgressTracker 使用自定义渲染器"""

    class MockRenderer(StreamRenderer):
        def __init__(self):
            self.events: list[str] = []

        def render_init(self, model: str) -> None:
            self.events.append(f"init:{model}")

        def render_assistant(self, content: str, accumulated_length: int) -> None:
            self.events.append(f"assistant:{content}")

        def render_tool_started(self, tool_count: int, tool) -> None:
            self.events.append(f"tool_started:{tool_count}")

        def render_tool_completed(self, tool) -> None:
            self.events.append("tool_completed")

        def render_diff_started(self, diff_count: int, tool) -> None:
            self.events.append(f"diff_started:{diff_count}")

        def render_diff_completed(self, tool, diff_info, show_diff: bool) -> None:
            self.events.append("diff_completed")

        def render_diff(self, diff_count: int, diff_info, show_diff: bool) -> None:
            self.events.append(f"diff:{diff_count}")

        def render_result(self, duration_ms: int, tool_count: int, text_length: int) -> None:
            self.events.append(f"result:{duration_ms}")

        def render_error(self, error: str) -> None:
            self.events.append(f"error:{error}")

    mock_renderer = MockRenderer()
    tracker = ProgressTracker(verbose=False, show_diff=True, renderer=mock_renderer)

    # 测试各种事件
    tracker.on_event(StreamEvent(type=StreamEventType.SYSTEM_INIT, model="test-model"))
    tracker.on_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Hello"))
    tracker.on_event(
        StreamEvent(
            type=StreamEventType.TOOL_STARTED,
            tool_call=ToolCallInfo(tool_type="read", path="test.py"),
        )
    )
    tracker.on_event(
        StreamEvent(
            type=StreamEventType.TOOL_COMPLETED,
            tool_call=ToolCallInfo(tool_type="read", path="test.py", success=True),
        )
    )
    tracker.on_event(StreamEvent(type=StreamEventType.RESULT, duration_ms=1000))

    # 验证渲染器被正确调用
    assert "init:test-model" in mock_renderer.events
    assert "assistant:Hello" in mock_renderer.events
    assert "tool_started:1" in mock_renderer.events
    assert "tool_completed" in mock_renderer.events
    assert "result:1000" in mock_renderer.events

    # 验证跟踪器状态
    assert tracker.model == "test-model"
    assert tracker.accumulated_text == "Hello"
    assert tracker.tool_count == 1
    assert tracker.is_complete is True


def test_progress_tracker_default_renderer() -> None:
    """测试 ProgressTracker 使用默认渲染器"""
    tracker = ProgressTracker(verbose=True, show_diff=False)

    # 验证默认渲染器类型
    assert tracker.renderer is not None
    # 默认渲染器应该是 TerminalStreamRenderer（基础版本）
    assert isinstance(tracker.renderer, TerminalStreamRenderer)
    assert tracker.renderer.verbose is True

    # 测试 verbose 属性传递
    assert tracker.verbose is True
    assert tracker.show_diff is False


def test_progress_tracker_file_tracking() -> None:
    """测试 ProgressTracker 文件跟踪功能"""
    # 使用 Mock 渲染器避免实际输出
    mock_renderer = Mock(spec=StreamRenderer)
    tracker = ProgressTracker(renderer=mock_renderer)

    # 模拟写入文件
    tracker.on_event(
        StreamEvent(
            type=StreamEventType.TOOL_COMPLETED,
            tool_call=ToolCallInfo(tool_type="write", path="new_file.py", success=True),
        )
    )

    # 模拟读取文件
    tracker.on_event(
        StreamEvent(
            type=StreamEventType.TOOL_COMPLETED,
            tool_call=ToolCallInfo(tool_type="read", path="existing.py", success=True),
        )
    )

    # 模拟编辑文件
    tracker.on_event(
        StreamEvent(
            type=StreamEventType.DIFF_COMPLETED,
            tool_call=ToolCallInfo(tool_type="edit", path="edited.py", success=True),
        )
    )

    # 验证文件跟踪
    assert "new_file.py" in tracker.files_written
    assert "existing.py" in tracker.files_read
    assert "edited.py" in tracker.files_edited


def test_progress_tracker_error_tracking() -> None:
    """测试 ProgressTracker 错误跟踪功能"""
    mock_renderer = Mock(spec=StreamRenderer)
    tracker = ProgressTracker(renderer=mock_renderer)

    tracker.on_event(
        StreamEvent(
            type=StreamEventType.ERROR,
            data={"error": "Test error message"},
        )
    )

    assert "Test error message" in tracker.errors
    mock_renderer.render_error.assert_called_once_with("Test error message")


def test_progress_tracker_get_summary() -> None:
    """测试 ProgressTracker 获取摘要功能"""
    mock_renderer = Mock(spec=StreamRenderer)
    tracker = ProgressTracker(renderer=mock_renderer)

    tracker.on_event(StreamEvent(type=StreamEventType.SYSTEM_INIT, model="gpt-5"))
    tracker.on_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Hello"))
    tracker.on_event(StreamEvent(type=StreamEventType.RESULT, duration_ms=500))

    summary = tracker.get_summary()

    assert summary["model"] == "gpt-5"
    assert summary["text_length"] == 5  # "Hello"
    assert summary["duration_ms"] == 500
    assert summary["is_complete"] is True
    assert summary["total_events"] == 3


# ============== 配置项正确生效测试 ==============


def test_stream_event_logger_config_options(tmp_path: Path) -> None:
    """测试 StreamEventLogger 配置项"""
    detail_dir = tmp_path / "detail"
    raw_dir = tmp_path / "raw"

    logger = StreamEventLogger(
        agent_id="config-test",
        agent_role="worker",
        agent_name="test-name",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir=str(raw_dir),
        aggregate_assistant_messages=True,
    )

    # 验证配置被正确应用
    assert logger.agent_id == "config-test"
    assert logger.agent_role == "worker"
    assert logger.agent_name == "test-name"
    assert logger.console is False
    assert logger.aggregate_assistant_messages is True
    assert logger.detail_dir == str(detail_dir)
    assert logger.raw_dir == str(raw_dir)

    logger.close()


def test_terminal_stream_renderer_config_options() -> None:
    """测试基础 TerminalStreamRenderer 配置项"""
    renderer = TerminalStreamRenderer(verbose=True)

    # 验证配置被正确应用
    assert renderer.verbose is True


def test_advanced_terminal_renderer_config_options() -> None:
    """测试 AdvancedTerminalRenderer 配置项"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=False,
        typing_delay=0.05,
        word_mode=True,
        show_status_bar=False,
        status_bar_position="top",
        min_width=60,
        max_width=120,
        output=output,
    )

    # 验证配置被正确应用
    assert renderer.use_color is False
    assert renderer.typing_delay == 0.05
    assert renderer.word_mode is True
    assert renderer.show_status_bar is False
    assert renderer.status_bar_position == "top"
    assert renderer.min_width == 60
    assert renderer.max_width == 120
    assert renderer.output is output


def test_progress_tracker_config_options() -> None:
    """测试 ProgressTracker 配置项"""
    mock_renderer = Mock(spec=StreamRenderer)
    tracker = ProgressTracker(
        verbose=True,
        show_diff=False,
        renderer=mock_renderer,
    )

    assert tracker.verbose is True
    assert tracker.show_diff is False
    assert tracker.renderer is mock_renderer


# ============== 差异格式化工具函数测试 ==============


def test_format_diff() -> None:
    """测试统一差异格式生成"""
    old = "line1\nline2\nline3"
    new = "line1\nmodified\nline3"

    diff = format_diff(old, new, "test.py")

    assert "a/test.py" in diff
    assert "b/test.py" in diff
    assert "-line2" in diff
    assert "+modified" in diff


def test_format_inline_diff() -> None:
    """测试行内差异格式生成"""
    old = "line1\nline2"
    new = "line1\nline3"

    diff = format_inline_diff(old, new)

    assert "- line2" in diff
    assert "+ line3" in diff


def test_format_colored_diff() -> None:
    """测试带颜色的差异格式"""
    old = "old"
    new = "new"

    # 启用 ANSI 颜色
    colored = format_colored_diff(old, new, use_ansi=True)
    assert "\033[31m" in colored  # 红色（删除）
    assert "\033[32m" in colored  # 绿色（添加）

    # 禁用 ANSI 颜色
    no_color = format_colored_diff(old, new, use_ansi=False)
    assert "\033[" not in no_color


def test_get_diff_stats() -> None:
    """测试差异统计信息"""
    old = "line1\nline2\nline3"
    new = "line1\nmodified\nline3\nline4"

    stats = get_diff_stats(old, new)

    assert stats["old_lines"] == 3
    assert stats["new_lines"] == 4
    assert stats["insertions"] >= 1  # 至少添加了 modified 和 line4
    assert stats["deletions"] >= 1  # 至少删除了 line2
    assert 0 <= stats["similarity"] <= 1


# ============== AdvancedTerminalRenderer 与 StreamRenderer 接口兼容性测试 ==============


def test_advanced_terminal_renderer_implements_stream_renderer() -> None:
    """测试 AdvancedTerminalRenderer 正确实现了 StreamRenderer 接口"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=False,
        typing_delay=0,
        show_status_bar=False,
        output=output,
    )

    # 验证是 StreamRenderer 的实例
    assert isinstance(renderer, StreamRenderer)

    # 验证所有抽象方法都存在且可调用
    assert callable(renderer.render_init)
    assert callable(renderer.render_assistant)
    assert callable(renderer.render_tool_started)
    assert callable(renderer.render_tool_completed)
    assert callable(renderer.render_diff_started)
    assert callable(renderer.render_diff_completed)
    assert callable(renderer.render_diff)
    assert callable(renderer.render_result)
    assert callable(renderer.render_error)


def test_advanced_terminal_renderer_with_progress_tracker() -> None:
    """测试 AdvancedTerminalRenderer 与 ProgressTracker 配合使用"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=False,
        typing_delay=0,
        show_status_bar=False,
        output=output,
    )

    # 使用 AdvancedTerminalRenderer 作为 ProgressTracker 的渲染器
    tracker = ProgressTracker(verbose=False, show_diff=True, renderer=renderer)

    # 验证渲染器被正确设置
    assert tracker.renderer is renderer

    # 测试各种事件
    tracker.on_event(StreamEvent(type=StreamEventType.SYSTEM_INIT, model="test-model"))
    tracker.on_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Hello"))
    tracker.on_event(
        StreamEvent(
            type=StreamEventType.TOOL_STARTED,
            tool_call=ToolCallInfo(tool_type="read", path="test.py"),
        )
    )
    tracker.on_event(
        StreamEvent(
            type=StreamEventType.TOOL_COMPLETED,
            tool_call=ToolCallInfo(tool_type="read", path="test.py", success=True),
        )
    )
    tracker.on_event(StreamEvent(type=StreamEventType.RESULT, duration_ms=1000))

    # 验证跟踪器状态
    assert tracker.model == "test-model"
    assert tracker.accumulated_text == "Hello"
    assert tracker.tool_count == 1
    assert tracker.is_complete is True

    # 验证渲染器输出
    result = output.getvalue()
    assert "test-model" in result
    assert "Hello" in result


def test_advanced_terminal_renderer_render_methods() -> None:
    """测试 AdvancedTerminalRenderer 所有渲染方法的输出"""
    output = io.StringIO()
    renderer = AdvancedTerminalRenderer(
        use_color=False,
        typing_delay=0,
        show_status_bar=False,
        output=output,
    )

    # 测试 render_init
    renderer.render_init("gpt-5.2-high")
    assert "gpt-5.2-high" in output.getvalue()

    # 测试 render_assistant
    output.truncate(0)
    output.seek(0)
    renderer.render_assistant("Hello World", accumulated_length=11)
    assert "Hello World" in output.getvalue()

    # 测试 render_tool_started
    output.truncate(0)
    output.seek(0)
    tool = ToolCallInfo(tool_type="read", path="test.py")
    renderer.render_tool_started(tool_count=1, tool=tool)
    result = output.getvalue()
    assert "read" in result
    assert "test.py" in result

    # 测试 render_tool_completed
    output.truncate(0)
    output.seek(0)
    tool.success = True
    renderer.render_tool_completed(tool)
    assert "✓" in output.getvalue()

    # 测试 render_diff_started
    output.truncate(0)
    output.seek(0)
    diff_tool = ToolCallInfo(tool_type="edit", path="src/main.py", is_diff=True)
    renderer.render_diff_started(diff_count=1, tool=diff_tool)
    assert "src/main.py" in output.getvalue()

    # 测试 render_diff_completed
    output.truncate(0)
    output.seek(0)
    diff_info = DiffInfo(
        path="src/main.py",
        old_string="old line",
        new_string="new line",
    )
    renderer.render_diff_completed(diff_tool, diff_info, show_diff=True)
    assert "✓" in output.getvalue()

    # 测试 render_diff
    output.truncate(0)
    output.seek(0)
    renderer.render_diff(diff_count=1, diff_info=diff_info, show_diff=True)
    assert "src/main.py" in output.getvalue()

    # 测试 render_result
    output.truncate(0)
    output.seek(0)
    renderer.render_result(duration_ms=500, tool_count=3, text_length=100)
    assert "500" in output.getvalue()

    # 测试 render_error
    output.truncate(0)
    output.seek(0)
    renderer.render_error("Test error")
    assert "Test error" in output.getvalue()


def test_advanced_terminal_renderer_interface_signature() -> None:
    """测试 AdvancedTerminalRenderer 方法签名与 StreamRenderer 一致"""
    import inspect

    # 获取 StreamRenderer 的抽象方法
    stream_renderer_methods = {
        name: method
        for name, method in inspect.getmembers(StreamRenderer, predicate=inspect.isfunction)
        if not name.startswith('_')
    }

    # 获取 AdvancedTerminalRenderer 的方法
    advanced_methods = {
        name: method
        for name, method in inspect.getmembers(AdvancedTerminalRenderer, predicate=inspect.isfunction)
        if not name.startswith('_')
    }

    # 验证所有抽象方法都被实现
    for method_name in stream_renderer_methods:
        assert method_name in advanced_methods, f"方法 {method_name} 未在 AdvancedTerminalRenderer 中实现"


# ============== _build_terminal_renderer 配置测试 ==============


def test_build_terminal_renderer_disabled() -> None:
    """测试禁用控制台渲染器时返回 None"""
    from cursor.client import CursorAgentClient, CursorAgentConfig

    config = CursorAgentConfig(
        stream_console_renderer=False,  # 禁用控制台渲染器
    )
    client = CursorAgentClient(config=config)

    renderer = client._build_terminal_renderer()
    assert renderer is None


def test_build_terminal_renderer_basic() -> None:
    """测试基础渲染器配置返回 TerminalStreamRenderer"""
    from cursor.client import CursorAgentClient, CursorAgentConfig

    config = CursorAgentConfig(
        stream_console_renderer=True,   # 启用控制台渲染器
        stream_advanced_renderer=False,  # 使用基础渲染器
        stream_console_verbose=True,     # 详细模式
    )
    client = CursorAgentClient(config=config)

    renderer = client._build_terminal_renderer()
    assert renderer is not None
    assert isinstance(renderer, TerminalStreamRenderer)
    assert renderer.verbose is True


def test_build_terminal_renderer_advanced() -> None:
    """测试高级渲染器配置返回 AdvancedTerminalRenderer"""
    from cursor.client import CursorAgentClient, CursorAgentConfig

    config = CursorAgentConfig(
        stream_console_renderer=True,    # 启用控制台渲染器
        stream_advanced_renderer=True,   # 使用高级渲染器
        stream_color_enabled=False,      # 禁用颜色
        stream_typing_effect=True,       # 启用打字效果
        stream_show_status_bar=False,    # 禁用状态栏
    )
    client = CursorAgentClient(config=config)

    renderer = client._build_terminal_renderer()
    assert renderer is not None
    assert isinstance(renderer, AdvancedTerminalRenderer)
    assert renderer.use_color is False
    assert renderer.typing_delay > 0  # 打字效果启用时应有延迟
    assert renderer.show_status_bar is False


def test_build_terminal_renderer_advanced_no_typing() -> None:
    """测试高级渲染器禁用打字效果时延迟为 0"""
    from cursor.client import CursorAgentClient, CursorAgentConfig

    config = CursorAgentConfig(
        stream_console_renderer=True,
        stream_advanced_renderer=True,
        stream_typing_effect=False,  # 禁用打字效果
    )
    client = CursorAgentClient(config=config)

    renderer = client._build_terminal_renderer()
    assert renderer is not None
    assert isinstance(renderer, AdvancedTerminalRenderer)
    assert renderer.typing_delay == 0.0


def test_build_terminal_renderer_returns_stream_renderer() -> None:
    """测试 _build_terminal_renderer 返回值符合 StreamRenderer 接口"""
    from cursor.client import CursorAgentClient, CursorAgentConfig

    # 测试基础渲染器
    config_basic = CursorAgentConfig(
        stream_console_renderer=True,
        stream_advanced_renderer=False,
    )
    client_basic = CursorAgentClient(config=config_basic)
    renderer_basic = client_basic._build_terminal_renderer()
    assert renderer_basic is not None
    assert isinstance(renderer_basic, StreamRenderer)

    # 测试高级渲染器
    config_advanced = CursorAgentConfig(
        stream_console_renderer=True,
        stream_advanced_renderer=True,
    )
    client_advanced = CursorAgentClient(config=config_advanced)
    renderer_advanced = client_advanced._build_terminal_renderer()
    assert renderer_advanced is not None
    assert isinstance(renderer_advanced, StreamRenderer)


# ============== 日志聚合在不同配置下的行为测试 ==============


def test_stream_event_logger_aggregation_with_multiple_events(tmp_path: Path) -> None:
    """测试聚合模式下多个 ASSISTANT 事件后跟不同类型事件"""
    detail_dir = tmp_path / "detail"
    logger = StreamEventLogger(
        agent_id="agg-multi",
        agent_role="worker",
        agent_name="test",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir="",
        aggregate_assistant_messages=True,
    )

    # 发送多个 ASSISTANT 事件
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Part 1 "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Part 2 "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Part 3"))

    # 发送 TOOL_STARTED，触发刷新
    logger.handle_event(
        StreamEvent(
            type=StreamEventType.TOOL_STARTED,
            tool_call=ToolCallInfo(tool_type="shell", args={"command": "ls"}),
        )
    )

    # 发送更多 ASSISTANT 事件
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Part 4 "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Part 5"))

    # 发送 RESULT，触发刷新
    logger.handle_event(StreamEvent(type=StreamEventType.RESULT, duration_ms=100))

    logger.close()

    detail_files = list(detail_dir.glob("*.log"))
    assert len(detail_files) == 1
    detail_content = detail_files[0].read_text(encoding="utf-8")

    # 验证聚合消息被正确记录
    assert "Part 1 Part 2 Part 3" in detail_content
    assert "Part 4 Part 5" in detail_content
    assert "工具开始" in detail_content
    assert "完成" in detail_content


def test_stream_event_logger_aggregation_empty_content(tmp_path: Path) -> None:
    """测试聚合模式下空内容的 ASSISTANT 事件"""
    detail_dir = tmp_path / "detail"
    logger = StreamEventLogger(
        agent_id="agg-empty",
        agent_role="planner",
        agent_name="",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir="",
        aggregate_assistant_messages=True,
    )

    # 发送空内容
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content=""))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content=""))

    # 发送有内容的事件
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Valid content"))

    logger.close()

    detail_files = list(detail_dir.glob("*.log"))
    assert len(detail_files) == 1
    detail_content = detail_files[0].read_text(encoding="utf-8")

    # 验证只有有效内容被记录
    assert "Valid content" in detail_content


def test_stream_event_logger_aggregation_interleaved_events(tmp_path: Path) -> None:
    """测试聚合模式下交错的事件类型"""
    detail_dir = tmp_path / "detail"
    logger = StreamEventLogger(
        agent_id="agg-interleaved",
        agent_role="reviewer",
        agent_name="",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir="",
        aggregate_assistant_messages=True,
    )

    # 交错发送不同类型的事件
    logger.handle_event(StreamEvent(type=StreamEventType.SYSTEM_INIT, model="test-model"))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="First "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="message"))
    logger.handle_event(
        StreamEvent(
            type=StreamEventType.TOOL_STARTED,
            tool_call=ToolCallInfo(tool_type="read", path="file.py"),
        )
    )
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Second "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="message"))
    logger.handle_event(
        StreamEvent(
            type=StreamEventType.DIFF,
            diff_info=DiffInfo(path="file.py", old_string="old", new_string="new"),
        )
    )
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Third message"))
    logger.handle_event(StreamEvent(type=StreamEventType.RESULT, duration_ms=200))

    logger.close()

    detail_files = list(detail_dir.glob("*.log"))
    assert len(detail_files) == 1
    detail_content = detail_files[0].read_text(encoding="utf-8")

    # 验证所有消息都被正确聚合和记录
    assert "First message" in detail_content
    assert "Second message" in detail_content
    assert "Third message" in detail_content
    assert "test-model" in detail_content
    assert "工具开始" in detail_content


def test_stream_event_logger_no_aggregation_vs_aggregation(tmp_path: Path) -> None:
    """比较聚合模式和非聚合模式的日志行数差异"""
    # 聚合模式
    agg_dir = tmp_path / "agg"
    logger_agg = StreamEventLogger(
        agent_id="compare-agg",
        agent_role="worker",
        agent_name="",
        console=False,
        detail_dir=str(agg_dir),
        raw_dir="",
        aggregate_assistant_messages=True,
    )

    # 非聚合模式
    no_agg_dir = tmp_path / "no_agg"
    logger_no_agg = StreamEventLogger(
        agent_id="compare-no-agg",
        agent_role="worker",
        agent_name="",
        console=False,
        detail_dir=str(no_agg_dir),
        raw_dir="",
        aggregate_assistant_messages=False,
    )

    # 发送相同的事件序列
    events = [
        StreamEvent(type=StreamEventType.ASSISTANT, content="Line 1"),
        StreamEvent(type=StreamEventType.ASSISTANT, content="Line 2"),
        StreamEvent(type=StreamEventType.ASSISTANT, content="Line 3"),
    ]

    for event in events:
        logger_agg.handle_event(event)
        logger_no_agg.handle_event(event)

    logger_agg.close()
    logger_no_agg.close()

    agg_files = list(agg_dir.glob("*.log"))
    no_agg_files = list(no_agg_dir.glob("*.log"))

    assert len(agg_files) == 1
    assert len(no_agg_files) == 1

    agg_content = agg_files[0].read_text(encoding="utf-8")
    no_agg_content = no_agg_files[0].read_text(encoding="utf-8")

    agg_lines = agg_content.strip().split("\n")
    no_agg_lines = no_agg_content.strip().split("\n")

    # 聚合模式应该产生更少的行数
    assert len(agg_lines) <= len(no_agg_lines)


def test_stream_event_logger_aggregation_with_raw_log(tmp_path: Path) -> None:
    """测试聚合模式不影响 raw 日志的逐行记录"""
    detail_dir = tmp_path / "detail"
    raw_dir = tmp_path / "raw"
    logger = StreamEventLogger(
        agent_id="agg-raw",
        agent_role="worker",
        agent_name="",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir=str(raw_dir),
        aggregate_assistant_messages=True,
    )

    # 发送多个事件并写入 raw 日志
    raw_lines = [
        '{"type":"assistant","message":{"content":[{"text":"Line 1"}]}}',
        '{"type":"assistant","message":{"content":[{"text":"Line 2"}]}}',
        '{"type":"assistant","message":{"content":[{"text":"Line 3"}]}}',
    ]

    for raw_line in raw_lines:
        logger.handle_raw_line(raw_line)

    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Line 1"))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Line 2"))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Line 3"))

    logger.close()

    raw_files = list(raw_dir.glob("*.jsonl"))
    assert len(raw_files) == 1
    raw_content = raw_files[0].read_text(encoding="utf-8")
    raw_file_lines = raw_content.strip().split("\n")

    # raw 日志应该保持每行记录
    assert len(raw_file_lines) == 3


def test_stream_event_logger_aggregation_only_assistant_events(tmp_path: Path) -> None:
    """测试只有 ASSISTANT 事件时，close() 正确刷新缓冲区"""
    detail_dir = tmp_path / "detail"
    logger = StreamEventLogger(
        agent_id="only-assistant",
        agent_role="worker",
        agent_name="",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir="",
        aggregate_assistant_messages=True,
    )

    # 只发送 ASSISTANT 事件
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Only "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="assistant "))
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="messages"))

    # 不发送任何其他事件，直接关闭
    logger.close()

    detail_files = list(detail_dir.glob("*.log"))
    assert len(detail_files) == 1
    detail_content = detail_files[0].read_text(encoding="utf-8")

    # 验证聚合后的消息被正确写入
    assert "Only assistant messages" in detail_content


def test_stream_event_logger_aggregation_error_event_flushes(tmp_path: Path) -> None:
    """测试 ERROR 事件触发聚合缓冲区刷新"""
    detail_dir = tmp_path / "detail"
    logger = StreamEventLogger(
        agent_id="error-flush",
        agent_role="worker",
        agent_name="",
        console=False,
        detail_dir=str(detail_dir),
        raw_dir="",
        aggregate_assistant_messages=True,
    )

    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="Before error"))
    logger.handle_event(
        StreamEvent(type=StreamEventType.ERROR, data={"error": "Something went wrong"})
    )
    logger.handle_event(StreamEvent(type=StreamEventType.ASSISTANT, content="After error"))

    logger.close()

    detail_files = list(detail_dir.glob("*.log"))
    assert len(detail_files) == 1
    detail_content = detail_files[0].read_text(encoding="utf-8")

    # 验证错误前的消息被刷新
    assert "Before error" in detail_content
    assert "错误: Something went wrong" in detail_content
    assert "After error" in detail_content
