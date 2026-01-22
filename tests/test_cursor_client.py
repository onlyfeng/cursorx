"""CursorAgentClient 测试

覆盖 CursorAgentClient 的核心功能：
- execute() 成功/失败场景
- stream_execute() 流式输出 (通过 output_format="stream-json")
- _build_prompt() 参数构建
- _build_command() 参数构建 (通过 _try_agent_cli)
- execute_with_retry() 重试机制
- check_agent_available() CLI 可用性检查
- parse_json_output() / parse_stream_json_output() 输出解析
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cursor.client import (
    CursorAgentClient,
    CursorAgentConfig,
    CursorAgentPool,
    CursorAgentResult,
    ModelPresets,
)
from cursor.executor import AgentResult


# ==================== CursorAgentConfig 测试 ====================


class TestCursorAgentConfig:
    """CursorAgentConfig 配置类测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = CursorAgentConfig()

        assert config.agent_path == "agent"
        assert config.model == "opus-4.5-thinking"
        assert config.timeout == 300
        assert config.max_retries == 3
        assert config.output_format == "text"
        assert config.non_interactive is True
        assert config.force_write is False

    def test_custom_config(self):
        """测试自定义配置"""
        config = CursorAgentConfig(
            agent_path="/custom/path/agent",
            model="gpt-5.2-high",
            timeout=600,
            max_retries=5,
            output_format="json",
            force_write=True,
            mode="plan",
        )

        assert config.agent_path == "/custom/path/agent"
        assert config.model == "gpt-5.2-high"
        assert config.timeout == 600
        assert config.max_retries == 5
        assert config.output_format == "json"
        assert config.force_write is True
        assert config.mode == "plan"

    def test_stream_config(self):
        """测试流式输出配置"""
        config = CursorAgentConfig(
            output_format="stream-json",
            stream_partial_output=True,
            stream_events_enabled=True,
            stream_log_console=False,
        )

        assert config.output_format == "stream-json"
        assert config.stream_partial_output is True
        assert config.stream_events_enabled is True
        assert config.stream_log_console is False

    def test_cloud_config(self):
        """测试云端配置"""
        config = CursorAgentConfig(
            cloud_enabled=True,
            cloud_timeout=1200,
            execution_mode="cloud",
        )

        assert config.cloud_enabled is True
        assert config.cloud_timeout == 1200
        assert config.execution_mode == "cloud"


class TestModelPresets:
    """模型预设测试"""

    def test_planner_preset(self):
        """测试规划者预设"""
        config = ModelPresets.PLANNER
        assert config.model == "gpt-5.2-high"
        assert config.timeout == 180

    def test_worker_preset(self):
        """测试执行者预设"""
        config = ModelPresets.WORKER
        assert config.model == "opus-4.5-thinking"
        assert config.timeout == 300

    def test_reviewer_preset(self):
        """测试评审者预设"""
        config = ModelPresets.REVIEWER
        assert config.model == "opus-4.5-thinking"
        assert config.timeout == 120


# ==================== CursorAgentResult 测试 ====================


class TestCursorAgentResult:
    """CursorAgentResult 结果类测试"""

    def test_success_result(self):
        """测试成功结果"""
        result = CursorAgentResult(
            success=True,
            output="任务完成",
            exit_code=0,
            duration=1.5,
        )

        assert result.success is True
        assert result.output == "任务完成"
        assert result.error is None
        assert result.exit_code == 0
        assert result.duration == 1.5

    def test_failure_result(self):
        """测试失败结果"""
        result = CursorAgentResult(
            success=False,
            output="",
            error="执行超时",
            exit_code=-1,
        )

        assert result.success is False
        assert result.error == "执行超时"
        assert result.exit_code == -1

    def test_result_with_files(self):
        """测试包含文件修改的结果"""
        result = CursorAgentResult(
            success=True,
            output="已修改文件",
            files_modified=["src/main.py", "tests/test_main.py"],
            command_used="agent -p '...' --model opus-4.5-thinking",
        )

        assert len(result.files_modified) == 2
        assert "src/main.py" in result.files_modified
        assert result.command_used != ""

    def test_result_timestamps(self):
        """测试结果时间戳"""
        started = datetime.now()
        result = CursorAgentResult(
            success=True,
            output="",
            started_at=started,
        )

        assert result.started_at == started
        assert result.completed_at is None

    def test_result_with_session_id(self):
        """测试包含 session_id 的结果"""
        result = CursorAgentResult(
            success=True,
            output="完成",
            session_id="test-session-12345",
        )

        assert result.session_id == "test-session-12345"

    def test_result_with_files_edited(self):
        """测试包含 files_edited 的结果"""
        result = CursorAgentResult(
            success=True,
            output="已编辑文件",
            files_modified=["new_file.py"],
            files_edited=["edited_file.py", "another.py"],
        )

        assert "new_file.py" in result.files_modified
        assert "edited_file.py" in result.files_edited
        assert "another.py" in result.files_edited

    def test_result_full_stream_json_fields(self):
        """测试完整的 stream-json 相关字段"""
        result = CursorAgentResult(
            success=True,
            output="任务完成",
            session_id="session-uuid-abc",
            files_modified=["created.py"],
            files_edited=["modified.py"],
            command_used="agent -p '...' --model opus-4.5-thinking",
        )

        assert result.session_id == "session-uuid-abc"
        assert result.files_modified == ["created.py"]
        assert result.files_edited == ["modified.py"]


# ==================== AgentResult.from_cli_result 测试 ====================


class TestAgentResultFromCliResult:
    """AgentResult.from_cli_result 转换测试"""

    def test_from_cli_result_basic(self):
        """测试基本转换"""
        cli_result = CursorAgentResult(
            success=True,
            output="完成",
            exit_code=0,
            duration=1.5,
        )

        agent_result = AgentResult.from_cli_result(cli_result)

        assert agent_result.success is True
        assert agent_result.output == "完成"
        assert agent_result.exit_code == 0
        assert agent_result.duration == 1.5
        assert agent_result.executor_type == "cli"

    def test_from_cli_result_with_session_id(self):
        """测试透传 session_id"""
        cli_result = CursorAgentResult(
            success=True,
            output="任务完成",
            session_id="session-from-stream-json",
        )

        agent_result = AgentResult.from_cli_result(cli_result)

        assert agent_result.session_id == "session-from-stream-json"

    def test_from_cli_result_merges_files(self):
        """测试合并 files_modified 和 files_edited"""
        cli_result = CursorAgentResult(
            success=True,
            output="文件操作完成",
            files_modified=["new_file.py"],
            files_edited=["edited1.py", "edited2.py"],
        )

        agent_result = AgentResult.from_cli_result(cli_result)

        # files_modified 应包含所有文件
        assert "new_file.py" in agent_result.files_modified
        assert "edited1.py" in agent_result.files_modified
        assert "edited2.py" in agent_result.files_modified

    def test_from_cli_result_no_duplicates_in_merged_files(self):
        """测试合并时不产生重复"""
        cli_result = CursorAgentResult(
            success=True,
            output="文件操作完成",
            files_modified=["shared.py", "only_modified.py"],
            files_edited=["shared.py", "only_edited.py"],
        )

        agent_result = AgentResult.from_cli_result(cli_result)

        # 文件不应该重复
        assert agent_result.files_modified.count("shared.py") == 1
        assert "only_modified.py" in agent_result.files_modified
        assert "only_edited.py" in agent_result.files_modified

    def test_from_cli_result_empty_files(self):
        """测试空文件列表"""
        cli_result = CursorAgentResult(
            success=True,
            output="无文件操作",
        )

        agent_result = AgentResult.from_cli_result(cli_result)

        assert agent_result.files_modified == []
        assert agent_result.session_id is None

    def test_from_cli_result_preserves_timestamps(self):
        """测试保留时间戳"""
        started = datetime.now()
        completed = datetime.now()

        cli_result = CursorAgentResult(
            success=True,
            output="完成",
            started_at=started,
            completed_at=completed,
        )

        agent_result = AgentResult.from_cli_result(cli_result)

        assert agent_result.started_at == started
        assert agent_result.completed_at == completed

    def test_from_cli_result_with_error(self):
        """测试包含错误的结果转换"""
        cli_result = CursorAgentResult(
            success=False,
            output="",
            error="执行失败",
            exit_code=-1,
        )

        agent_result = AgentResult.from_cli_result(cli_result)

        assert agent_result.success is False
        assert agent_result.error == "执行失败"
        assert agent_result.exit_code == -1


# ==================== CursorAgentClient 初始化测试 ====================


class TestCursorAgentClientInit:
    """CursorAgentClient 初始化测试"""

    def test_default_init(self):
        """测试默认初始化"""
        client = CursorAgentClient()

        assert client.config is not None
        assert client._session_id is not None
        assert len(client._session_id) == 16  # 8 bytes hex

    def test_custom_config_init(self):
        """测试自定义配置初始化"""
        config = CursorAgentConfig(model="gpt-5.2-high")
        client = CursorAgentClient(config=config)

        assert client.config.model == "gpt-5.2-high"

    @patch("shutil.which")
    def test_find_agent_executable_with_which(self, mock_which):
        """测试通过 which 查找可执行文件"""
        mock_which.return_value = "/usr/local/bin/agent"

        with patch("os.path.isfile", return_value=True):
            client = CursorAgentClient()
            assert client._agent_path == "/usr/local/bin/agent"

    def test_find_agent_executable_fallback(self):
        """测试可执行文件查找回退"""
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile", return_value=False):
                client = CursorAgentClient()
                # 回退到默认路径
                assert client._agent_path == "agent"


# ==================== _build_prompt() 测试 ====================


class TestBuildPrompt:
    """_build_prompt() 方法测试"""

    def test_simple_instruction(self):
        """测试简单指令"""
        client = CursorAgentClient()
        prompt = client._build_prompt("请分析代码")

        assert prompt == "请分析代码"

    def test_instruction_with_context(self):
        """测试带上下文的指令"""
        client = CursorAgentClient()
        context = {
            "task_id": "task-001",
            "priority": "high",
        }
        prompt = client._build_prompt("请执行任务", context)

        assert "请执行任务" in prompt
        assert "## 上下文信息" in prompt
        assert "task_id" in prompt
        assert "task-001" in prompt

    def test_instruction_with_complex_context(self):
        """测试带复杂上下文的指令"""
        client = CursorAgentClient()
        context = {
            "files": ["src/main.py", "src/utils.py"],
            "config": {"debug": True, "verbose": False},
        }
        prompt = client._build_prompt("请修改代码", context)

        assert "请修改代码" in prompt
        assert "```json" in prompt
        assert "src/main.py" in prompt
        assert '"debug": true' in prompt


# ==================== execute() 成功场景测试 ====================


class TestExecuteSuccess:
    """execute() 成功场景测试"""

    @pytest.mark.asyncio
    async def test_execute_success_with_cli(self):
        """测试 CLI 执行成功"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Task completed successfully", b"")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await client.execute("请分析代码")

        assert result.success is True
        assert result.output == "Task completed successfully"
        assert result.exit_code == 0
        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_execute_with_working_directory(self):
        """测试指定工作目录执行"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            await client.execute("test", working_directory="/tmp/test")

            # 验证 cwd 参数
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["cwd"] == "/tmp/test"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """测试带上下文执行"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Done", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await client.execute(
                "请修改文件",
                context={"file": "main.py"},
            )

        assert result.success is True


# ==================== execute() 失败场景测试 ====================


class TestExecuteFailure:
    """execute() 失败场景测试"""

    @pytest.mark.asyncio
    async def test_execute_cli_returns_error(self):
        """测试 CLI 返回错误"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Error: Invalid command")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await client.execute("无效命令")

        assert result.success is False
        assert result.exit_code == 1
        assert "Invalid command" in result.error

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """测试执行超时"""
        config = CursorAgentConfig(timeout=1)
        client = CursorAgentClient(config=config)

        async def slow_communicate():
            await asyncio.sleep(10)
            return (b"", b"")

        mock_process = AsyncMock()
        mock_process.communicate = slow_communicate

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await client.execute("慢任务", timeout=1)

        assert result.success is False
        assert "超时" in result.error

    @pytest.mark.asyncio
    async def test_execute_cli_not_found(self):
        """测试 CLI 未找到"""
        client = CursorAgentClient()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("agent not found"),
        ):
            result = await client.execute("测试")

        # 应该回退到模拟模式
        assert result.success is True
        assert "[Mock]" in result.output

    @pytest.mark.asyncio
    async def test_execute_exception(self):
        """测试执行异常"""
        client = CursorAgentClient()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = await client.execute("测试")

        # 回退到模拟模式
        assert result.success is True


# ==================== 流式输出测试 ====================


class TestStreamExecute:
    """流式输出测试 (output_format="stream-json")"""

    @pytest.mark.asyncio
    async def test_stream_json_output(self):
        """测试 stream-json 输出格式"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 模拟 stream-json 输出
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "正在分析..."}]}}',
            '{"type": "result", "duration_ms": 1234}',
        ]
        stream_output = "\n".join(stream_lines).encode()

        async def mock_readline():
            """模拟逐行读取"""
            for line in stream_lines:
                yield (line + "\n").encode()
            yield b""

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            # 禁用流式日志以简化测试
            client.config.stream_events_enabled = False
            result = await client.execute("分析代码")

        assert result.success is True
        assert "正在分析..." in result.output

    @pytest.mark.asyncio
    async def test_stream_partial_output_config(self):
        """测试增量流式输出配置"""
        config = CursorAgentConfig(
            output_format="stream-json",
            stream_partial_output=True,
        )
        client = CursorAgentClient(config=config)

        assert client.config.stream_partial_output is True

    @pytest.mark.asyncio
    async def test_stream_json_extract_session_id(self):
        """测试 stream-json 模式下能正确提取 session_id"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 模拟包含 session_id 的 stream-json 输出
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking", "session_id": "test-session-uuid-12345"}',
            '{"type": "assistant", "message": {"content": [{"text": "完成"}]}}',
            '{"type": "result", "duration_ms": 100}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("测试 session_id")

        assert result.success is True
        assert result.session_id == "test-session-uuid-12345"

    @pytest.mark.asyncio
    async def test_stream_json_extract_files_modified_and_edited(self):
        """测试 stream-json 模式下能正确提取 files_modified 和 files_edited"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 模拟包含文件操作的 stream-json 输出
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking", "session_id": "session-abc"}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"writeToolCall": {"args": {"path": "new_file.py"}, "result": {"success": {}}}}}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"strReplaceToolCall": {"args": {"path": "existing.py", "old_string": "old", "new_string": "new"}, "result": {"success": {}}}}}',
            '{"type": "diff", "path": "another.py", "old_string": "a", "new_string": "b"}',
            '{"type": "result", "duration_ms": 200}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("测试文件操作")

        assert result.success is True
        assert result.session_id == "session-abc"
        # files_modified 应包含 write 操作的文件
        assert "new_file.py" in result.files_modified
        # files_edited 应包含 str_replace 和 diff 操作的文件
        assert "existing.py" in result.files_edited
        assert "another.py" in result.files_edited

    @pytest.mark.asyncio
    async def test_stream_json_files_deduplication(self):
        """测试 stream-json 模式下文件列表去重"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 模拟对同一文件多次操作
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "tool_call", "subtype": "started", "tool_call": {"strReplaceToolCall": {"args": {"path": "same_file.py", "old_string": "a", "new_string": "b"}}}}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"strReplaceToolCall": {"args": {"path": "same_file.py", "old_string": "a", "new_string": "b"}, "result": {"success": {}}}}}',
            '{"type": "diff", "path": "same_file.py", "old_string": "c", "new_string": "d"}',
            '{"type": "result", "duration_ms": 100}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("测试去重")

        assert result.success is True
        # 文件应该只出现一次（去重）
        assert result.files_edited.count("same_file.py") == 1


# ==================== 命令构建测试 ====================


class TestBuildCommand:
    """命令构建测试 (通过 _try_agent_cli 验证)"""

    @pytest.mark.asyncio
    async def test_build_command_basic(self):
        """测试基本命令构建"""
        config = CursorAgentConfig(
            model="opus-4.5-thinking",
            non_interactive=True,
        )
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            await client.execute("测试")

            # 验证命令参数
            call_args = mock_exec.call_args.args
            assert "-p" in call_args
            assert "--model" in call_args
            assert "opus-4.5-thinking" in call_args

    @pytest.mark.asyncio
    async def test_build_command_with_force(self):
        """测试带 --force 的命令构建"""
        config = CursorAgentConfig(force_write=True)
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            await client.execute("修改文件")

            call_args = mock_exec.call_args.args
            assert "--force" in call_args

    @pytest.mark.asyncio
    async def test_build_command_with_mode(self):
        """测试带 --mode 的命令构建"""
        config = CursorAgentConfig(mode="plan")
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            await client.execute("规划任务")

            call_args = mock_exec.call_args.args
            assert "--mode" in call_args
            assert "plan" in call_args

    @pytest.mark.asyncio
    async def test_build_command_with_output_format(self):
        """测试带 --output-format 的命令构建"""
        config = CursorAgentConfig(output_format="json")
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"result": "ok"}', b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            await client.execute("测试")

            call_args = mock_exec.call_args.args
            assert "--output-format" in call_args
            assert "json" in call_args

    @pytest.mark.asyncio
    async def test_build_command_with_resume(self):
        """测试恢复会话的命令构建（通过 config）"""
        config = CursorAgentConfig(resume_thread_id="session-123")
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            await client.execute("继续")

            call_args = mock_exec.call_args.args
            assert "--resume" in call_args
            assert "session-123" in call_args

    @pytest.mark.asyncio
    async def test_build_command_with_session_id_param(self):
        """测试通过 session_id 参数恢复会话的命令构建"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            # 通过 session_id 参数传递会话 ID
            await client.execute("继续任务", session_id="param-session-456")

            call_args = mock_exec.call_args.args
            assert "--resume" in call_args
            assert "param-session-456" in call_args

    @pytest.mark.asyncio
    async def test_session_id_param_overrides_config(self):
        """测试 session_id 参数优先于 config.resume_thread_id"""
        config = CursorAgentConfig(resume_thread_id="config-session-old")
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            # session_id 参数应该覆盖 config 中的值
            await client.execute("继续任务", session_id="param-session-new")

            call_args = mock_exec.call_args.args
            assert "--resume" in call_args
            assert "param-session-new" in call_args
            # 确保不是 config 中的旧值
            assert "config-session-old" not in call_args

    @pytest.mark.asyncio
    async def test_execute_with_session_id_no_exception(self):
        """测试带 session_id 的本地执行不应抛异常"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Session resumed successfully", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            # 带 session_id 的执行应该正常工作，不抛异常
            result = await client.execute(
                "继续之前的任务",
                session_id="test-session-789"
            )

        assert result.success is True
        assert result.output == "Session resumed successfully"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_build_command_with_api_key(self):
        """测试 API 密钥环境变量"""
        config = CursorAgentConfig(api_key="test-key-123")
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"OK", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            await client.execute("测试")

            # 验证环境变量
            call_kwargs = mock_exec.call_args.kwargs
            assert "env" in call_kwargs
            assert call_kwargs["env"]["CURSOR_API_KEY"] == "test-key-123"


# ==================== execute_with_retry() 测试 ====================


class TestExecuteWithRetry:
    """execute_with_retry() 重试机制测试"""

    @pytest.mark.asyncio
    async def test_retry_success_first_try(self):
        """测试首次成功"""
        client = CursorAgentClient()

        with patch.object(
            client, "execute", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = CursorAgentResult(
                success=True, output="成功"
            )

            result = await client.execute_with_retry("测试")

        assert result.success is True
        assert mock_execute.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """测试失败后重试成功"""
        config = CursorAgentConfig(max_retries=3, retry_delay=0.1)
        client = CursorAgentClient(config=config)

        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return CursorAgentResult(success=False, error="临时错误")
            return CursorAgentResult(success=True, output="成功")

        with patch.object(client, "execute", side_effect=mock_execute):
            result = await client.execute_with_retry("测试")

        assert result.success is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_all_failed(self):
        """测试所有重试都失败"""
        config = CursorAgentConfig(max_retries=3, retry_delay=0.1)
        client = CursorAgentClient(config=config)

        with patch.object(
            client, "execute", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = CursorAgentResult(
                success=False, error="持续失败"
            )

            result = await client.execute_with_retry("测试")

        assert result.success is False
        assert mock_execute.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_custom_retries(self):
        """测试自定义重试次数"""
        client = CursorAgentClient()

        with patch.object(
            client, "execute", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = CursorAgentResult(
                success=False, error="错误"
            )
            # Mock asyncio.sleep 以避免测试超时（指数退避总等待时间可能超过 30s）
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.execute_with_retry("测试", max_retries=5)

        assert mock_execute.call_count == 5


# ==================== check_agent_available() 测试 ====================


class TestCheckAgentAvailable:
    """check_agent_available() 测试"""

    def test_agent_available(self):
        """测试 agent CLI 可用"""
        client = CursorAgentClient()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = client.check_agent_available()

        assert result is True
        mock_run.assert_called_once()

    def test_agent_not_available(self):
        """测试 agent CLI 不可用"""
        client = CursorAgentClient()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            result = client.check_agent_available()

        assert result is False

    def test_agent_check_exception(self):
        """测试检查异常"""
        client = CursorAgentClient()

        with patch(
            "subprocess.run",
            side_effect=FileNotFoundError("agent not found"),
        ):
            result = client.check_agent_available()

        assert result is False


# ==================== 输出解析测试 ====================


class TestStreamJsonFilesModifiedSessionIdExtraction:
    """stream-json 输入样本驱动测试：files_modified/session_id 提取"""

    @pytest.mark.asyncio
    async def test_extract_session_id_from_system_init(self):
        """测试从 system/init 事件提取 session_id"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 模拟 stream-json 输出，包含 session_id
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking", "session_id": "sess-abc123"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "开始处理..."}]}}',
            '{"type": "result", "duration_ms": 500}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("测试任务")

        assert result.success is True
        assert result.session_id == "sess-abc123"

    @pytest.mark.asyncio
    async def test_extract_files_modified_from_tool_calls(self):
        """测试从 tool_call 事件提取 files_modified"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 模拟 stream-json 输出，包含写入文件的工具调用
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "tool_call", "subtype": "started", "tool_call": {"writeToolCall": {"args": {"path": "src/main.py"}}}}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"writeToolCall": {"args": {"path": "src/main.py"}, "result": {"success": {"linesCreated": 50}}}}}',
            '{"type": "tool_call", "subtype": "started", "tool_call": {"writeToolCall": {"args": {"path": "src/utils.py"}}}}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"writeToolCall": {"args": {"path": "src/utils.py"}, "result": {"success": {"linesCreated": 20}}}}}',
            '{"type": "result", "duration_ms": 1000}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("创建文件")

        assert result.success is True
        assert "src/main.py" in result.files_modified
        assert "src/utils.py" in result.files_modified

    @pytest.mark.asyncio
    async def test_extract_files_edited_from_str_replace_tool(self):
        """测试从 str_replace 工具调用提取 files_edited"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 模拟 stream-json 输出，包含编辑文件的工具调用
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "tool_call", "subtype": "started", "tool_call": {"strReplaceToolCall": {"args": {"path": "config.yaml"}}}}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"strReplaceToolCall": {"args": {"path": "config.yaml"}, "result": {"success": true}}}}',
            '{"type": "result", "duration_ms": 300}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("修改配置")

        assert result.success is True
        assert "config.yaml" in result.files_edited

    @pytest.mark.asyncio
    async def test_combined_session_id_and_files_modified(self):
        """测试同时提取 session_id 和 files_modified"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 完整的 stream-json 样本
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking", "session_id": "combined-sess-xyz"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "开始创建文件..."}]}}',
            '{"type": "tool_call", "subtype": "started", "tool_call": {"writeToolCall": {"args": {"path": "new_file.py"}}}}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"writeToolCall": {"args": {"path": "new_file.py"}, "result": {"success": {"linesCreated": 100}}}}}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "文件创建完成"}]}}',
            '{"type": "result", "duration_ms": 2000}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("创建新文件")

        # 验证 session_id 和 files_modified 都被正确提取
        assert result.success is True
        assert result.session_id == "combined-sess-xyz"
        assert "new_file.py" in result.files_modified
        assert "开始创建文件..." in result.output or "文件创建完成" in result.output

    @pytest.mark.asyncio
    async def test_files_modified_deduplication(self):
        """测试 files_modified 去重"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 同一文件多次写入应该去重
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "tool_call", "subtype": "started", "tool_call": {"writeToolCall": {"args": {"path": "repeated.py"}}}}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"writeToolCall": {"args": {"path": "repeated.py"}, "result": {"success": {}}}}}',
            '{"type": "tool_call", "subtype": "started", "tool_call": {"writeToolCall": {"args": {"path": "repeated.py"}}}}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"writeToolCall": {"args": {"path": "repeated.py"}, "result": {"success": {}}}}}',
            '{"type": "result", "duration_ms": 100}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("重复写入")

        # 验证去重：同一文件只出现一次
        assert result.files_modified.count("repeated.py") == 1

    @pytest.mark.asyncio
    async def test_no_session_id_when_not_present(self):
        """测试没有 session_id 时返回 None"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # system/init 事件没有 session_id
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "result", "duration_ms": 100}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[(line + "\n").encode() for line in stream_lines] + [b""]
        )
        mock_stderr = AsyncMock()
        mock_stderr.readline = AsyncMock(return_value=b"")

        mock_process = AsyncMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("无 session")

        assert result.success is True
        assert result.session_id is None


class TestOutputParsing:
    """输出解析测试"""

    def test_parse_json_output_success(self):
        """测试解析成功的 JSON 输出"""
        output = json.dumps({
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 1234,
            "result": "任务完成",
            "session_id": "test-session",
        })

        result = CursorAgentClient.parse_json_output(output)

        assert result["type"] == "result"
        assert result["subtype"] == "success"
        assert result["result"] == "任务完成"

    def test_parse_json_output_invalid(self):
        """测试解析无效的 JSON 输出"""
        output = "This is not JSON"

        result = CursorAgentClient.parse_json_output(output)

        assert result["type"] == "error"
        assert result["result"] == output

    def test_parse_stream_json_output(self):
        """测试解析 stream-json 输出"""
        output = """{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}
{"type": "assistant", "message": {"content": [{"text": "分析中..."}]}}
{"type": "tool_call", "subtype": "started", "tool_call": {"readToolCall": {"args": {"path": "main.py"}}}}
{"type": "result", "duration_ms": 5000}"""

        events = CursorAgentClient.parse_stream_json_output(output)

        assert len(events) == 4
        assert events[0]["type"] == "system"
        assert events[1]["type"] == "assistant"
        assert events[2]["type"] == "tool_call"
        assert events[3]["type"] == "result"

    def test_parse_stream_json_output_with_invalid_lines(self):
        """测试解析包含无效行的 stream-json 输出"""
        output = """{"type": "system", "subtype": "init"}
invalid json line
{"type": "result", "duration_ms": 100}"""

        events = CursorAgentClient.parse_stream_json_output(output)

        # 应该跳过无效行
        assert len(events) == 2
        assert events[0]["type"] == "system"
        assert events[1]["type"] == "result"


# ==================== list_models() 测试 ====================


class TestListModels:
    """list_models() 测试"""

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """测试获取模型列表成功"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"gpt-4\nopus-4.5\ngpt-5.2-high\n", b"")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            models = await client.list_models()

        assert len(models) == 3
        assert "gpt-4" in models
        assert "opus-4.5" in models

    @pytest.mark.asyncio
    async def test_list_models_failure(self):
        """测试获取模型列表失败"""
        client = CursorAgentClient()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=Exception("Network error"),
        ):
            models = await client.list_models()

        assert models == []

    def test_list_models_sync(self):
        """测试同步获取模型列表"""
        client = CursorAgentClient()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=b"model-1\nmodel-2\n",
            )

            models = client.list_models_sync()

        assert len(models) == 2


# ==================== list_sessions() 测试 ====================


class TestListSessions:
    """list_sessions() 测试"""

    @pytest.mark.asyncio
    async def test_list_sessions_success(self):
        """测试获取会话列表成功"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"session-1\nsession-2\n", b"")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            sessions = await client.list_sessions()

        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_failure(self):
        """测试获取会话列表失败"""
        client = CursorAgentClient()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=Exception("Error"),
        ):
            sessions = await client.list_sessions()

        assert sessions == []


# ==================== resume_session() 测试 ====================


class TestResumeSession:
    """resume_session() 测试"""

    @pytest.mark.asyncio
    async def test_resume_session_with_id(self):
        """测试恢复指定会话"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Resumed", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            result = await client.resume_session("session-123")

            call_args = mock_exec.call_args.args
            assert "--resume" in call_args
            assert "session-123" in call_args

        assert result.success is True

    @pytest.mark.asyncio
    async def test_resume_latest_session(self):
        """测试恢复最新会话"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Resumed", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            result = await client.resume_session()

            call_args = mock_exec.call_args.args
            assert "resume" in call_args

        assert result.success is True

    @pytest.mark.asyncio
    async def test_resume_session_failure(self):
        """测试恢复会话失败"""
        client = CursorAgentClient()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=Exception("Session not found"),
        ):
            result = await client.resume_session("invalid-session")

        assert result.success is False
        assert "Session not found" in result.error


# ==================== get_status() 测试 ====================


class TestGetStatus:
    """get_status() 测试"""

    @pytest.mark.asyncio
    async def test_get_status_authenticated(self):
        """测试已认证状态"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Logged in as user@example.com", b"")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            status = await client.get_status()

        assert status["authenticated"] is True
        assert "user@example.com" in status["output"]

    @pytest.mark.asyncio
    async def test_get_status_not_authenticated(self):
        """测试未认证状态"""
        client = CursorAgentClient()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Not logged in")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            status = await client.get_status()

        assert status["authenticated"] is False
        assert "Not logged in" in status["error"]


# ==================== 静态方法测试 ====================


class TestStaticMethods:
    """静态方法测试"""

    def test_install_instructions(self):
        """测试安装说明"""
        instructions = CursorAgentClient.install_instructions()

        assert "curl" in instructions
        assert "cursor.com/install" in instructions


# ==================== CursorAgentPool 测试 ====================


class TestCursorAgentPool:
    """CursorAgentPool 连接池测试"""

    @pytest.mark.asyncio
    async def test_pool_initialization(self):
        """测试连接池初始化"""
        pool = CursorAgentPool(size=3)

        await pool.initialize()

        assert pool._initialized is True
        assert len(pool._clients) == 3
        assert pool._available.qsize() == 3

    @pytest.mark.asyncio
    async def test_pool_acquire_release(self):
        """测试获取和释放客户端"""
        pool = CursorAgentPool(size=2)
        await pool.initialize()

        # 获取客户端
        client1 = await pool.acquire()
        assert pool._available.qsize() == 1

        client2 = await pool.acquire()
        assert pool._available.qsize() == 0

        # 释放客户端
        await pool.release(client1)
        assert pool._available.qsize() == 1

        await pool.release(client2)
        assert pool._available.qsize() == 2

    @pytest.mark.asyncio
    async def test_pool_acquire_timeout(self):
        """测试获取客户端超时"""
        pool = CursorAgentPool(size=1)
        await pool.initialize()

        # 获取唯一的客户端
        client = await pool.acquire()

        # 再次获取应该超时
        with pytest.raises(RuntimeError, match="超时"):
            await pool.acquire(timeout=0.1)

        # 释放
        await pool.release(client)

    @pytest.mark.asyncio
    async def test_pool_execute(self):
        """测试通过池执行任务"""
        pool = CursorAgentPool(size=2)

        # Mock execute 方法
        async def mock_execute(*args, **kwargs):
            return CursorAgentResult(success=True, output="Pool executed")

        with patch.object(
            CursorAgentClient,
            "execute",
            side_effect=mock_execute,
        ):
            result = await pool.execute("测试任务")

        assert result.success is True
        assert result.output == "Pool executed"

    @pytest.mark.asyncio
    async def test_pool_with_custom_config(self):
        """测试自定义配置的连接池"""
        config = CursorAgentConfig(model="gpt-5.2-high")
        pool = CursorAgentPool(size=2, config=config)
        await pool.initialize()

        client = await pool.acquire()
        assert client.config.model == "gpt-5.2-high"

        await pool.release(client)

    @pytest.mark.asyncio
    async def test_pool_auto_initialize(self):
        """测试自动初始化"""
        pool = CursorAgentPool(size=2)
        assert pool._initialized is False

        # acquire 会自动初始化
        client = await pool.acquire()
        assert pool._initialized is True

        await pool.release(client)


# ==================== Mock 执行测试 ====================


class TestMockExecution:
    """模拟执行测试"""

    @pytest.mark.asyncio
    async def test_mock_execution_fallback(self):
        """测试模拟执行回退"""
        client = CursorAgentClient()

        # 模拟 CLI 不可用
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("agent not found"),
        ):
            result = await client.execute("测试任务")

        # 应该使用模拟模式
        assert result.success is True
        assert "[Mock]" in result.output
        assert "模拟执行成功" in result.output

    @pytest.mark.asyncio
    async def test_mock_execution_direct(self):
        """测试直接调用模拟执行"""
        client = CursorAgentClient()

        result = await client._mock_execution(
            prompt="测试提示",
            working_directory="/tmp",
        )

        assert result["success"] is True
        assert "[Mock]" in result["output"]
        assert result["exit_code"] == 0


# ==================== 集成场景测试 ====================


# ==================== 流读取测试 ====================


class TestStreamLineReading:
    """流行读取测试

    验证 _read_stream_lines 方法能正确处理：
    1. 正常行读取
    2. 超长行 (>16MB) 分块读取
    3. LimitOverrunError 异常捕获和恢复
    """

    @pytest.mark.asyncio
    async def test_read_normal_lines(self):
        """测试正常行能被正确读取"""
        client = CursorAgentClient()

        # 创建 mock StreamReader
        mock_stream = AsyncMock()
        lines = [
            b'{"type": "system", "subtype": "init"}\n',
            b'{"type": "assistant", "content": "Hello"}\n',
            b'{"type": "result", "duration_ms": 100}\n',
            b'',  # EOF
        ]
        mock_stream.readline = AsyncMock(side_effect=lines)

        # 设置足够的截止时间
        deadline = asyncio.get_event_loop().time() + 60

        # 收集读取的行
        collected_lines = []
        async for line in client._read_stream_lines(mock_stream, deadline):
            collected_lines.append(line)

        # 验证结果
        assert len(collected_lines) == 3
        assert '{"type": "system", "subtype": "init"}' in collected_lines[0]
        assert '{"type": "assistant", "content": "Hello"}' in collected_lines[1]
        assert '{"type": "result", "duration_ms": 100}' in collected_lines[2]

    @pytest.mark.asyncio
    async def test_read_long_line_via_chunked_read(self):
        """测试超长行能通过分块读取正确处理"""
        client = CursorAgentClient()

        # 模拟超长内容（大于默认 limit）
        long_content = "x" * (1024 * 1024 * 2)  # 2MB 内容
        long_line = f'{{"type": "data", "content": "{long_content}"}}\n'

        # 创建 mock StreamReader，分块返回超长行
        mock_stream = AsyncMock()
        mock_stream._limit = 64 * 1024  # 默认 64KB limit

        # 模拟分块返回
        chunk_size = 1024 * 1024  # 1MB
        chunks = []
        remaining = long_line.encode()
        while remaining:
            chunk = remaining[:chunk_size]
            remaining = remaining[chunk_size:]
            chunks.append(chunk)
        chunks.append(b'')  # EOF

        call_count = [0]

        async def mock_read(size):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(chunks):
                return chunks[idx]
            return b''

        mock_stream.read = mock_read

        # 设置截止时间
        deadline = asyncio.get_event_loop().time() + 60

        # 测试 _read_long_line 方法
        result = await client._read_long_line(mock_stream, deadline)

        # 验证结果
        assert len(result) == len(long_line.encode())
        assert b'"type": "data"' in result

    @pytest.mark.asyncio
    async def test_limit_overrun_error_recovery(self):
        """测试 LimitOverrunError 异常能被正确捕获和恢复"""
        client = CursorAgentClient()

        # 创建 mock StreamReader
        mock_stream = AsyncMock()
        mock_stream._limit = 64 * 1024  # 64KB limit

        call_count = [0]
        normal_line = b'{"type": "result"}\n'
        long_content = b'x' * (1024 * 1024)  # 1MB 内容

        async def mock_readline():
            nonlocal call_count
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                # 第一次调用：抛出 LimitOverrunError
                raise asyncio.LimitOverrunError(
                    "Separator is found, but chunk is longer than limit",
                    consumed=1024 * 100,
                )
            elif idx == 1:
                # 恢复后返回正常行
                return normal_line
            else:
                return b''  # EOF

        async def mock_readuntil(sep):
            # 模拟读取超长行后的数据直到换行符
            return long_content + b'\n'

        mock_stream.readline = mock_readline
        mock_stream.readuntil = mock_readuntil

        # 设置截止时间
        deadline = asyncio.get_event_loop().time() + 60

        # 收集读取的行
        collected_lines = []
        async for line in client._read_stream_lines(mock_stream, deadline):
            collected_lines.append(line)

        # 验证：应该恢复并继续读取正常行
        assert len(collected_lines) >= 1
        assert '{"type": "result"}' in collected_lines[-1]

    @pytest.mark.asyncio
    async def test_limit_overrun_with_fallback_to_chunked_read(self):
        """测试 LimitOverrunError 后回退到分块读取"""
        client = CursorAgentClient()

        mock_stream = AsyncMock()
        mock_stream._limit = 64 * 1024

        call_count = [0]
        chunk_data = b'{"type": "long_data", "value": "' + b'y' * 1000 + b'"}\n'

        async def mock_readline():
            nonlocal call_count
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                raise asyncio.LimitOverrunError(
                    "Separator is found, but chunk is longer than limit",
                    consumed=50000,
                )
            elif idx == 1:
                return b'{"type": "normal"}\n'
            else:
                return b''

        async def mock_readuntil(sep):
            # 也抛出 LimitOverrunError，触发分块读取
            raise asyncio.LimitOverrunError(
                "chunk is longer than limit",
                consumed=50000,
            )

        read_call_count = [0]

        async def mock_read(size):
            nonlocal read_call_count
            idx = read_call_count[0]
            read_call_count[0] += 1
            if idx == 0:
                return chunk_data
            return b''

        mock_stream.readline = mock_readline
        mock_stream.readuntil = mock_readuntil
        mock_stream.read = mock_read

        deadline = asyncio.get_event_loop().time() + 60

        collected_lines = []
        async for line in client._read_stream_lines(mock_stream, deadline):
            collected_lines.append(line)

        # 验证：分块读取的内容应该被收集
        assert len(collected_lines) >= 1

    @pytest.mark.asyncio
    async def test_value_error_recovery(self):
        """测试 ValueError 异常能被正确捕获和恢复"""
        client = CursorAgentClient()

        mock_stream = AsyncMock()

        call_count = [0]
        normal_line = b'{"type": "final"}\n'
        chunk_data = b'{"type": "recovered"}\n'

        async def mock_readline():
            nonlocal call_count
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                # 抛出 ValueError（行超过 limit）
                raise ValueError("Line is too long")
            elif idx == 1:
                return normal_line
            else:
                return b''

        read_call_count = [0]

        async def mock_read(size):
            nonlocal read_call_count
            idx = read_call_count[0]
            read_call_count[0] += 1
            if idx == 0:
                return chunk_data
            return b''

        mock_stream.readline = mock_readline
        mock_stream.read = mock_read

        deadline = asyncio.get_event_loop().time() + 60

        collected_lines = []
        async for line in client._read_stream_lines(mock_stream, deadline):
            collected_lines.append(line)

        # 验证：应该恢复并继续读取
        assert len(collected_lines) >= 1

    @pytest.mark.asyncio
    async def test_incomplete_read_error_handling(self):
        """测试 IncompleteReadError 能被正确处理"""
        client = CursorAgentClient()

        mock_stream = AsyncMock()

        call_count = [0]
        partial_data = b'{"partial": true}'

        async def mock_readline():
            nonlocal call_count
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                raise asyncio.LimitOverrunError(
                    "Separator is found, but chunk is longer than limit",
                    consumed=1000,
                )
            elif idx == 1:
                return b'{"type": "next"}\n'
            else:
                return b''

        async def mock_readuntil(sep):
            # 抛出 IncompleteReadError，模拟流结束
            raise asyncio.IncompleteReadError(partial_data, expected=None)

        mock_stream.readline = mock_readline
        mock_stream.readuntil = mock_readuntil

        deadline = asyncio.get_event_loop().time() + 60

        collected_lines = []
        async for line in client._read_stream_lines(mock_stream, deadline):
            collected_lines.append(line)

        # 验证：应该继续处理
        assert len(collected_lines) >= 1

    @pytest.mark.asyncio
    async def test_read_long_line_with_newline_in_chunks(self):
        """测试分块读取时能正确检测换行符"""
        client = CursorAgentClient()

        mock_stream = AsyncMock()

        # 模拟多个分块，最后一个包含换行符
        chunks = [
            b'{"start": "',
            b'middle content here',
            b'", "end": true}\n',  # 包含换行符
            b'',  # EOF
        ]
        chunk_idx = [0]

        async def mock_read(size):
            idx = chunk_idx[0]
            chunk_idx[0] += 1
            if idx < len(chunks):
                return chunks[idx]
            return b''

        mock_stream.read = mock_read

        deadline = asyncio.get_event_loop().time() + 60

        result = await client._read_long_line(mock_stream, deadline)

        # 验证：应该在遇到换行符时停止
        expected = b''.join(chunks[:3])
        assert result == expected
        assert b'\n' in result

    @pytest.mark.asyncio
    async def test_read_long_line_max_size_limit(self):
        """测试分块读取时超过最大行大小限制"""
        client = CursorAgentClient()

        mock_stream = AsyncMock()

        # 模拟返回大量数据但没有换行符
        big_chunk = b'x' * (1024 * 1024)  # 1MB

        async def mock_read(size):
            return big_chunk  # 持续返回大块数据

        mock_stream.read = mock_read

        deadline = asyncio.get_event_loop().time() + 60

        # 这应该因为超过最大行大小而停止
        result = await client._read_long_line(mock_stream, deadline)

        # 验证：结果应该被截断
        # 最大行大小是 32MB
        assert len(result) <= 33 * 1024 * 1024  # 允许一些余量

    @pytest.mark.asyncio
    async def test_stream_timeout_handling(self):
        """测试流读取超时处理"""
        client = CursorAgentClient()

        mock_stream = AsyncMock()

        async def slow_readline():
            await asyncio.sleep(10)
            return b'too slow\n'

        mock_stream.readline = slow_readline

        # 设置很短的截止时间
        deadline = asyncio.get_event_loop().time() + 0.1

        collected_lines = []
        with pytest.raises(asyncio.TimeoutError):
            async for line in client._read_stream_lines(mock_stream, deadline):
                collected_lines.append(line)


# ==================== 集成场景测试 ====================


class TestIntegrationScenarios:
    """集成场景测试"""

    @pytest.mark.asyncio
    async def test_planner_workflow(self):
        """测试规划者工作流"""
        config = CursorAgentConfig(
            model="gpt-5.2-high",
            mode="plan",
            output_format="json",
            force_write=False,
        )
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(
                b'{"type": "result", "result": "Plan: 1. Analyze 2. Implement"}',
                b"",
            )
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            result = await client.execute("分析项目结构")

            # 验证规划者配置
            call_args = mock_exec.call_args.args
            assert "--mode" in call_args
            assert "plan" in call_args
            assert "--force" not in call_args

        assert result.success is True

    @pytest.mark.asyncio
    async def test_worker_workflow(self):
        """测试执行者工作流"""
        config = CursorAgentConfig(
            model="opus-4.5-thinking",
            mode="agent",
            output_format="text",
            force_write=True,
        )
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"File modified successfully", b"")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            result = await client.execute(
                "修改 src/main.py",
                context={"files": ["src/main.py"]},
            )

            # 验证执行者配置
            call_args = mock_exec.call_args.args
            assert "--mode" in call_args
            assert "agent" in call_args
            assert "--force" in call_args

        assert result.success is True

    @pytest.mark.asyncio
    async def test_worker_workflow_code_mode_compat(self):
        """测试执行者工作流 - code 模式兼容性

        注意: mode="code" 被映射为 "agent" 以保持向后兼容。
        实际命令为: --mode agent
        """
        config = CursorAgentConfig(
            model="opus-4.5-thinking",
            mode="code",  # 使用旧的 code 模式
            output_format="text",
            force_write=True,
        )
        client = CursorAgentClient(config=config)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"File modified successfully", b"")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            result = await client.execute(
                "修改 src/main.py",
                context={"files": ["src/main.py"]},
            )

            # 验证 code 模式被映射为 agent
            call_args = mock_exec.call_args.args
            assert "--mode" in call_args
            mode_idx = call_args.index("--mode")
            assert call_args[mode_idx + 1] == "agent"  # code 被映射为 agent
            assert "--force" in call_args

        assert result.success is True

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """测试并发执行"""
        pool = CursorAgentPool(size=3)

        async def mock_execute(*args, **kwargs):
            await asyncio.sleep(0.1)
            return CursorAgentResult(success=True, output="Done")

        with patch.object(
            CursorAgentClient,
            "execute",
            side_effect=mock_execute,
        ):
            # 并发执行 3 个任务
            tasks = [
                pool.execute("任务1"),
                pool.execute("任务2"),
                pool.execute("任务3"),
            ]
            results = await asyncio.gather(*tasks)

        assert all(r.success for r in results)
        assert len(results) == 3
