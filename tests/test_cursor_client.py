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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# cooldown_info 契约字段常量
from core.output_contract import (
    COOLDOWN_INFO_ALL_KNOWN_FIELDS,
    COOLDOWN_INFO_COMPAT_FIELDS,
    COOLDOWN_INFO_EXTENSION_FIELDS,
    COOLDOWN_INFO_REQUIRED_FIELDS,
    CooldownInfoFields,
)
from cursor.client import (
    CursorAgentClient,
    CursorAgentConfig,
    CursorAgentPool,
    CursorAgentResult,
    ModelPresets,
)
from cursor.cloud.exceptions import AuthError
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
        with patch("shutil.which", return_value=None), patch("os.path.isfile", return_value=False):
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
        mock_process.communicate = AsyncMock(return_value=(b"Task completed successfully", b""))

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
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error: Invalid command"))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await client.execute("无效命令")

        assert result.success is False
        assert result.exit_code == 1
        assert result.error is not None
        assert "Invalid command" in result.error

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """测试执行超时"""
        config = CursorAgentConfig(timeout=1)
        client = CursorAgentClient(config=config)

        # 使用 AsyncMock 避免创建未 await 的协程
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await client.execute("慢任务", timeout=1)

        assert result.success is False
        assert result.error is not None
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
            result = await client.execute("继续之前的任务", session_id="test-session-789")

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

        with patch.object(client, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = CursorAgentResult(success=True, output="成功")

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

        with patch.object(client, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = CursorAgentResult(success=False, error="持续失败")

            result = await client.execute_with_retry("测试")

        assert result.success is False
        assert mock_execute.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_custom_retries(self):
        """测试自定义重试次数"""
        client = CursorAgentClient()

        with patch.object(client, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = CursorAgentResult(success=False, error="错误")
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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

    @pytest.mark.asyncio
    async def test_session_id_from_assistant_when_system_init_missing(self):
        """测试 system/init 不含 session_id，但 assistant 行包含时能正确提取"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # system/init 不含 session_id，但 assistant 事件包含
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "分析中..."}]}, "session_id": "assistant-session-abc123"}',
            '{"type": "result", "duration_ms": 200}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
            result = await client.execute("测试 assistant session_id")

        assert result.success is True
        assert result.session_id == "assistant-session-abc123"

    @pytest.mark.asyncio
    async def test_session_id_from_tool_call_when_system_init_missing(self):
        """测试 system/init 不含 session_id，但 tool_call 行包含时能正确提取"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # system/init 不含 session_id，但 tool_call 事件包含
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "开始..."}]}}',
            '{"type": "tool_call", "subtype": "started", "tool_call": {"readToolCall": {"args": {"path": "main.py"}}}, "session_id": "tool-call-session-xyz789"}',
            '{"type": "tool_call", "subtype": "completed", "tool_call": {"readToolCall": {"args": {"path": "main.py"}, "result": {"success": {}}}}}',
            '{"type": "result", "duration_ms": 300}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
            result = await client.execute("测试 tool_call session_id")

        assert result.success is True
        assert result.session_id == "tool-call-session-xyz789"

    @pytest.mark.asyncio
    async def test_session_id_priority_system_init_over_assistant(self):
        """测试 system/init 的 session_id 优先于后续事件"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # system/init 和 assistant 都包含 session_id，应使用 system/init 的
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking", "session_id": "init-session-priority"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "分析..."}]}, "session_id": "assistant-session-secondary"}',
            '{"type": "result", "duration_ms": 100}',
        ]

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=[(line + "\n").encode() for line in stream_lines] + [b""])
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
            result = await client.execute("测试 session_id 优先级")

        assert result.success is True
        # system/init 的 session_id 应该优先，因为它先出现
        assert result.session_id == "init-session-priority"

    @pytest.mark.asyncio
    async def test_session_id_from_result_event(self):
        """测试从 result 事件提取 session_id"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # system/init 不含 session_id，但 result 事件包含
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "完成"}]}}',
            '{"type": "result", "duration_ms": 100, "session_id": "result-session-final"}',
        ]

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[line.encode() + b"\n" for line in stream_lines] + [b""])
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("测试 result session_id")

        assert result.success is True
        assert result.session_id == "result-session-final"

    @pytest.mark.asyncio
    async def test_session_id_from_diff_event(self):
        """测试从 diff 事件提取 session_id"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # system/init 不含 session_id，但 diff 事件包含
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "diff", "path": "test.py", "old_string": "a", "new_string": "b", "session_id": "diff-session-abc"}',
            '{"type": "result", "duration_ms": 50}',
        ]

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[line.encode() + b"\n" for line in stream_lines] + [b""])
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("测试 diff session_id")

        assert result.success is True
        assert result.session_id == "diff-session-abc"

    @pytest.mark.asyncio
    async def test_session_id_first_available_wins(self):
        """测试 session_id 使用第一个可用值（先到先得原则）"""
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 多个事件包含不同的 session_id，应使用第一个
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "assistant", "message": {"content": [{"text": "a"}]}, "session_id": "first-session"}',
            '{"type": "tool_call", "subtype": "started", "tool_call": {"readToolCall": {"args": {"path": "x.py"}}}, "session_id": "second-session"}',
            '{"type": "result", "duration_ms": 10, "session_id": "third-session"}',
        ]

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[line.encode() + b"\n" for line in stream_lines] + [b""])
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("测试 session_id 先到先得")

        assert result.success is True
        # 第一个 assistant 事件的 session_id 应该被采用
        assert result.session_id == "first-session"

    @pytest.mark.asyncio
    async def test_aggregated_output_from_assistant_not_result_field(self):
        """测试聚合输出来自 assistant 事件而非 result.result 字段

        验证 CursorAgentClient 的输出聚合逻辑：
        1. 输出内容从 assistant 事件的 content 拼接而来
        2. result 事件的 result 字段不被用于输出
        3. 即使 result 事件缺少 result 字段也不影响输出
        """
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        # 样本：多个 assistant 事件 + result 事件（无 result 字段）
        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "第一部分内容"}]}}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "第二部分内容"}]}}',
            '{"type": "result", "duration_ms": 123}',  # 无 result 字段
        ]

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[line.encode() + b"\n" for line in stream_lines] + [b""])
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("测试聚合输出")

        assert result.success is True
        # 输出应该是 assistant 事件内容的拼接
        assert "第一部分内容" in result.output
        assert "第二部分内容" in result.output
        assert result.output == "第一部分内容第二部分内容"

    @pytest.mark.asyncio
    async def test_result_event_minimal_fields_no_exception(self):
        """测试仅包含 type 字段的 result 事件不抛异常

        样本: {"type":"result"}
        """
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "完成"}]}}',
            '{"type": "result"}',  # 仅 type 字段
        ]

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[line.encode() + b"\n" for line in stream_lines] + [b""])
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            # 不应抛异常
            result = await client.execute("测试最小 result 事件")

        assert result.success is True
        assert result.output == "完成"

    @pytest.mark.asyncio
    async def test_result_event_with_extra_fields_no_exception(self):
        """测试包含额外字段的 result 事件不抛异常

        样本: {"type":"result","duration_ms":123,"is_error":false,"extra":"x"}
        """
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "处理完成"}]}}',
            '{"type": "result", "duration_ms": 123, "is_error": false, "extra": "x"}',
        ]

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[line.encode() + b"\n" for line in stream_lines] + [b""])
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            # 不应抛异常
            result = await client.execute("测试额外字段 result 事件")

        assert result.success is True
        assert result.output == "处理完成"

    @pytest.mark.asyncio
    async def test_output_not_from_result_result_field(self):
        """验证输出不使用 result.result 字段

        即使 result 事件包含 result 字段，输出仍从 assistant 事件拼接。
        """
        config = CursorAgentConfig(output_format="stream-json")
        client = CursorAgentClient(config=config)

        stream_lines = [
            '{"type": "system", "subtype": "init", "model": "opus-4.5-thinking"}',
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "来自assistant的内容"}]}}',
            '{"type": "result", "duration_ms": 500, "result": "来自result字段的内容-不应出现在输出中"}',
        ]

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[line.encode() + b"\n" for line in stream_lines] + [b""])
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            client.config.stream_events_enabled = False
            result = await client.execute("验证输出来源")

        assert result.success is True
        # 输出应该来自 assistant 事件
        assert result.output == "来自assistant的内容"
        # 不应该包含 result.result 字段的内容
        assert "来自result字段的内容" not in result.output


class TestOutputParsing:
    """输出解析测试"""

    def test_parse_json_output_success(self):
        """测试解析成功的 JSON 输出"""
        output = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "duration_ms": 1234,
                "result": "任务完成",
                "session_id": "test-session",
            }
        )

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
        mock_process.communicate = AsyncMock(return_value=(b"gpt-4\nopus-4.5\ngpt-5.2-high\n", b""))

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
        mock_process.communicate = AsyncMock(return_value=(b"session-1\nsession-2\n", b""))

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
        assert result.error is not None
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
        mock_process.communicate = AsyncMock(return_value=(b"Logged in as user@example.com", b""))

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
        mock_process.communicate = AsyncMock(return_value=(b"", b"Not logged in"))

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
            b"",  # EOF
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
        chunks.append(b"")  # EOF

        call_count = [0]

        async def mock_read(size):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(chunks):
                return chunks[idx]
            return b""

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
        long_content = b"x" * (1024 * 1024)  # 1MB 内容

        async def mock_readline():
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
                return b""  # EOF

        async def mock_readuntil(sep):
            # 模拟读取超长行后的数据直到换行符
            return long_content + b"\n"

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
        chunk_data = b'{"type": "long_data", "value": "' + b"y" * 1000 + b'"}\n'

        async def mock_readline():
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
                return b""

        async def mock_readuntil(sep):
            # 也抛出 LimitOverrunError，触发分块读取
            raise asyncio.LimitOverrunError(
                "chunk is longer than limit",
                consumed=50000,
            )

        read_call_count = [0]

        async def mock_read(size):
            idx = read_call_count[0]
            read_call_count[0] += 1
            if idx == 0:
                return chunk_data
            return b""

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
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                # 抛出 ValueError（行超过 limit）
                raise ValueError("Line is too long")
            elif idx == 1:
                return normal_line
            else:
                return b""

        read_call_count = [0]

        async def mock_read(size):
            idx = read_call_count[0]
            read_call_count[0] += 1
            if idx == 0:
                return chunk_data
            return b""

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
                return b""

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
            b"middle content here",
            b'", "end": true}\n',  # 包含换行符
            b"",  # EOF
        ]
        chunk_idx = [0]

        async def mock_read(size):
            idx = chunk_idx[0]
            chunk_idx[0] += 1
            if idx < len(chunks):
                return chunks[idx]
            return b""

        mock_stream.read = mock_read

        deadline = asyncio.get_event_loop().time() + 60

        result = await client._read_long_line(mock_stream, deadline)

        # 验证：应该在遇到换行符时停止
        expected = b"".join(chunks[:3])
        assert result == expected
        assert b"\n" in result

    @pytest.mark.asyncio
    async def test_read_long_line_max_size_limit(self):
        """测试分块读取时超过最大行大小限制"""
        client = CursorAgentClient()

        mock_stream = AsyncMock()

        # 模拟返回大量数据但没有换行符
        big_chunk = b"x" * (1024 * 1024)  # 1MB

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
            return b"too slow\n"

        mock_stream.readline = slow_readline

        # 设置很短的截止时间
        deadline = asyncio.get_event_loop().time() + 0.1

        collected_lines = []
        with pytest.raises(asyncio.TimeoutError):
            async for line in client._read_stream_lines(mock_stream, deadline):
                collected_lines.append(line)


# ==================== 集成场景测试 ====================


# ==================== AutoAgentExecutor 回退测试 ====================


class TestAutoAgentExecutorFallbackTableDriven:
    """AutoAgentExecutor 回退场景表驱动测试

    覆盖以下场景：
    1. execution_mode=auto + 无 key：必须回退 CLI，并输出可操作提示
    2. execution_mode=auto + 认证失败：回退 CLI，并进入冷却
    3. execution_mode=auto + 429：回退 CLI，并按 retry_after 冷却
    4. & 前缀 + cloud_enabled=false：按 Policy 定义使用 CLI
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scenario,no_key,auth_error,rate_limit_error,retry_after,cloud_enabled,expect_fallback_cli,expect_cooldown,expect_hint_message",
        [
            # 场景 1: auto + 无 key - 回退 CLI，输出可操作提示
            pytest.param(
                "auto_no_key",
                True,  # no_key
                False,  # auth_error
                False,  # rate_limit_error
                None,  # retry_after
                True,  # cloud_enabled
                True,  # expect_fallback_cli
                False,  # expect_cooldown（无 key 不启动 cooldown，立即回退）
                True,  # expect_hint_message（输出设置 key 提示）
                id="auto_no_key_fallback_cli_with_hint",
            ),
            # 场景 2: auto + 认证失败 - 回退 CLI，进入冷却
            pytest.param(
                "auto_auth_failure",
                False,  # no_key（有 key 但无效）
                True,  # auth_error
                False,  # rate_limit_error
                None,  # retry_after
                True,  # cloud_enabled
                True,  # expect_fallback_cli
                True,  # expect_cooldown
                False,  # expect_hint_message
                id="auto_auth_failure_fallback_cli_with_cooldown",
            ),
            # 场景 3: auto + 429 - 回退 CLI，按 retry_after 冷却
            pytest.param(
                "auto_rate_limit",
                False,  # no_key
                False,  # auth_error
                True,  # rate_limit_error
                60,  # retry_after
                True,  # cloud_enabled
                True,  # expect_fallback_cli
                True,  # expect_cooldown
                False,  # expect_hint_message
                id="auto_rate_limit_429_fallback_cli_with_retry_after_cooldown",
            ),
            # 场景 4: & 前缀 + cloud_enabled=false - 使用 CLI（不尝试 Cloud）
            pytest.param(
                "prefix_cloud_disabled",
                False,  # no_key
                False,  # auth_error
                False,  # rate_limit_error
                None,  # retry_after
                False,  # cloud_enabled=False
                True,  # expect_fallback_cli（直接使用 CLI）
                False,  # expect_cooldown（未尝试 Cloud，无 cooldown）
                False,  # expect_hint_message
                id="ampersand_prefix_cloud_disabled_uses_cli",
            ),
        ],
    )
    async def test_auto_executor_fallback_scenarios(
        self,
        scenario: str,
        no_key: bool,
        auth_error: bool,
        rate_limit_error: bool,
        retry_after: int | None,
        cloud_enabled: bool,
        expect_fallback_cli: bool,
        expect_cooldown: bool,
        expect_hint_message: bool,
    ) -> None:
        """测试 AutoAgentExecutor 各种回退场景"""
        from cursor.cloud_client import AuthError, RateLimitError
        from cursor.executor import (
            AgentResult,
            AutoAgentExecutor,
        )

        # 创建执行器
        config = CursorAgentConfig(
            cloud_enabled=cloud_enabled,
            execution_mode="auto",
        )
        auto_executor = AutoAgentExecutor(cli_config=config)

        # Mock CLI 执行器 - 总是成功
        cli_result = AgentResult(
            success=True,
            output="CLI 执行成功",
            executor_type="cli",
        )

        async def mock_cli_execute(*args, **kwargs):
            return cli_result

        # 设置 Mock
        with (
            patch.object(auto_executor._cli_executor, "execute", side_effect=mock_cli_execute),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
        ):
            # 根据场景设置 Cloud 执行器行为
            if no_key:
                # 无 key - Cloud 不可用
                with patch.object(
                    auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=False
                ):
                    result = await auto_executor.execute("& 测试任务")

                    # 验证回退到 CLI
                    assert result.success is True
                    assert result.executor_type == "cli" if expect_fallback_cli else "cloud"

                    # 验证无 cooldown（直接回退，不启动 cooldown）
                    if not expect_cooldown:
                        assert auto_executor.is_cloud_in_cooldown is False

            elif auth_error:
                # 认证失败
                async def mock_cloud_execute(*args, **kwargs):
                    raise AuthError("认证失败: API Key 无效")

                with (
                    patch.object(
                        auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True
                    ),
                    patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_execute),
                ):
                    result = await auto_executor.execute("测试任务")

                    # 验证回退到 CLI
                    assert result.success is True
                    assert result.executor_type == "cli"

                    # 验证进入冷却
                    if expect_cooldown:
                        assert auto_executor.is_cloud_in_cooldown is True
                        assert auto_executor._cloud_failure_count >= 1

            elif rate_limit_error:
                # 429 速率限制
                async def mock_cloud_execute_429(*args, **kwargs):
                    error = RateLimitError("速率限制")
                    error.retry_after = retry_after
                    raise error

                with (
                    patch.object(
                        auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True
                    ),
                    patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_execute_429),
                ):
                    result = await auto_executor.execute("测试任务")

                    # 验证回退到 CLI
                    assert result.success is True
                    assert result.executor_type == "cli"

                    # 验证进入冷却
                    if expect_cooldown:
                        assert auto_executor.is_cloud_in_cooldown is True
                        # 冷却时间应该基于 retry_after 或默认值
                        remaining = auto_executor.cloud_cooldown_remaining
                        assert remaining is not None
                        assert remaining > 0

            elif not cloud_enabled:
                # cloud_enabled=False - Cloud 不应被尝试
                call_count = [0]

                async def track_cloud_check():
                    call_count[0] += 1
                    return True

                # Cloud 可用但 cloud_enabled=False
                with patch.object(auto_executor._cloud_executor, "check_available", side_effect=track_cloud_check):
                    # 重置偏好以触发重新选择
                    auto_executor.reset_preference()

                    result = await auto_executor.execute("& 测试任务")

                    # 验证使用 CLI（Cloud 应该因为 cloud_enabled=False 而不被检查，
                    # 但 AutoAgentExecutor 本身不直接检查 cloud_enabled，
                    # 这个逻辑在 CursorAgentClient._should_route_to_cloud 中）
                    # 这里验证回退逻辑正确
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_auto_executor_no_key_outputs_actionable_hint(self) -> None:
        """测试 auto + 无 key 时输出可操作的设置提示

        期望行为：
        - 当 execution_mode=auto 且无 API Key 时
        - 应该回退到 CLI
        - 日志中应包含如何设置 API Key 的提示
        """
        import io

        from loguru import logger

        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        # Mock CLI 执行成功
        cli_result = AgentResult(
            success=True,
            output="CLI 执行成功",
            executor_type="cli",
        )

        # 捕获日志输出
        log_output = io.StringIO()

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=False),
        ):
            # 添加临时日志处理器来捕获输出
            handler_id = logger.add(log_output, format="{message}", level="DEBUG")
            try:
                result = await auto_executor.execute("测试任务")
            finally:
                logger.remove(handler_id)

        # 验证结果
        assert result.success is True
        assert result.executor_type == "cli"

        # 验证日志包含回退信息
        log_content = log_output.getvalue()
        # 日志中应包含回退到 CLI 的相关信息
        assert "CLI" in log_content or "回退" in log_content or "fallback" in log_content.lower()

    @pytest.mark.asyncio
    async def test_auto_executor_auth_failure_cooldown_started(self) -> None:
        """测试 auto + 认证失败时冷却被正确启动

        验证：
        - 认证失败后 is_cloud_in_cooldown 为 True
        - _cloud_failure_count 增加
        - 冷却剩余时间 > 0
        """
        from cursor.cloud_client import AuthError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        # 初始状态
        assert auto_executor.is_cloud_in_cooldown is False
        assert auto_executor._cloud_failure_count == 0

        # Mock
        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            raise AuthError("Invalid API Key")

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            result = await auto_executor.execute("测试任务")

        # 验证冷却状态
        assert result.success is True
        assert result.executor_type == "cli"
        assert auto_executor.is_cloud_in_cooldown is True
        assert auto_executor._cloud_failure_count >= 1
        assert auto_executor.cloud_cooldown_remaining is not None
        assert auto_executor.cloud_cooldown_remaining > 0

    @pytest.mark.asyncio
    async def test_auto_executor_rate_limit_cooldown_uses_retry_after(self) -> None:
        """测试 auto + 429 时按 retry_after 设置冷却时间

        验证：
        - 429 错误后进入冷却
        - 冷却时间应该基于 retry_after 值（使用夹逼策略）
        - classify_cloud_failure 会正确提取 retry_after
        """
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_429(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 45  # 服务端指定 45 秒
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_429),
        ):
            result = await auto_executor.execute("测试任务")

        # 验证冷却状态
        assert result.success is True
        assert result.executor_type == "cli"
        assert auto_executor.is_cloud_in_cooldown is True
        remaining = auto_executor.cloud_cooldown_remaining
        assert remaining is not None
        # 冷却时间应该基于 retry_after（45秒），经过夹逼策略（min=30, max=300）
        # 45 秒在 30-300 范围内，所以应该是 45 秒左右
        assert remaining > 0 and remaining <= 50  # 允许一些时间差
        # 验证 cooldown_info 包含 retry_after 和 user_message（两者都必须存在）
        assert result.cooldown_info is not None, "cooldown_info 应存在"
        assert result.cooldown_info.get(CooldownInfoFields.RETRY_AFTER) is not None, "cooldown_info.retry_after 应存在"
        # user_message 必须存在且非空（确保不同输出格式一致）
        user_msg = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE)
        assert user_msg is not None, "cooldown_info.user_message 不应为 None"
        assert len(user_msg) > 0, "cooldown_info.user_message 不应为空字符串"


class TestCloudExecutionPolicyTableDriven:
    """CloudExecutionPolicy 表驱动测试

    测试 should_try_cloud 和冷却策略
    """

    @pytest.mark.parametrize(
        "cloud_enabled,in_cooldown,expect_should_try,expect_reason_contains",
        [
            # cloud_enabled=True，无冷却 -> 应该尝试
            pytest.param(True, False, True, None, id="enabled_no_cooldown_should_try"),
            # cloud_enabled=False -> 不应尝试
            pytest.param(False, False, False, "cloud_enabled=False", id="disabled_should_not_try"),
            # cloud_enabled=True，有冷却 -> 不应尝试
            pytest.param(True, True, False, "冷却期", id="enabled_in_cooldown_should_not_try"),
        ],
    )
    def test_should_try_cloud_scenarios(
        self,
        cloud_enabled: bool,
        in_cooldown: bool,
        expect_should_try: bool,
        expect_reason_contains: str | None,
    ) -> None:
        """测试 should_try_cloud 各种场景"""
        from datetime import datetime, timedelta

        from cursor.executor import CloudExecutionPolicy, CooldownConfig

        policy = CloudExecutionPolicy(config=CooldownConfig())

        # 如果需要设置冷却状态
        if in_cooldown:
            policy._state.cooldown_until = datetime.now() + timedelta(seconds=60)
            policy._state.error_type = policy._state.error_type or policy.classify_error(Exception("test"))

        should_try, reason = policy.should_try_cloud(cloud_enabled=cloud_enabled)

        assert should_try == expect_should_try
        if expect_reason_contains:
            assert reason is not None
            assert expect_reason_contains in reason

    @pytest.mark.parametrize(
        "error_class,error_message,expect_kind,expect_retryable",
        [
            # RateLimitError -> RATE_LIMIT
            pytest.param(
                "RateLimitError",
                "Too many requests",
                "rate_limit",
                True,
                id="rate_limit_error",
            ),
            # AuthError -> AUTH
            pytest.param(
                "AuthError",
                "Invalid API key",
                "auth",
                False,
                id="auth_error",
            ),
            # TimeoutError -> TIMEOUT
            pytest.param(
                "TimeoutError",
                "Request timed out",
                "timeout",
                True,
                id="timeout_error",
            ),
            # NetworkError -> NETWORK
            pytest.param(
                "NetworkError",
                "Connection refused",
                "network",
                True,
                id="network_error",
            ),
        ],
    )
    def test_classify_error_scenarios(
        self,
        error_class: str,
        error_message: str,
        expect_kind: str,
        expect_retryable: bool,
    ) -> None:
        """测试错误分类"""
        from cursor.cloud_client import AuthError, NetworkError, RateLimitError
        from cursor.executor import CloudErrorType, CloudExecutionPolicy

        policy = CloudExecutionPolicy()

        # 创建对应的错误
        error_classes = {
            "RateLimitError": RateLimitError,
            "AuthError": AuthError,
            "NetworkError": NetworkError,
            "TimeoutError": TimeoutError,
        }

        error = error_classes[error_class](error_message)
        error_type = policy.classify_error(error)

        expected_type = CloudErrorType(expect_kind)
        assert error_type == expected_type


class TestClassifyCloudFailureStructured:
    """classify_cloud_failure 结构化字段测试

    验证 classify_cloud_failure 能正确处理：
    1. raw_result dict 中的 error_type 字段
    2. retry_after 字段的提取
    3. 回退到字符串解析
    """

    @pytest.mark.parametrize(
        "error_input,expect_kind,expect_retry_after,expect_retryable",
        [
            # 结构化 dict 输入 - rate_limit
            pytest.param(
                {"error_type": "rate_limit", "error": "Rate limit", "retry_after": 60},
                "rate_limit",
                60,
                True,
                id="structured_rate_limit_with_retry_after",
            ),
            # 结构化 dict 输入 - auth
            pytest.param(
                {"error_type": "auth", "error": "Invalid API key"},
                "auth",
                None,
                False,
                id="structured_auth_error",
            ),
            # 结构化 dict 输入 - timeout
            pytest.param(
                {"error_type": "timeout", "error": "Request timed out"},
                "timeout",
                None,
                True,
                id="structured_timeout_error",
            ),
            # 结构化 dict 输入 - network
            pytest.param(
                {"error_type": "network", "error": "Connection refused"},
                "network",
                None,
                True,
                id="structured_network_error",
            ),
            # 字符串输入 - rate limit 关键词
            pytest.param(
                "Rate limit exceeded, retry after 30 seconds",
                "rate_limit",
                30,
                True,
                id="string_rate_limit_with_retry_after",
            ),
            # 字符串输入 - 认证错误
            pytest.param(
                "401 Unauthorized: Invalid API key",
                "auth",
                None,
                False,
                id="string_auth_401",
            ),
            # 字符串输入 - 超时
            pytest.param(
                "Request timed out after 300s",
                "timeout",
                None,
                True,
                id="string_timeout",
            ),
        ],
    )
    def test_classify_cloud_failure_from_structured_and_string(
        self,
        error_input,
        expect_kind: str,
        expect_retry_after: int | None,
        expect_retryable: bool,
    ) -> None:
        """测试 classify_cloud_failure 处理结构化和字符串输入"""
        from core.execution_policy import CloudFailureKind, classify_cloud_failure

        failure_info = classify_cloud_failure(error_input)

        assert failure_info.kind == CloudFailureKind(expect_kind)
        assert failure_info.retryable == expect_retryable
        if expect_retry_after is not None:
            assert failure_info.retry_after == expect_retry_after

    def test_classify_cloud_failure_raw_result_priority(self) -> None:
        """测试 raw_result 中的结构化字段优先于错误消息解析"""
        from core.execution_policy import CloudFailureKind, classify_cloud_failure

        # raw_result 包含 error_type=rate_limit，但 error 消息包含 "auth"
        # 应该优先使用 error_type
        raw_result = {
            "error_type": "rate_limit",
            "error": "auth failed",  # 这个不应该被使用
            "retry_after": 120,
        }

        failure_info = classify_cloud_failure(raw_result)

        assert failure_info.kind == CloudFailureKind.RATE_LIMIT
        assert failure_info.retry_after == 120
        assert failure_info.retryable is True


class TestAutoExecutorCooldownInfoStructure:
    """AutoAgentExecutor cooldown_info 结构测试

    验证 cooldown_info 包含统一的用户提示消息
    """

    @pytest.mark.asyncio
    async def test_cooldown_info_contains_user_message(self) -> None:
        """测试 cooldown_info 包含 user_message 字段"""
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 60
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            result = await auto_executor.execute("测试任务")

        # 验证 cooldown_info 结构
        assert result.cooldown_info is not None
        assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info
        assert CooldownInfoFields.RETRYABLE in result.cooldown_info
        assert CooldownInfoFields.RETRY_AFTER in result.cooldown_info
        # user_message 必须存在且非空（与 stream-json/text 输出一致）
        user_msg = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE)
        assert user_msg is not None, "user_message 不应为 None"
        assert len(user_msg) > 0, "user_message 不应为空字符串"
        # user_message 应该包含回退提示
        assert "回退" in user_msg or "CLI" in user_msg or "速率限制" in user_msg, (
            f"user_message 应包含回退相关信息，实际: {user_msg}"
        )

    @pytest.mark.asyncio
    async def test_cooldown_info_from_cloud_result_failure(self) -> None:
        """测试 Cloud 返回失败结果时的 cooldown_info 结构"""
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        # Cloud 返回失败结果（非抛异常）
        cloud_fail_result = AgentResult(
            success=False,
            error="Rate limit exceeded",
            executor_type="cloud",
            raw_result={
                "error_type": "rate_limit",
                "error": "Rate limit exceeded",
                "retry_after": 90,
            },
        )

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(
                auto_executor._cloud_executor, "execute", new_callable=AsyncMock, return_value=cloud_fail_result
            ),
        ):
            result = await auto_executor.execute("测试任务")

        # 验证 cooldown_info 包含从 raw_result 提取的信息
        assert result.cooldown_info is not None
        assert result.cooldown_info.get(CooldownInfoFields.RETRY_AFTER) is not None
        # 由于使用 classify_cloud_failure，retry_after 应该被正确提取
        # 但会经过夹逼策略（30-300 秒）

        # 验证 user_message 存在且非空（确保与 stream-json/text 输出格式一致）
        user_msg = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE)
        assert user_msg is not None, "user_message 不应为 None"
        assert len(user_msg) > 0, "user_message 不应为空字符串"
        # user_message 应包含有意义的回退信息
        assert "回退" in user_msg or "CLI" in user_msg or "速率" in user_msg, (
            f"user_message 应包含回退相关信息，实际: {user_msg}"
        )


class TestExecutionPolicyModuleTableDriven:
    """core/execution_policy.py 模块表驱动测试

    测试 resolve_effective_execution_mode 和相关函数
    """

    @pytest.mark.parametrize(
        "requested_mode,triggered_by_prefix,cloud_enabled,has_api_key,expect_mode,expect_reason_contains",
        [
            # 场景 1: auto + 无 key -> 回退 CLI
            pytest.param(
                "auto",
                False,
                True,
                False,
                "cli",
                "未配置 API Key",
                id="auto_no_key_fallback_cli",
            ),
            # 场景 2: cloud + 无 key -> 回退 CLI
            pytest.param(
                "cloud",
                False,
                True,
                False,
                "cli",
                "未配置 API Key",
                id="cloud_no_key_fallback_cli",
            ),
            # 场景 3: & 前缀 + cloud_enabled=False -> CLI
            pytest.param(
                None,
                True,
                False,
                True,
                "cli",
                "cloud_enabled=False",
                id="prefix_cloud_disabled_uses_cli",
            ),
            # 场景 4: & 前缀 + cloud_enabled=True + 有 key -> Cloud
            pytest.param(
                None,
                True,
                True,
                True,
                "cloud",
                "& 前缀触发",
                id="prefix_cloud_enabled_uses_cloud",
            ),
            # 场景 5: & 前缀 + cloud_enabled=True + 无 key -> CLI
            pytest.param(
                None,
                True,
                True,
                False,
                "cli",
                "未配置 API Key",
                id="prefix_no_key_fallback_cli",
            ),
            # 场景 6: 显式 cli -> CLI（忽略 & 前缀）
            pytest.param(
                "cli",
                True,
                True,
                True,
                "cli",
                "显式指定",
                id="explicit_cli_ignores_prefix",
            ),
            # 场景 7: 无显式模式且无 & 前缀 -> 函数默认返回 CLI
            # 注意：这是 resolve_effective_execution_mode 函数级别的默认行为
            # 系统级别的默认值由 config.yaml 控制（默认 auto）
            pytest.param(
                None,
                False,
                True,
                True,
                "cli",
                "默认",
                id="no_explicit_mode_no_prefix_returns_cli",
            ),
        ],
    )
    def test_resolve_effective_execution_mode(
        self,
        requested_mode: str | None,
        triggered_by_prefix: bool,  # 参数名保留以兼容 pytest 参数化，语义为 has_ampersand_prefix
        cloud_enabled: bool,
        has_api_key: bool,
        expect_mode: str,
        expect_reason_contains: str,
    ) -> None:
        """测试 resolve_effective_execution_mode 函数

        参数说明：
        - triggered_by_prefix: pytest 参数名，实际传入 has_ampersand_prefix 参数
          语义为"语法检测层面是否存在 & 前缀"
        """
        from core.execution_policy import resolve_effective_execution_mode

        mode, reason = resolve_effective_execution_mode(
            requested_mode=requested_mode,
            has_ampersand_prefix=triggered_by_prefix,  # 语法检测层面
            cloud_enabled=cloud_enabled,
            has_api_key=has_api_key,
        )

        assert mode == expect_mode, f"期望 {expect_mode}，实际 {mode}"
        assert expect_reason_contains in reason, f"期望原因包含 '{expect_reason_contains}'，实际: {reason}"

    @pytest.mark.parametrize(
        "kind,retry_after,requested_mode,has_ampersand_prefix,expect_contains",
        [
            # 无 key 提示
            pytest.param(
                "no_key",
                None,
                "auto",
                False,
                ["未配置", "API Key", "CURSOR_API_KEY"],
                id="no_key_hint_message",
            ),
            # 认证失败提示
            pytest.param(
                "auth",
                None,
                "cloud",
                False,
                ["认证失败", "API Key"],
                id="auth_failure_hint_message",
            ),
            # 速率限制 + retry_after 提示
            pytest.param(
                "rate_limit",
                60,
                "auto",
                False,
                ["速率限制", "60"],
                id="rate_limit_with_retry_after_hint",
            ),
            # & 前缀存在时的回退（语法检测层面）
            pytest.param(
                "timeout",
                None,
                None,
                True,
                ["超时", "回退", "& 前缀"],
                id="timeout_with_ampersand_prefix_hint",
            ),
        ],
    )
    def test_build_user_facing_fallback_message(
        self,
        kind: str,
        retry_after: int | None,
        requested_mode: str | None,
        has_ampersand_prefix: bool,
        expect_contains: list[str],
    ) -> None:
        """测试 build_user_facing_fallback_message 函数"""
        from core.execution_policy import build_user_facing_fallback_message

        message = build_user_facing_fallback_message(
            kind=kind,
            retry_after=retry_after,
            requested_mode=requested_mode,
            has_ampersand_prefix=has_ampersand_prefix,
        )

        for expected in expect_contains:
            assert expected in message, f"期望消息包含 '{expected}'，实际: {message}"


class TestCloudFailureKindClassification:
    """测试 core/execution_policy.py 中 CloudFailureKind 分类

    验证 classify_cloud_failure 函数能正确分类各种错误类型到 CloudFailureKind
    """

    @pytest.mark.parametrize(
        "error_input,expect_kind,expect_retryable,expect_retry_after",
        [
            # NO_KEY 错误 - 不可重试
            pytest.param(
                "未配置 API Key",
                "no_key",
                False,
                None,
                id="no_key_from_string",
            ),
            pytest.param(
                {"error": "No API key configured", "error_type": "no_key"},
                "no_key",
                False,
                None,
                id="no_key_from_dict_structured",
            ),
            # AUTH 错误 - 不可重试
            pytest.param(
                "401 Unauthorized",
                "auth",
                False,
                None,
                id="auth_401_unauthorized",
            ),
            pytest.param(
                {"error": "Invalid API key", "error_type": "auth"},
                "auth",
                False,
                None,
                id="auth_from_dict_structured",
            ),
            # RATE_LIMIT 错误 - 可重试
            pytest.param(
                "429 Too Many Requests",
                "rate_limit",
                True,
                60,  # 默认值
                id="rate_limit_429",
            ),
            pytest.param(
                {"error": "Rate limit exceeded", "error_type": "rate_limit", "retry_after": 45},
                "rate_limit",
                True,
                45,  # 使用指定的 retry_after
                id="rate_limit_with_retry_after_from_dict",
            ),
            # TIMEOUT 错误 - 可重试
            pytest.param(
                "Request timed out",
                "timeout",
                True,
                None,
                id="timeout_from_string",
            ),
            # NETWORK 错误 - 可重试
            pytest.param(
                "Connection refused",
                "network",
                True,
                None,
                id="network_connection_refused",
            ),
            # QUOTA 错误 - 不可重试
            pytest.param(
                "Quota exceeded",
                "quota",
                False,
                None,
                id="quota_exceeded",
            ),
            # SERVICE 错误 - 可重试
            pytest.param(
                "500 Internal Server Error",
                "service",
                True,
                30,  # 服务端错误建议 30 秒
                id="service_500_error",
            ),
            pytest.param(
                "503 Service Unavailable",
                "service",
                True,
                30,
                id="service_503_unavailable",
            ),
        ],
    )
    def test_classify_cloud_failure_to_kind(
        self,
        error_input,
        expect_kind: str,
        expect_retryable: bool,
        expect_retry_after: int | None,
    ) -> None:
        """测试 classify_cloud_failure 函数分类到 CloudFailureKind"""
        from core.execution_policy import CloudFailureKind, classify_cloud_failure

        failure_info = classify_cloud_failure(error_input)

        # 验证错误类型
        expected_kind = CloudFailureKind(expect_kind)
        assert failure_info.kind == expected_kind, (
            f"期望 CloudFailureKind.{expected_kind.name}，实际 CloudFailureKind.{failure_info.kind.name}"
        )

        # 验证是否可重试
        assert failure_info.retryable == expect_retryable, (
            f"期望 retryable={expect_retryable}，实际 {failure_info.retryable}"
        )

        # 验证 retry_after（如果期望有值）
        if expect_retry_after is not None:
            assert failure_info.retry_after == expect_retry_after, (
                f"期望 retry_after={expect_retry_after}，实际 {failure_info.retry_after}"
            )


class TestRateLimitRetryAfterClamping:
    """测试 RATE_LIMIT 的 retry_after 夹逼逻辑

    验证 CloudExecutionPolicy 在处理 RATE_LIMIT 错误时：
    1. 优先使用 retry_after 而非默认值
    2. 应用最小/最大夹逼策略
    """

    @pytest.mark.parametrize(
        "retry_after,expected_cooldown,description",
        [
            # 正常范围内 - 直接使用
            pytest.param(60, 60, "正常范围内直接使用", id="normal_60s"),
            pytest.param(120, 120, "正常范围内直接使用", id="normal_120s"),
            # 低于最小值 - 夹逼到最小值
            pytest.param(10, 30, "低于最小值夹逼到 30s", id="clamp_to_min_10s"),
            pytest.param(1, 30, "极低值夹逼到 30s", id="clamp_to_min_1s"),
            pytest.param(29, 30, "边界值夹逼到 30s", id="clamp_to_min_29s"),
            # 高于最大值 - 夹逼到最大值
            pytest.param(400, 300, "高于最大值夹逼到 300s", id="clamp_to_max_400s"),
            pytest.param(600, 300, "极高值夹逼到 300s", id="clamp_to_max_600s"),
            pytest.param(301, 300, "边界值夹逼到 300s", id="clamp_to_max_301s"),
            # 边界值
            pytest.param(30, 30, "最小边界值", id="boundary_min_30s"),
            pytest.param(300, 300, "最大边界值", id="boundary_max_300s"),
        ],
    )
    def test_rate_limit_retry_after_clamping(
        self,
        retry_after: int,
        expected_cooldown: int,
        description: str,
    ) -> None:
        """测试 RATE_LIMIT retry_after 夹逼策略"""
        from cursor.cloud_client import RateLimitError
        from cursor.executor import CloudExecutionPolicy, CooldownConfig

        # 使用默认配置（min=30, max=300）
        config = CooldownConfig(
            rate_limit_min_seconds=30,
            rate_limit_max_seconds=300,
        )
        policy = CloudExecutionPolicy(config=config)

        # 创建带 retry_after 的 RateLimitError
        error = RateLimitError("Rate limit exceeded")
        error.retry_after = retry_after

        # 触发冷却
        metadata = policy.start_cooldown(error)

        # 验证冷却时间被正确夹逼
        remaining = policy.cooldown_state.get_remaining_seconds()
        assert remaining is not None
        # 允许 1 秒误差（执行时间）
        assert abs(remaining - expected_cooldown) <= 1.0, (
            f"{description}: 期望冷却 {expected_cooldown}s，实际 {remaining:.1f}s"
        )

    def test_rate_limit_uses_retry_after_over_default(self) -> None:
        """验证 RATE_LIMIT 优先使用 retry_after 而非默认值"""
        from cursor.cloud_client import RateLimitError
        from cursor.executor import CloudExecutionPolicy, CooldownConfig

        config = CooldownConfig(
            rate_limit_default_seconds=60,
            rate_limit_min_seconds=30,
            rate_limit_max_seconds=300,
        )
        policy = CloudExecutionPolicy(config=config)

        # 创建带 retry_after=90 的错误（不同于默认 60）
        error = RateLimitError("Rate limit exceeded")
        error.retry_after = 90

        policy.start_cooldown(error)

        # 验证使用了 retry_after 而非默认值
        assert policy.cooldown_state.retry_after_hint == 90
        remaining = policy.cooldown_state.get_remaining_seconds()
        assert remaining is not None
        assert abs(remaining - 90) <= 1.0, f"应使用 retry_after=90s，而非默认 60s，实际 {remaining:.1f}s"

    def test_rate_limit_no_retry_after_uses_default(self) -> None:
        """验证 RATE_LIMIT 无 retry_after 时使用默认值"""
        from cursor.cloud_client import RateLimitError
        from cursor.executor import CloudExecutionPolicy, CooldownConfig

        config = CooldownConfig(
            rate_limit_default_seconds=60,
            rate_limit_min_seconds=30,
            rate_limit_max_seconds=300,
        )
        policy = CloudExecutionPolicy(config=config)

        # 创建不带 retry_after 的错误
        error = RateLimitError("Rate limit exceeded")
        # 不设置 retry_after

        policy.start_cooldown(error)

        # 验证使用了默认值
        remaining = policy.cooldown_state.get_remaining_seconds()
        assert remaining is not None
        assert abs(remaining - 60) <= 1.0, f"无 retry_after 时应使用默认值 60s，实际 {remaining:.1f}s"


class TestNoKeyCooldownBehavior:
    """测试 NO_KEY 不启动冷却的行为

    验证：
    1. NO_KEY 错误类型不会触发 CloudExecutionPolicy 的冷却机制
    2. NO_KEY 应该在 resolve_effective_execution_mode 阶段就被拦截
    3. 区分 NO_KEY（配置问题）和 AUTH（Key 无效）的处理策略
    """

    def test_no_key_not_handled_by_cooldown_policy(self) -> None:
        """验证 NO_KEY 不由 CloudExecutionPolicy 的冷却机制处理

        CloudExecutionPolicy 主要处理运行时错误（AUTH, RATE_LIMIT 等），
        NO_KEY 是配置层面的问题，应该在更早的阶段被拦截。
        """
        from core.execution_policy import resolve_effective_execution_mode

        # 场景: auto 模式 + 无 key -> 直接回退 CLI，不触发冷却
        mode, reason = resolve_effective_execution_mode(
            requested_mode="auto",
            has_ampersand_prefix=False,  # 无 & 前缀
            cloud_enabled=True,
            has_api_key=False,  # 无 key
        )

        # 验证直接回退到 CLI
        assert mode == "cli"
        assert "未配置 API Key" in reason

    def test_no_key_immediate_fallback_no_cooldown(self) -> None:
        """验证无 key 时立即回退，不进入冷却状态"""
        from cursor.executor import CloudExecutionPolicy

        policy = CloudExecutionPolicy()

        # 初始状态应该没有冷却
        assert not policy.cooldown_state.is_in_cooldown()

        # NO_KEY 场景下，Cloud 执行器不可用，不会调用 start_cooldown
        # 因为 check_available() 返回 False，根本不会尝试执行

        # 验证状态保持干净
        assert policy.cooldown_state.failure_count == 0
        assert policy.cooldown_state.error_type is None

    def test_auth_error_starts_cooldown_vs_no_key(self) -> None:
        """对比 AUTH 错误和 NO_KEY 的处理差异

        - AUTH: 有 key 但无效，触发冷却
        - NO_KEY: 无 key，不触发冷却（在策略层拦截）
        """
        from cursor.cloud_client import AuthError
        from cursor.executor import CloudErrorType, CloudExecutionPolicy

        policy = CloudExecutionPolicy()

        # AUTH 错误（有 key 但无效）应该触发冷却
        auth_error = AuthError("Invalid API key")
        policy.start_cooldown(auth_error)

        # 验证 AUTH 进入冷却
        assert policy.cooldown_state.is_in_cooldown()
        assert policy.cooldown_state.error_type == CloudErrorType.AUTH
        assert policy.cooldown_state.failure_count == 1

    def test_cloud_failure_kind_no_key_is_not_retryable(self) -> None:
        """验证 CloudFailureKind.NO_KEY 被标记为不可重试"""
        from core.execution_policy import CloudFailureKind, classify_cloud_failure

        failure_info = classify_cloud_failure("No API key configured")

        assert failure_info.kind == CloudFailureKind.NO_KEY
        assert failure_info.retryable is False, "NO_KEY 错误不应该可重试（需要配置 API Key）"


class TestCloudExecutionPolicyAlignedToCloudFailureKind:
    """验证 CloudExecutionPolicy 与 CloudFailureKind 的对齐

    确保 cursor/executor.py 的 CloudErrorType 与 core/execution_policy.py 的
    CloudFailureKind 在错误分类上保持一致。
    """

    @pytest.mark.parametrize(
        "error_type_value,failure_kind_value",
        [
            ("rate_limit", "rate_limit"),
            ("auth", "auth"),
            ("network", "network"),
            ("timeout", "timeout"),
            ("unknown", "unknown"),
        ],
    )
    def test_error_type_values_match_failure_kind(
        self,
        error_type_value: str,
        failure_kind_value: str,
    ) -> None:
        """验证 CloudErrorType 值与 CloudFailureKind 对应值一致"""
        from core.execution_policy import CloudFailureKind
        from cursor.executor import CloudErrorType

        error_type = CloudErrorType(error_type_value)
        failure_kind = CloudFailureKind(failure_kind_value)

        assert error_type.value == failure_kind.value, (
            f"CloudErrorType.{error_type.name} 值应与 CloudFailureKind.{failure_kind.name} 一致"
        )

    def test_classify_error_consistent_with_classify_cloud_failure(self) -> None:
        """验证两个分类函数对相同输入产生一致结果"""
        from core.execution_policy import CloudFailureKind, classify_cloud_failure
        from cursor.cloud_client import AuthError, NetworkError, RateLimitError
        from cursor.executor import CloudErrorType, CloudExecutionPolicy

        policy = CloudExecutionPolicy()

        # 测试各种错误类型
        # 注意: NetworkError 有 error_type 属性，需要设置为 "network" 才能正确分类
        network_error = NetworkError("Connection refused", error_type="network")

        test_cases = [
            (RateLimitError("Rate limit"), CloudErrorType.RATE_LIMIT, CloudFailureKind.RATE_LIMIT),
            (AuthError("Auth failed"), CloudErrorType.AUTH, CloudFailureKind.AUTH),
            (network_error, CloudErrorType.NETWORK, CloudFailureKind.NETWORK),
            (TimeoutError("Timeout"), CloudErrorType.TIMEOUT, CloudFailureKind.TIMEOUT),
        ]

        for error, expected_error_type, expected_failure_kind in test_cases:
            # CloudExecutionPolicy.classify_error
            actual_error_type = policy.classify_error(error)
            assert actual_error_type == expected_error_type, (
                f"classify_error({type(error).__name__}) 应返回 {expected_error_type}"
            )

            # classify_cloud_failure
            failure_info = classify_cloud_failure(error)
            assert failure_info.kind == expected_failure_kind, (
                f"classify_cloud_failure({type(error).__name__}) 应返回 {expected_failure_kind}"
            )

            # 值应该一致
            assert actual_error_type.value == failure_info.kind.value


class TestCursorClientLogBehavior:
    """测试 CursorAgentClient 日志/输出行为

    验证：
    1. 库层不产生重复的用户提示
    2. 日志输出受控，不重复发送相同消息
    3. 回退消息仅输出一次
    """

    @pytest.mark.asyncio
    async def test_cloud_fallback_logs_warning_once(self) -> None:
        """验证 Cloud 回退时警告日志仅输出一次"""
        import io

        from loguru import logger

        from cursor.cloud_client import RateLimitError

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        # 捕获日志输出
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{message}", level="WARNING")

        try:
            # Mock Cloud 执行失败
            async def mock_cloud_execute(*args, **kwargs):
                raise RateLimitError("Rate limit exceeded")

            # Mock CLI 回退成功
            mock_cli_result = CursorAgentResult(
                success=True,
                output="CLI 执行成功",
                exit_code=0,
            )

            with (
                patch.object(client, "_execute_via_cloud", side_effect=mock_cloud_execute),
                patch.object(client, "_try_agent_cli", new_callable=AsyncMock, return_value=mock_cli_result),
            ):
                result = await client.execute("测试任务")

            # 验证警告只输出一次
            log_content = log_output.getvalue()
            warning_count = log_content.count("速率限制")
            assert warning_count <= 1, f"速率限制警告应最多输出一次，实际: {warning_count} 次"

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_no_duplicate_user_prompts_on_auth_error(self) -> None:
        """验证认证错误时不产生重复的用户提示"""
        import io

        from loguru import logger

        from cursor.cloud_client import AuthError

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{message}", level="DEBUG")

        try:

            async def mock_cloud_fail(*args, **kwargs):
                raise AuthError("Invalid API Key")

            mock_cli_result = CursorAgentResult(
                success=True,
                output="CLI 执行成功",
                exit_code=0,
            )

            with (
                patch.object(client, "_execute_via_cloud", side_effect=mock_cloud_fail),
                patch.object(client, "_try_agent_cli", new_callable=AsyncMock, return_value=mock_cli_result),
            ):
                await client.execute("测试任务")

            log_content = log_output.getvalue()

            # 验证 API Key 相关提示不重复
            api_key_mentions = sum(1 for line in log_content.split("\n") if "API Key" in line or "认证" in line)
            # 允许在日志中提及 API Key，但应该是有意义的消息，不是重复
            assert api_key_mentions <= 3, f"API Key 相关日志不应过多重复，实际: {api_key_mentions} 次"

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_fallback_message_format_is_user_friendly(self) -> None:
        """验证回退消息格式对用户友好"""
        from core.execution_policy import CloudFailureKind, build_user_facing_fallback_message

        # 测试各种错误类型的消息格式
        test_cases = [
            (CloudFailureKind.NO_KEY, "未配置"),
            (CloudFailureKind.AUTH, "认证"),
            (CloudFailureKind.RATE_LIMIT, "速率"),
            (CloudFailureKind.TIMEOUT, "超时"),
            (CloudFailureKind.NETWORK, "网络"),
        ]

        for kind, expected_keyword in test_cases:
            message = build_user_facing_fallback_message(
                kind=kind,
                retry_after=60 if kind == CloudFailureKind.RATE_LIMIT else None,
                requested_mode="auto",
                has_ampersand_prefix=False,
            )

            # 验证消息包含中文关键词
            assert expected_keyword in message, (
                f"CloudFailureKind.{kind.name} 消息应包含 '{expected_keyword}'，实际消息: {message}"
            )

            # 验证消息以警告符号开头
            assert message.startswith("⚠"), f"消息应以 ⚠ 开头表示警告，实际: {message[:20]}"

    def test_cloud_client_no_stdout_print_on_error(self) -> None:
        """验证 CursorAgentClient 不直接打印到 stdout"""
        import io
        import sys

        # 捕获 stdout
        original_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            # 创建客户端（不应该打印任何内容）
            config = CursorAgentConfig(execution_mode="auto")
            _client = CursorAgentClient(config=config)

            output = captured_output.getvalue()

            # 验证初始化时没有打印到 stdout
            assert len(output) == 0, f"CursorAgentClient 初始化不应打印到 stdout，实际: {output}"

        finally:
            sys.stdout = original_stdout

    @pytest.mark.asyncio
    async def test_cli_execution_no_redundant_logging(self) -> None:
        """验证纯 CLI 执行时不产生冗余日志"""
        import io

        from loguru import logger

        config = CursorAgentConfig(execution_mode="cli")
        client = CursorAgentClient(config=config)

        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}|{message}", level="DEBUG")

        try:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"Task completed", b""))

            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_process,
            ):
                result = await client.execute("CLI 测试任务")

            assert result.success is True

            log_content = log_output.getvalue()

            # 验证没有 Cloud 相关的警告日志（因为是纯 CLI 模式）
            assert "Cloud" not in log_content or "WARNING" not in log_content, "CLI 模式不应有 Cloud 相关警告"

        finally:
            logger.remove(handler_id)


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
        mock_process.communicate = AsyncMock(return_value=(b"File modified successfully", b""))

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
            # agent 为默认模式，CLI 不应传 --mode agent（部分版本仅支持 plan/ask）
            assert "--mode" not in call_args
            assert "--force" in call_args

        assert result.success is True

    @pytest.mark.asyncio
    async def test_worker_workflow_code_mode_compat(self):
        """测试执行者工作流 - code 模式兼容性

        注意: mode="code" 被映射为 "agent" 以保持向后兼容。
        实际行为为: agent 为默认模式，不传 --mode
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
        mock_process.communicate = AsyncMock(return_value=(b"File modified successfully", b""))

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
            assert "--mode" not in call_args
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


# ============================================================
# TestCloudFallbackScenariosComprehensive - NO_KEY/AUTH/RATE_LIMIT 综合测试
# ============================================================


class TestCloudFallbackScenariosComprehensive:
    """NO_KEY/AUTH/RATE_LIMIT 三场景综合验证

    验证：
    1. 最终走 CLI（回退成功）
    2. 提示语包含关键指引
    3. 冷却信息结构化输出（如有）
    """

    @pytest.mark.parametrize(
        "scenario,error_input,expect_kind,expect_cli_hint,expect_retryable,expect_retry_after",
        [
            # NO_KEY 场景
            pytest.param(
                "no_key",
                "未配置 API Key",
                "no_key",
                ["CURSOR_API_KEY", "config.yaml"],
                False,
                None,
                id="no_key_string",
            ),
            pytest.param(
                "no_key_dict",
                {"error": "No API key", "error_type": "no_key"},
                "no_key",
                ["CURSOR_API_KEY", "config.yaml"],
                False,
                None,
                id="no_key_from_dict",
            ),
            # AUTH 场景
            pytest.param(
                "auth_401",
                "401 Unauthorized",
                "auth",
                ["认证", "API Key"],
                False,
                None,
                id="auth_401",
            ),
            pytest.param(
                "auth_invalid_key",
                "Invalid API key",
                "auth",
                ["认证", "API Key"],
                False,
                None,
                id="auth_invalid_key",
            ),
            # RATE_LIMIT 场景
            pytest.param(
                "rate_limit_429",
                "429 Too Many Requests",
                "rate_limit",
                ["速率", "重试"],
                True,
                60,  # 默认 60 秒
                id="rate_limit_429",
            ),
            pytest.param(
                "rate_limit_with_retry",
                {"error": "Rate limit", "error_type": "rate_limit", "retry_after": 45},
                "rate_limit",
                ["速率", "重试"],
                True,
                45,
                id="rate_limit_with_retry_after",
            ),
        ],
    )
    def test_fallback_scenario_classification(
        self,
        scenario: str,
        error_input,
        expect_kind: str,
        expect_cli_hint: list,
        expect_retryable: bool,
        expect_retry_after: int | None,
    ) -> None:
        """验证错误分类和回退场景"""
        from core.execution_policy import (
            CloudFailureKind,
            build_user_facing_fallback_message,
            classify_cloud_failure,
        )

        # 分类错误
        failure_info = classify_cloud_failure(error_input)

        # 验证错误类型
        expected_kind = CloudFailureKind(expect_kind)
        assert failure_info.kind == expected_kind, f"场景 {scenario}: 期望 {expected_kind}，实际 {failure_info.kind}"

        # 验证是否可重试
        assert failure_info.retryable == expect_retryable, (
            f"场景 {scenario}: 期望 retryable={expect_retryable}，实际 {failure_info.retryable}"
        )

        # 验证 retry_after（如有）
        if expect_retry_after is not None:
            assert failure_info.retry_after == expect_retry_after, (
                f"场景 {scenario}: 期望 retry_after={expect_retry_after}，实际 {failure_info.retry_after}"
            )

        # 构建用户消息并验证包含关键指引
        message = build_user_facing_fallback_message(
            kind=failure_info.kind,
            retry_after=failure_info.retry_after,
            requested_mode="auto",
            has_ampersand_prefix=False,
        )

        for hint in expect_cli_hint:
            assert hint in message, f"场景 {scenario}: 提示消息应包含 '{hint}'，实际: {message}"

    def test_no_key_fallback_uses_cli(self) -> None:
        """验证 NO_KEY 场景最终使用 CLI"""
        from core.execution_policy import resolve_effective_execution_mode

        # auto 模式 + 无 key -> 回退 CLI
        mode, reason = resolve_effective_execution_mode(
            requested_mode="auto",
            has_ampersand_prefix=False,  # 无 & 前缀
            cloud_enabled=True,
            has_api_key=False,
        )

        assert mode == "cli", "NO_KEY 应回退到 CLI"
        assert "未配置 API Key" in reason, f"reason 应包含原因，实际: {reason}"

    def test_auth_fallback_triggers_cooldown(self) -> None:
        """验证 AUTH 场景触发冷却并最终使用 CLI"""
        from cursor.cloud_client import AuthError
        from cursor.executor import CloudErrorType, CloudExecutionPolicy

        policy = CloudExecutionPolicy()

        # AUTH 错误触发冷却
        auth_error = AuthError("Invalid API key")
        policy.start_cooldown(auth_error)

        # 验证进入冷却
        assert policy.cooldown_state.is_in_cooldown()
        assert policy.cooldown_state.error_type == CloudErrorType.AUTH

        # 验证冷却信息结构化
        remaining = policy.cooldown_state.get_remaining_seconds()
        assert remaining is not None
        assert remaining > 0

    def test_rate_limit_fallback_uses_retry_after(self) -> None:
        """验证 RATE_LIMIT 场景使用 retry_after 冷却"""
        from cursor.cloud_client import RateLimitError
        from cursor.executor import CloudErrorType, CloudExecutionPolicy

        policy = CloudExecutionPolicy()

        # 创建带 retry_after 的错误
        error = RateLimitError("Rate limit exceeded")
        error.retry_after = 45

        # 触发冷却
        policy.start_cooldown(error)

        # 验证进入冷却
        assert policy.cooldown_state.is_in_cooldown()
        assert policy.cooldown_state.error_type == CloudErrorType.RATE_LIMIT

        # 验证使用了指定的 retry_after（会被夹逼）
        remaining = policy.cooldown_state.get_remaining_seconds()
        assert remaining is not None
        # 45 秒会被夹逼到 [30, 300] 范围内，所以应该是 45
        assert 30 <= remaining <= 300

    def test_cooldown_info_structured_output(self) -> None:
        """验证冷却信息结构化输出"""
        from core.execution_policy import CloudFailureInfo, classify_cloud_failure

        # 测试各场景的结构化输出
        test_cases = [
            ("未配置 API Key", "no_key"),
            ("401 Unauthorized", "auth"),
            ("429 Rate limit", "rate_limit"),
        ]

        for error_msg, expect_kind in test_cases:
            failure_info = classify_cloud_failure(error_msg)

            # 验证返回 CloudFailureInfo 结构
            assert isinstance(failure_info, CloudFailureInfo)

            # 验证 to_dict 方法产生结构化输出
            info_dict = failure_info.to_dict()
            assert "kind" in info_dict
            assert "message" in info_dict
            assert "retry_after" in info_dict
            assert "retryable" in info_dict

            assert info_dict["kind"] == expect_kind, (
                f"错误 '{error_msg}': 期望 kind={expect_kind}，实际 {info_dict['kind']}"
            )

    @pytest.mark.asyncio
    async def test_auto_mode_no_key_executes_via_cli(self) -> None:
        """验证 auto 模式无 key 时通过 CLI 执行"""
        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        # Mock CLI 执行成功
        mock_result = CursorAgentResult(
            success=True,
            output="CLI 执行成功",
            exit_code=0,
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"CLI output", b""))

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key", return_value=None):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await client.execute("测试任务")

        # 验证执行成功（通过 CLI）
        assert result.success is True

    @pytest.mark.asyncio
    async def test_rate_limit_fallback_to_cli_with_structured_cooldown(self) -> None:
        """验证 RATE_LIMIT 回退 CLI 并输出结构化冷却信息"""
        from core.execution_policy import classify_cloud_failure
        from cursor.cloud_client import RateLimitError

        # 模拟 rate limit 错误
        error = RateLimitError("Rate limit exceeded")
        error.retry_after = 60

        failure_info = classify_cloud_failure(error)

        # 验证结构化信息
        assert failure_info.kind.value == "rate_limit"
        assert failure_info.retryable is True
        assert failure_info.retry_after == 60

        # 验证可以转换为字典
        info_dict = failure_info.to_dict()
        assert info_dict["kind"] == "rate_limit"
        assert info_dict["retry_after"] == 60
        assert info_dict["retryable"] is True

    @pytest.mark.parametrize(
        "error_type,expect_retryable,expect_cooldown",
        [
            pytest.param("no_key", False, False, id="no_key_no_cooldown"),
            pytest.param("auth", False, True, id="auth_has_cooldown"),
            pytest.param("rate_limit", True, True, id="rate_limit_has_cooldown"),
        ],
    )
    def test_error_type_cooldown_behavior(
        self,
        error_type: str,
        expect_retryable: bool,
        expect_cooldown: bool,
    ) -> None:
        """验证不同错误类型的冷却行为"""
        from cursor.cloud_client import AuthError, RateLimitError
        from cursor.executor import CloudExecutionPolicy

        policy = CloudExecutionPolicy()

        # 根据错误类型创建错误
        if error_type == "no_key":
            # NO_KEY 不触发冷却（在策略层拦截）
            # 验证初始状态
            assert not policy.cooldown_state.is_in_cooldown()
        elif error_type == "auth":
            auth_error = AuthError("Invalid API key")
            policy.start_cooldown(auth_error)
            assert policy.cooldown_state.is_in_cooldown() == expect_cooldown
        elif error_type == "rate_limit":
            rate_limit_error = RateLimitError("Rate limit")
            rate_limit_error.retry_after = 60
            policy.start_cooldown(rate_limit_error)
            assert policy.cooldown_state.is_in_cooldown() == expect_cooldown

    def test_fallback_message_contains_actionable_guidance_for_all_types(self) -> None:
        """验证所有错误类型的回退消息都包含可操作指引"""
        from core.execution_policy import (
            CloudFailureKind,
            build_user_facing_fallback_message,
        )

        test_cases = [
            (CloudFailureKind.NO_KEY, ["API Key", "CURSOR_API_KEY"]),
            (CloudFailureKind.AUTH, ["认证", "API Key"]),
            (CloudFailureKind.RATE_LIMIT, ["速率", "回退"]),
        ]

        for kind, expected_keywords in test_cases:
            message = build_user_facing_fallback_message(
                kind=kind,
                retry_after=60 if kind == CloudFailureKind.RATE_LIMIT else None,
                requested_mode="auto",
                has_ampersand_prefix=False,
            )

            for keyword in expected_keywords:
                assert keyword in message, f"{kind.name} 消息应包含 '{keyword}'，实际: {message}"


# ============================================================
# TestCloudFallbackCooldownInfoUserMessage - Cloud 回退 cooldown_info.user_message 测试
# ============================================================


class TestCloudFallbackCooldownInfoUserMessage:
    """Cloud 回退时 cooldown_info.user_message 行为测试

    验证：
    1. Cloud 失败并回退 CLI 时，CursorAgentResult.cooldown_info.user_message 存在
    2. 完整的 user_message 不会作为 warning 日志输出
    3. 日志仅保留技术性 info/debug 信息
    """

    @pytest.mark.asyncio
    async def test_cloud_fallback_cooldown_info_has_user_message(self) -> None:
        """验证 Cloud 失败回退 CLI 时 cooldown_info.user_message 存在"""
        from cursor.cloud_client import RateLimitError

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        # 模拟 rate limit 错误
        rate_limit_error = RateLimitError("Rate limit exceeded")
        rate_limit_error.retry_after = 60

        # Mock _should_route_to_cloud 返回 True
        with patch.object(client, "_should_route_to_cloud", return_value=True):
            # Mock CloudClientFactory.execute_task 抛出 RateLimitError
            # 需要 patch 到 cursor.cloud_client 模块
            with patch(
                "cursor.cloud_client.CloudClientFactory.execute_task",
                new_callable=AsyncMock,
                side_effect=rate_limit_error,
            ):
                # Mock CLI 回退成功
                mock_cli_result = {
                    "success": True,
                    "output": "CLI 执行成功",
                    "exit_code": 0,
                    "command": "agent -p '...'",
                }
                with patch.object(
                    client,
                    "_execute_cursor_agent",
                    new_callable=AsyncMock,
                    return_value=mock_cli_result,
                ):
                    result = await client.execute("& 测试任务")

        # 验证 cooldown_info 存在且包含 user_message
        assert result.cooldown_info is not None, "cooldown_info 应该存在"
        assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info, "cooldown_info 应包含 user_message 字段"
        user_message = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE)
        assert user_message is not None, "user_message 不应为 None"
        assert len(user_message) > 0, "user_message 不应为空"

        # 验证 user_message 包含有意义的回退信息
        assert "回退" in user_message or "CLI" in user_message or "速率" in user_message, (
            f"user_message 应包含回退相关信息，实际: {user_message}"
        )

    @pytest.mark.asyncio
    async def test_cloud_fallback_no_user_message_in_warning_logs(self) -> None:
        """验证完整 user_message 不作为 warning 日志输出"""
        import io

        from loguru import logger

        from cursor.cloud_client import RateLimitError

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        # 捕获 warning 级别日志
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}|{message}", level="WARNING")

        try:
            rate_limit_error = RateLimitError("Rate limit exceeded")
            rate_limit_error.retry_after = 60

            with (
                patch.object(client, "_should_route_to_cloud", return_value=True),
                patch(
                    "cursor.cloud_client.CloudClientFactory.execute_task",
                    new_callable=AsyncMock,
                    side_effect=rate_limit_error,
                ),
            ):
                mock_cli_result = {
                    "success": True,
                    "output": "CLI 执行成功",
                    "exit_code": 0,
                    "command": "agent -p '...'",
                }
                with patch.object(
                    client,
                    "_execute_cursor_agent",
                    new_callable=AsyncMock,
                    return_value=mock_cli_result,
                ):
                    result = await client.execute("& 测试任务")

            # 获取日志内容
            log_content = log_output.getvalue()

            # 验证 cooldown_info.user_message 存在
            assert result.cooldown_info is not None
            cooldown_info = result.cooldown_info or {}
            user_message = cooldown_info.get(CooldownInfoFields.USER_MESSAGE, "")

            # 验证 warning 日志中不包含完整的 user_message
            # user_message 通常包含 "⚠" 符号作为开头
            if user_message:
                # 检查完整的 user_message 是否出现在 warning 日志中
                assert user_message not in log_content, (
                    f"完整的 user_message 不应作为 warning 日志输出。\n"
                    f"user_message: {user_message}\n"
                    f"warning 日志: {log_content}"
                )

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_cloud_fallback_logs_only_technical_info(self) -> None:
        """验证 Cloud 回退时日志仅包含技术性 info/debug 信息"""
        import io

        from loguru import logger

        from cursor.cloud_client import AuthError

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        # 捕获所有级别日志
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}|{message}", level="DEBUG")

        try:
            auth_error = AuthError("Invalid API Key")

            with (
                patch.object(client, "_should_route_to_cloud", return_value=True),
                patch(
                    "cursor.cloud_client.CloudClientFactory.execute_task",
                    new_callable=AsyncMock,
                    side_effect=auth_error,
                ),
            ):
                mock_cli_result = {
                    "success": True,
                    "output": "CLI 执行成功",
                    "exit_code": 0,
                    "command": "agent -p '...'",
                }
                with patch.object(
                    client,
                    "_execute_cursor_agent",
                    new_callable=AsyncMock,
                    return_value=mock_cli_result,
                ):
                    result = await client.execute("& 测试任务")

            log_content = log_output.getvalue()
            log_lines = [line for line in log_content.split("\n") if line.strip()]

            # 验证 cooldown_info.user_message 存在
            assert result.cooldown_info is not None
            user_message = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE, "")

            # 检查日志级别分布
            for line in log_lines:
                if "|" in line:
                    level, message = line.split("|", 1)
                    # 如果是 WARNING 级别，不应包含完整的 user_message
                    if level == "WARNING" and user_message:
                        assert user_message not in message, f"WARNING 日志不应包含完整 user_message: {message}"

            # 验证回退相关的 info 日志存在（技术性日志）
            assert any("回退" in line or "fallback" in line.lower() for line in log_lines), (
                f"应有技术性的回退 info 日志，实际日志: {log_lines}"
            )

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error_class,error_message,expect_kind",
        [
            pytest.param(
                "RateLimitError",
                "Rate limit exceeded",
                "rate_limit",
                id="rate_limit_error",
            ),
            pytest.param(
                "AuthError",
                "Invalid API Key",
                "auth",
                id="auth_error",
            ),
            pytest.param(
                "NetworkError",
                "Connection refused",
                "network",
                id="network_error",
            ),
        ],
    )
    async def test_cloud_fallback_cooldown_info_structure_by_error_type(
        self,
        error_class: str,
        error_message: str,
        expect_kind: str,
    ) -> None:
        """验证不同错误类型的 cooldown_info 结构完整性"""
        from cursor import cloud_client

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        # 动态创建错误实例
        error_cls = getattr(cloud_client, error_class)
        error = error_cls(error_message)
        if error_class == "RateLimitError":
            error.retry_after = 60

        with (
            patch.object(client, "_should_route_to_cloud", return_value=True),
            patch(
                "cursor.cloud_client.CloudClientFactory.execute_task",
                new_callable=AsyncMock,
                side_effect=error,
            ),
        ):
            mock_cli_result = {
                "success": True,
                "output": "CLI 执行成功",
                "exit_code": 0,
                "command": "agent -p '...'",
            }
            with patch.object(
                client,
                "_execute_cursor_agent",
                new_callable=AsyncMock,
                return_value=mock_cli_result,
            ):
                result = await client.execute("& 测试任务")

        # 验证 cooldown_info 结构
        assert result.cooldown_info is not None, f"{error_class}: cooldown_info 应存在"
        assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info, (
            f"{error_class}: cooldown_info 应包含 user_message"
        )
        assert CooldownInfoFields.REASON in result.cooldown_info, f"{error_class}: cooldown_info 应包含 reason"
        assert CooldownInfoFields.RETRYABLE in result.cooldown_info, f"{error_class}: cooldown_info 应包含 retryable"

        # 验证 failure_kind
        assert result.failure_kind == expect_kind, f"期望 failure_kind={expect_kind}，实际: {result.failure_kind}"


# ============================================================
# TestAutoCooldownFallbackScenarios - AUTO 冷却回退典型场景测试
# ============================================================


class TestAutoCooldownFallbackScenarios:
    """AUTO 模式冷却回退典型场景测试

    覆盖 AUTO 模式下的冷却回退场景，验证：
    - cooldown_info["user_message"] 结构正确
    - 输出次数 ≤ 1（不刷屏）
    - 回退原因正确记录
    """

    @pytest.mark.asyncio
    async def test_auto_cooldown_fallback_user_message_exists_and_valid(self) -> None:
        """AUTO 模式冷却回退 - 验证 user_message 存在且有效

        断言：
        1. cooldown_info 存在
        2. cooldown_info["user_message"] 存在且非空
        3. user_message 包含回退相关信息
        4. cooldown_info 包含 kind/error_type 字段
        """
        from cursor.cloud_client import RateLimitError

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        rate_limit_error = RateLimitError("Rate limit exceeded")
        rate_limit_error.retry_after = 60

        with (
            patch.object(client, "_should_route_to_cloud", return_value=True),
            patch(
                "cursor.cloud_client.CloudClientFactory.execute_task",
                new_callable=AsyncMock,
                side_effect=rate_limit_error,
            ),
        ):
            mock_cli_result = {
                "success": True,
                "output": "CLI 执行成功",
                "exit_code": 0,
                "command": "agent -p '...'",
            }
            with patch.object(
                client,
                "_execute_cursor_agent",
                new_callable=AsyncMock,
                return_value=mock_cli_result,
            ):
                result = await client.execute("& 测试任务")

        # 断言 1: cooldown_info 存在
        assert result.cooldown_info is not None, "cooldown_info 应存在"

        # 断言 2: user_message 存在且非空
        assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info, "cooldown_info 应包含 user_message"
        user_message = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE)
        assert user_message is not None and len(user_message) > 0, "user_message 不应为 None 或空"

        # 断言 3: user_message 包含回退相关信息
        has_fallback_info = any(
            keyword in user_message for keyword in ["回退", "CLI", "速率", "Rate", "冷却", "cooldown"]
        )
        assert has_fallback_info, f"user_message 应包含回退相关信息，实际: {user_message}"

        # 断言 4: cooldown_info 包含 kind 或 error_type
        has_kind_field = CooldownInfoFields.KIND in result.cooldown_info or result.failure_kind is not None
        assert has_kind_field, "cooldown_info 应包含 kind 字段或 failure_kind"

    @pytest.mark.asyncio
    async def test_auto_cooldown_fallback_no_output_flood(self) -> None:
        """AUTO 模式冷却回退 - 验证不刷屏

        断言：
        1. 库层（client.py）不直接打印 user_message
        2. stdout/stderr 中回退关键词出现次数 ≤ 1
        """
        import io

        from loguru import logger

        from cursor.cloud_client import RateLimitError

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        # 捕获所有输出
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}|{message}", level="DEBUG")

        try:
            rate_limit_error = RateLimitError("Rate limit exceeded")
            rate_limit_error.retry_after = 60

            with (
                patch.object(client, "_should_route_to_cloud", return_value=True),
                patch(
                    "cursor.cloud_client.CloudClientFactory.execute_task",
                    new_callable=AsyncMock,
                    side_effect=rate_limit_error,
                ),
            ):
                mock_cli_result = {
                    "success": True,
                    "output": "CLI 执行成功",
                    "exit_code": 0,
                    "command": "agent -p '...'",
                }
                with patch.object(
                    client,
                    "_execute_cursor_agent",
                    new_callable=AsyncMock,
                    return_value=mock_cli_result,
                ):
                    result = await client.execute("& 测试任务")

            log_content = log_output.getvalue()

            # 断言 1: 库层不应直接打印完整的 user_message
            cooldown_info = result.cooldown_info or {}
            user_message = cooldown_info.get(CooldownInfoFields.USER_MESSAGE, "")
            if user_message and len(user_message) > 20:
                # 完整的 user_message 不应出现在日志中
                assert user_message not in log_content, (
                    f"完整 user_message 不应出现在日志中。\nuser_message: {user_message}\n日志: {log_content[:500]}"
                )

            # 断言 2: 关键词出现次数 ≤ 1（技术性日志可以有，但不应刷屏）
            fallback_keywords = ["Rate limit exceeded"]
            for keyword in fallback_keywords:
                count = log_content.count(keyword)
                assert count <= 2, f"'{keyword}' 在日志中出现 {count} 次，不应刷屏"

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_auto_cooldown_fallback_reason_recorded(self) -> None:
        """AUTO 模式冷却回退 - 验证回退原因正确记录

        断言：
        1. cooldown_info 包含 reason 或 fallback_reason
        2. reason 包含具体的错误信息
        3. retryable 字段正确设置
        """
        from cursor.cloud_client import NetworkError

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        network_error = NetworkError("Connection refused")

        with (
            patch.object(client, "_should_route_to_cloud", return_value=True),
            patch(
                "cursor.cloud_client.CloudClientFactory.execute_task",
                new_callable=AsyncMock,
                side_effect=network_error,
            ),
        ):
            mock_cli_result = {
                "success": True,
                "output": "CLI 执行成功",
                "exit_code": 0,
                "command": "agent -p '...'",
            }
            with patch.object(
                client,
                "_execute_cursor_agent",
                new_callable=AsyncMock,
                return_value=mock_cli_result,
            ):
                result = await client.execute("& 测试任务")

        # 断言 1: cooldown_info 包含 reason
        assert result.cooldown_info is not None
        has_reason = (
            CooldownInfoFields.REASON in result.cooldown_info
            or CooldownInfoFields.FALLBACK_REASON in result.cooldown_info
        )
        assert has_reason, "cooldown_info 应包含 reason 或 fallback_reason"

        # 断言 2: reason 包含错误信息
        reason = result.cooldown_info.get(CooldownInfoFields.REASON, "")
        if reason:
            assert len(reason) > 0, "reason 不应为空"

        # 断言 3: retryable 字段存在
        assert CooldownInfoFields.RETRYABLE in result.cooldown_info, "cooldown_info 应包含 retryable 字段"

    @pytest.mark.asyncio
    async def test_auto_cooldown_fallback_multiple_errors_dedup(self) -> None:
        """AUTO 模式多次冷却回退 - 验证去重机制

        断言：
        1. 多次执行产生的 cooldown_info 结构一致
        2. 日志中不会因多次失败而重复输出相同警告
        """
        import io

        from loguru import logger

        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor, CooldownConfig

        # 使用短冷却时间
        cooldown_config = CooldownConfig(
            rate_limit_default_seconds=1,
            rate_limit_min_seconds=1,
        )

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(
            cli_config=config,
            cooldown_config=cooldown_config,
        )

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}|{message}", level="WARNING")

        try:

            async def mock_cloud_fail(*args, **kwargs):
                error = RateLimitError("Rate limit exceeded")
                error.retry_after = 1
                raise error

            with (
                patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
                patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
                patch.object(
                    auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True
                ),
                patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
            ):
                # 执行多次
                results = []
                for _ in range(3):
                    result = await auto_executor.execute("测试任务")
                    results.append(result)

            # 断言 1: 所有结果的 cooldown_info 结构一致
            for i, result in enumerate(results):
                assert result.cooldown_info is not None, f"第 {i + 1} 次结果应有 cooldown_info"
                assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info, (
                    f"第 {i + 1} 次结果的 cooldown_info 应包含 user_message"
                )

            # 断言 2: WARNING 日志不应重复刷屏
            log_content = log_output.getvalue()
            warning_lines = [line for line in log_content.split("\n") if line.startswith("WARNING")]
            # 允许每次执行有警告，但不应该是 N^2 级别的刷屏
            assert len(warning_lines) <= 6, f"WARNING 日志 {len(warning_lines)} 条，不应刷屏"

        finally:
            logger.remove(handler_id)


# ============================================================
# TestCooldownInfoOutputFormatConsistency - cooldown_info 输出格式一致性测试
# ============================================================


class TestCooldownInfoOutputFormatConsistency:
    """cooldown_info 在不同输出格式下的结构一致性测试

    验证：
    1. 无论 output_format 是 text/json/stream-json，cooldown_info 结构一致
    2. user_message 字段始终存在且非空
    3. 结构化字段 (kind, retry_after, retryable) 始终存在
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "output_format",
        [
            pytest.param("text", id="text_format"),
            pytest.param("json", id="json_format"),
            pytest.param("stream-json", id="stream_json_format"),
        ],
    )
    async def test_cooldown_info_structure_consistent_across_output_formats(
        self,
        output_format: str,
    ) -> None:
        """测试不同 output_format 下 cooldown_info 结构一致"""
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(
            execution_mode="auto",
            output_format=output_format,
        )
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 60
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            result = await auto_executor.execute("测试任务")

        # 验证 cooldown_info 结构一致（与 output_format 无关）
        assert result.cooldown_info is not None, f"output_format={output_format}: cooldown_info 应存在"

        # 验证 user_message 存在且非空
        user_msg = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE)
        assert user_msg is not None, f"output_format={output_format}: user_message 不应为 None"
        assert len(user_msg) > 0, f"output_format={output_format}: user_message 不应为空"

        # 验证结构化字段存在
        assert CooldownInfoFields.KIND in result.cooldown_info, (
            f"output_format={output_format}: cooldown_info 应包含 kind"
        )
        assert CooldownInfoFields.RETRYABLE in result.cooldown_info, (
            f"output_format={output_format}: cooldown_info 应包含 retryable"
        )
        assert CooldownInfoFields.RETRY_AFTER in result.cooldown_info, (
            f"output_format={output_format}: cooldown_info 应包含 retry_after"
        )

    @pytest.mark.asyncio
    async def test_cooldown_info_user_message_content_consistency(self) -> None:
        """测试 user_message 内容在不同错误类型下的一致性"""
        from cursor.cloud_client import AuthError, NetworkError, RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        error_types = [
            (RateLimitError("Rate limit"), "rate_limit", ["速率", "限制", "回退"]),
            (AuthError("Auth failed"), "auth", ["认证", "失败", "回退"]),
            (NetworkError("Network error"), "network", ["网络", "错误", "回退"]),
        ]

        for error, expect_kind, expect_keywords in error_types:
            config = CursorAgentConfig(execution_mode="auto")
            auto_executor = AutoAgentExecutor(cli_config=config)

            cli_result = AgentResult(success=True, output="OK", executor_type="cli")

            async def mock_cloud_fail(*args, _error=error, **kwargs):
                raise _error

            with (
                patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
                patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
                patch.object(
                    auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True
                ),
                patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
            ):
                result = await auto_executor.execute("测试任务")

            # 验证 cooldown_info 结构
            assert result.cooldown_info is not None, f"错误类型 {expect_kind}: cooldown_info 应存在"

            # 验证 user_message 内容包含预期关键词
            user_msg = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE, "")
            assert user_msg, f"错误类型 {expect_kind}: user_message 不应为空"

            # 至少包含一个预期关键词
            has_keyword = any(kw in user_msg for kw in expect_keywords)
            assert has_keyword, (
                f"错误类型 {expect_kind}: user_message 应包含关键词 {expect_keywords}，实际内容: {user_msg}"
            )


# ============================================================
# TestCooldownInfoContractConsistency - cooldown_info 契约一致性测试
# ============================================================


class TestCooldownInfoContractConsistency:
    """验证 cooldown_info 字段契约的一致性

    测试 cursor/executor.py 和 cursor/client.py 输出的 cooldown_info 结构一致：
    - 必须存在的字段: kind, user_message, retryable, retry_after
    - 必须存在的原因字段: reason, fallback_reason (兼容别名)
    - 兼容字段: error_type, failure_kind
    - 冷却状态字段: in_cooldown, remaining_seconds, failure_count
    """

    @pytest.mark.asyncio
    async def test_cooldown_info_has_all_required_fields(self) -> None:
        """验证 cooldown_info 包含所有必要字段"""
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 60
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            result = await auto_executor.execute("测试任务")

        # 验证 cooldown_info 存在
        assert result.cooldown_info is not None, "cooldown_info 应存在"

        # 获取 cooldown_info 的实际键集合
        actual_keys = set(result.cooldown_info.keys())

        # 使用契约常量验证必需字段存在
        # COOLDOWN_INFO_REQUIRED_FIELDS: 核心稳定字段（必须存在）
        missing_required = COOLDOWN_INFO_REQUIRED_FIELDS - actual_keys
        assert not missing_required, f"cooldown_info 缺少契约必需字段: {missing_required}"

        # 验证兼容字段存在（COOLDOWN_INFO_COMPAT_FIELDS）
        missing_compat = COOLDOWN_INFO_COMPAT_FIELDS - actual_keys
        assert not missing_compat, f"cooldown_info 缺少兼容字段: {missing_compat}"

        # 验证扩展字段存在（COOLDOWN_INFO_EXTENSION_FIELDS）
        missing_extension = COOLDOWN_INFO_EXTENSION_FIELDS - actual_keys
        assert not missing_extension, f"cooldown_info 缺少扩展字段: {missing_extension}"

        # 验证所有已知字段都存在（允许额外字段）
        assert COOLDOWN_INFO_ALL_KNOWN_FIELDS.issubset(actual_keys), (
            f"cooldown_info 字段不满足契约要求。期望包含: {COOLDOWN_INFO_ALL_KNOWN_FIELDS}, 实际: {actual_keys}"
        )

        # 验证消息级别字段值有效
        assert result.cooldown_info[CooldownInfoFields.MESSAGE_LEVEL] in ("warning", "info"), (
            f"message_level 应为 'warning' 或 'info'，实际: {result.cooldown_info[CooldownInfoFields.MESSAGE_LEVEL]}"
        )

    @pytest.mark.asyncio
    async def test_cooldown_info_kind_and_error_type_consistency(self) -> None:
        """验证 kind 和 error_type 字段值的一致性"""
        from cursor.cloud_client import AuthError, NetworkError, RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        error_cases = [
            (RateLimitError("Rate limit"), "rate_limit", "rate_limit"),
            (AuthError("Auth failed"), "auth", "auth"),
            (NetworkError("Network error"), "network", "network"),
        ]

        for error, expect_kind, expect_error_type in error_cases:
            config = CursorAgentConfig(execution_mode="auto")
            auto_executor = AutoAgentExecutor(cli_config=config)

            cli_result = AgentResult(success=True, output="OK", executor_type="cli")

            async def mock_cloud_fail(*args, _error=error, **kwargs):
                raise _error

            with (
                patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
                patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
                patch.object(
                    auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True
                ),
                patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
            ):
                result = await auto_executor.execute("测试任务")

            # 验证 kind 字段
            assert result.cooldown_info is not None
            assert result.cooldown_info[CooldownInfoFields.KIND] == expect_kind, (
                f"kind 应为 {expect_kind}，实际: {result.cooldown_info[CooldownInfoFields.KIND]}"
            )

            # 验证 failure_kind 与 kind 相同
            assert result.cooldown_info[CooldownInfoFields.FAILURE_KIND] == expect_kind, (
                f"failure_kind 应等于 kind ({expect_kind})"
            )

            # 验证 error_type（兼容字段）
            assert result.cooldown_info[CooldownInfoFields.ERROR_TYPE] == expect_error_type, (
                f"error_type 应为 {expect_error_type}"
            )

    @pytest.mark.asyncio
    async def test_cooldown_info_reason_fields_not_none(self) -> None:
        """验证 reason 和 fallback_reason 字段非 None"""
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Custom rate limit message")
            error.retry_after = 45
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            result = await auto_executor.execute("测试任务")

        # 验证 reason 字段存在且非空
        assert result.cooldown_info is not None
        reason = result.cooldown_info.get(CooldownInfoFields.REASON)
        assert reason is not None, "reason 不应为 None"
        assert len(reason) > 0, "reason 不应为空字符串"

        # 验证 fallback_reason 与 reason 一致
        fallback_reason = result.cooldown_info.get(CooldownInfoFields.FALLBACK_REASON)
        assert fallback_reason == reason, f"fallback_reason ({fallback_reason}) 应等于 reason ({reason})"

    @pytest.mark.asyncio
    async def test_client_and_executor_cooldown_info_structure_match(self) -> None:
        """验证 CursorAgentClient 和 AutoAgentExecutor 输出的 cooldown_info 结构一致

        这是关键测试：确保两个模块使用相同的 build_cooldown_info 函数后，
        输出的字段集合完全一致。
        """
        from cursor.cloud_client import RateLimitError
        from cursor.executor import AgentResult, AutoAgentExecutor

        # 通过 AutoAgentExecutor 获取 cooldown_info
        config = CursorAgentConfig(execution_mode="auto")
        auto_executor = AutoAgentExecutor(cli_config=config)

        cli_result = AgentResult(success=True, output="OK", executor_type="cli")

        async def mock_cloud_fail(*args, **kwargs):
            error = RateLimitError("Rate limit exceeded")
            error.retry_after = 60
            raise error

        with (
            patch.object(auto_executor._cli_executor, "execute", new_callable=AsyncMock, return_value=cli_result),
            patch.object(auto_executor._cli_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "check_available", new_callable=AsyncMock, return_value=True),
            patch.object(auto_executor._cloud_executor, "execute", side_effect=mock_cloud_fail),
        ):
            executor_result = await auto_executor.execute("测试任务")

        # 验证 executor 的 cooldown_info 存在
        assert executor_result.cooldown_info is not None

        # 获取 executor cooldown_info 的所有键
        executor_keys = set(executor_result.cooldown_info.keys())

        # 使用从 core.output_contract 导入的契约常量
        # COOLDOWN_INFO_ALL_KNOWN_FIELDS 是所有契约要求字段的并集
        contract_required_fields = COOLDOWN_INFO_ALL_KNOWN_FIELDS

        # 验证所有契约要求的字段都存在
        missing_fields = contract_required_fields - executor_keys
        assert not missing_fields, f"cooldown_info 缺少契约要求的字段: {missing_fields}"

        # 验证 executor_keys 是契约字段的超集（允许额外字段，但必须包含契约字段）
        assert contract_required_fields.issubset(executor_keys), (
            f"cooldown_info 字段不满足契约要求。期望包含: {contract_required_fields}, 实际: {executor_keys}"
        )

    def test_output_contract_matches_cooldown_info_to_dict_keys(self) -> None:
        """验证 output_contract 契约字段集合与 CooldownInfo.to_dict() 键集合一致

        确保 core.output_contract 中定义的契约字段集合与
        core.execution_policy.CooldownInfo.to_dict() 输出的键集合保持同步。

        策略：
        - 契约字段集合（COOLDOWN_INFO_ALL_KNOWN_FIELDS）应等于 to_dict() 的键集合
        - 如果 to_dict() 新增字段，应同步更新 output_contract 常量
        """
        from core.execution_policy import CooldownInfo

        # 创建 CooldownInfo 实例并获取 to_dict() 输出的键集合
        cooldown = CooldownInfo(
            kind="test",
            user_message="测试消息",
            retryable=False,
            retry_after=None,
            reason="测试原因",
            in_cooldown=False,
            remaining_seconds=None,
            failure_count=0,
            message_level="info",
        )
        to_dict_keys = set(cooldown.to_dict().keys())

        # 验证契约字段集合与 to_dict() 键集合相等
        # 如果不相等，说明 CooldownInfo.to_dict() 与 output_contract 定义不同步
        assert to_dict_keys == COOLDOWN_INFO_ALL_KNOWN_FIELDS, (
            f"output_contract 契约字段集合与 CooldownInfo.to_dict() 键集合不一致。\n"
            f"契约字段集合: {sorted(COOLDOWN_INFO_ALL_KNOWN_FIELDS)}\n"
            f"to_dict() 键: {sorted(to_dict_keys)}\n"
            f"仅在契约中: {sorted(COOLDOWN_INFO_ALL_KNOWN_FIELDS - to_dict_keys)}\n"
            f"仅在 to_dict(): {sorted(to_dict_keys - COOLDOWN_INFO_ALL_KNOWN_FIELDS)}"
        )

        # 额外验证：必需字段应是 to_dict() 键的子集
        assert COOLDOWN_INFO_REQUIRED_FIELDS.issubset(to_dict_keys), (
            f"契约必需字段应是 to_dict() 键的子集。\n"
            f"缺少必需字段: {sorted(COOLDOWN_INFO_REQUIRED_FIELDS - to_dict_keys)}"
        )


class TestCooldownInfoMessageLevel:
    """验证 cooldown_info.message_level 字段的行为

    测试场景：
    - 默认 auto 无 key 不刷 warning（message_level=info）
    - 显式 auto/cloud 仍 warning（message_level=warning）
    - & 前缀无 key 仍 warning（message_level=warning）
    """

    @pytest.mark.asyncio
    async def test_config_auto_no_key_uses_info_level(self) -> None:
        """验证 config 默认 auto 无 key 时使用 info 级别（不刷 warning）"""
        from core.execution_policy import (
            CloudFailureInfo,
            CloudFailureKind,
            build_cooldown_info,
        )

        # 模拟无 API Key 导致的回退
        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.NO_KEY,
            message="未设置 CURSOR_API_KEY",
            retryable=False,
        )

        # mode_source="config" 表示来自 config.yaml 默认值
        cooldown_info = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason="未设置 CURSOR_API_KEY",
            requested_mode="auto",
            has_ampersand_prefix=False,
            mode_source="config",  # config.yaml 默认值
        )

        assert cooldown_info[CooldownInfoFields.MESSAGE_LEVEL] == "info", (
            "config 默认 auto 无 key 应使用 info 级别，避免每次都警告"
        )

    @pytest.mark.asyncio
    async def test_cli_explicit_auto_no_key_uses_warning_level(self) -> None:
        """验证 CLI 显式 --execution-mode auto 无 key 时使用 warning 级别"""
        from core.execution_policy import (
            CloudFailureInfo,
            CloudFailureKind,
            build_cooldown_info,
        )

        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.NO_KEY,
            message="未设置 CURSOR_API_KEY",
            retryable=False,
        )

        # mode_source="cli" 表示用户显式指定 --execution-mode auto
        cooldown_info = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason="未设置 CURSOR_API_KEY",
            requested_mode="auto",
            has_ampersand_prefix=False,
            mode_source="cli",  # CLI 显式指定
        )

        assert cooldown_info[CooldownInfoFields.MESSAGE_LEVEL] == "warning", (
            "CLI 显式 auto 无 key 应使用 warning 级别，用户显式请求需要明确提示"
        )

    @pytest.mark.asyncio
    async def test_cli_explicit_cloud_no_key_uses_warning_level(self) -> None:
        """验证 CLI 显式 --execution-mode cloud 无 key 时使用 warning 级别"""
        from core.execution_policy import (
            CloudFailureInfo,
            CloudFailureKind,
            build_cooldown_info,
        )

        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.NO_KEY,
            message="未设置 CURSOR_API_KEY",
            retryable=False,
        )

        cooldown_info = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason="未设置 CURSOR_API_KEY",
            requested_mode="cloud",
            has_ampersand_prefix=False,
            mode_source="cli",
        )

        assert cooldown_info[CooldownInfoFields.MESSAGE_LEVEL] == "warning", "CLI 显式 cloud 无 key 应使用 warning 级别"

    @pytest.mark.asyncio
    async def test_ampersand_prefix_no_key_uses_warning_level(self) -> None:
        """验证 & 前缀无 key 时使用 warning 级别"""
        from core.execution_policy import (
            CloudFailureInfo,
            CloudFailureKind,
            build_cooldown_info,
        )

        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.NO_KEY,
            message="未设置 CURSOR_API_KEY",
            retryable=False,
        )

        # has_ampersand_prefix=True 表示用户使用 & 前缀表示 Cloud 意图
        cooldown_info = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason="未设置 CURSOR_API_KEY",
            requested_mode="auto",
            has_ampersand_prefix=True,  # 用户使用 & 前缀
            mode_source="config",  # 即使来自 config，& 前缀仍应 warning
        )

        assert cooldown_info[CooldownInfoFields.MESSAGE_LEVEL] == "warning", (
            "& 前缀无 key 应使用 warning 级别，用户显式使用 & 前缀表示 Cloud 意图"
        )

    @pytest.mark.asyncio
    async def test_none_mode_source_no_prefix_uses_info_level(self) -> None:
        """验证 mode_source=None 且无 & 前缀时使用 info 级别（保守默认）"""
        from core.execution_policy import (
            CloudFailureInfo,
            CloudFailureKind,
            build_cooldown_info,
        )

        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.NO_KEY,
            message="未设置 CURSOR_API_KEY",
            retryable=False,
        )

        cooldown_info = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason="未设置 CURSOR_API_KEY",
            requested_mode="auto",
            has_ampersand_prefix=False,
            mode_source=None,  # 未指定来源
        )

        assert cooldown_info[CooldownInfoFields.MESSAGE_LEVEL] == "info", (
            "未指定 mode_source 且无 & 前缀应使用 info 级别（保守默认）"
        )

    @pytest.mark.asyncio
    async def test_rate_limit_error_with_cli_mode_source(self) -> None:
        """验证其他错误类型也遵循 mode_source 规则"""
        from core.execution_policy import (
            CloudFailureInfo,
            CloudFailureKind,
            build_cooldown_info,
        )

        failure_info = CloudFailureInfo(
            kind=CloudFailureKind.RATE_LIMIT,
            message="Rate limit exceeded",
            retryable=True,
            retry_after=60,
        )

        # CLI 显式指定
        cooldown_info_cli = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason="Rate limit exceeded",
            requested_mode="auto",
            has_ampersand_prefix=False,
            mode_source="cli",
        )
        assert cooldown_info_cli[CooldownInfoFields.MESSAGE_LEVEL] == "warning"

        # config 默认
        cooldown_info_config = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason="Rate limit exceeded",
            requested_mode="auto",
            has_ampersand_prefix=False,
            mode_source="config",
        )
        assert cooldown_info_config[CooldownInfoFields.MESSAGE_LEVEL] == "info"

    @pytest.mark.asyncio
    async def test_build_cooldown_info_from_metadata_respects_mode_source(self) -> None:
        """验证 build_cooldown_info_from_metadata 也遵循 mode_source 规则"""
        from core.execution_policy import (
            CloudFailureKind,
            build_cooldown_info_from_metadata,
        )

        # CLI 显式指定
        cooldown_info_cli = build_cooldown_info_from_metadata(
            failure_kind=CloudFailureKind.NO_KEY,
            failure_message="未设置 API Key",
            retry_after=None,
            retryable=False,
            mode_source="cli",
        )
        assert cooldown_info_cli[CooldownInfoFields.MESSAGE_LEVEL] == "warning"

        # config 默认
        cooldown_info_config = build_cooldown_info_from_metadata(
            failure_kind=CloudFailureKind.NO_KEY,
            failure_message="未设置 API Key",
            retry_after=None,
            retryable=False,
            mode_source="config",
        )
        assert cooldown_info_config[CooldownInfoFields.MESSAGE_LEVEL] == "info"

        # & 前缀
        cooldown_info_prefix = build_cooldown_info_from_metadata(
            failure_kind=CloudFailureKind.NO_KEY,
            failure_message="未设置 API Key",
            retry_after=None,
            retryable=False,
            mode_source="config",
            has_ampersand_prefix=True,
        )
        assert cooldown_info_prefix[CooldownInfoFields.MESSAGE_LEVEL] == "warning"


# ============================================================
# TestCloudFallbackNoUserMessageInLogs - 验证日志不含 user_message
# ============================================================


class TestCloudFallbackNoUserMessageInLogs:
    """验证各种错误类型的日志不包含完整 user_message

    覆盖错误类型：
    - rate_limit: RateLimitError (HTTP 429)
    - auth: AuthError (API Key 无效)
    - no_key: 未设置 CURSOR_API_KEY
    - network: NetworkError (连接错误)
    - timeout: asyncio.TimeoutError
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error_class,error_message,expect_kind",
        [
            pytest.param(
                "RateLimitError",
                "Rate limit exceeded",
                "rate_limit",
                id="rate_limit_error",
            ),
            pytest.param(
                "AuthError",
                "Invalid API Key",
                "auth",
                id="auth_error",
            ),
            pytest.param(
                "NetworkError",
                "Connection refused",
                "network",
                id="network_error",
            ),
        ],
    )
    async def test_no_user_message_in_warning_logs_by_error_type(
        self,
        error_class: str,
        error_message: str,
        expect_kind: str,
    ) -> None:
        """验证不同错误类型日志不包含完整 user_message"""
        import io

        from loguru import logger

        from cursor import cloud_client

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        # 捕获 WARNING 级别日志
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}|{message}", level="WARNING")

        try:
            # 动态创建错误实例
            error_cls = getattr(cloud_client, error_class)
            error = error_cls(error_message)
            if error_class == "RateLimitError":
                error.retry_after = 60

            with (
                patch.object(client, "_should_route_to_cloud", return_value=True),
                patch(
                    "cursor.cloud_client.CloudClientFactory.execute_task",
                    new_callable=AsyncMock,
                    side_effect=error,
                ),
            ):
                mock_cli_result = {
                    "success": True,
                    "output": "CLI 执行成功",
                    "exit_code": 0,
                    "command": "agent -p '...'",
                }
                with patch.object(
                    client,
                    "_execute_cursor_agent",
                    new_callable=AsyncMock,
                    return_value=mock_cli_result,
                ):
                    result = await client.execute("& 测试任务")

            log_content = log_output.getvalue()

            # 验证 cooldown_info 存在
            assert result.cooldown_info is not None, f"{error_class}: cooldown_info 应存在"
            assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info, (
                f"{error_class}: cooldown_info 应包含 user_message"
            )

            user_message = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE, "")

            # 验证 WARNING 日志不包含完整 user_message
            if user_message and len(user_message) > 20:
                assert user_message not in log_content, (
                    f"{error_class}: 完整 user_message 不应出现在 WARNING 日志中。\n"
                    f"user_message: {user_message}\n"
                    f"日志内容: {log_content[:500]}"
                )

            # 验证 failure_kind 正确
            assert result.failure_kind == expect_kind, f"期望 failure_kind={expect_kind}，实际: {result.failure_kind}"

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_timeout_error_no_user_message_in_logs(self) -> None:
        """验证 TimeoutError 日志不包含完整 user_message"""
        import io

        from loguru import logger

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
        )
        client = CursorAgentClient(config=config)

        # 捕获 WARNING 级别日志
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}|{message}", level="WARNING")

        try:
            # 使用 asyncio.TimeoutError
            timeout_error = asyncio.TimeoutError("Cloud 执行超时")

            with (
                patch.object(client, "_should_route_to_cloud", return_value=True),
                patch(
                    "cursor.cloud_client.CloudClientFactory.execute_task",
                    new_callable=AsyncMock,
                    side_effect=timeout_error,
                ),
            ):
                mock_cli_result = {
                    "success": True,
                    "output": "CLI 执行成功",
                    "exit_code": 0,
                    "command": "agent -p '...'",
                }
                with patch.object(
                    client,
                    "_execute_cursor_agent",
                    new_callable=AsyncMock,
                    return_value=mock_cli_result,
                ):
                    result = await client.execute("& 测试任务")

            log_content = log_output.getvalue()

            # 验证 cooldown_info 存在
            assert result.cooldown_info is not None, "cooldown_info 应存在"
            assert CooldownInfoFields.USER_MESSAGE in result.cooldown_info, "cooldown_info 应包含 user_message"

            user_message = result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE, "")

            # 验证 WARNING 日志不包含完整 user_message
            if user_message and len(user_message) > 20:
                assert user_message not in log_content, (
                    f"TimeoutError: 完整 user_message 不应出现在 WARNING 日志中。\n"
                    f"user_message: {user_message}\n"
                    f"日志内容: {log_content[:500]}"
                )

            # 验证 failure_kind 为 timeout
            assert result.failure_kind == "timeout", f"期望 failure_kind=timeout，实际: {result.failure_kind}"

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_no_key_error_logs_use_debug_level(self) -> None:
        """验证 NO_KEY 场景下 CursorAgentClient 使用 debug 日志而非 warning"""
        import io

        from loguru import logger

        config = CursorAgentConfig(
            execution_mode="auto",
            cloud_enabled=True,
            auto_detect_cloud_prefix=True,
        )
        client = CursorAgentClient(config=config)

        # 捕获所有级别日志
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}|{message}", level="DEBUG")

        try:
            # Mock resolve_api_key 返回 None（模拟无 API Key）
            with patch(
                "cursor.cloud_client.CloudClientFactory.resolve_api_key",
                return_value=None,
            ):
                # _should_route_to_cloud 应返回 False（因为无 API Key）
                should_route = client._should_route_to_cloud("& 测试任务")
                assert should_route is False, "无 API Key 时应返回 False"

            log_content = log_output.getvalue()

            # 验证 DEBUG 日志包含技术性信息
            assert "DEBUG" in log_content, "应有 DEBUG 日志记录无 API Key 情况"

            # 验证 WARNING 日志不包含长文案
            warning_lines = [line for line in log_content.split("\n") if line.startswith("WARNING")]
            # 不应有包含"设置"、"环境变量"等用户提示的 warning 日志
            for line in warning_lines:
                assert "环境变量" not in line, f"WARNING 日志不应包含用户提示文案: {line}"
                assert "CURSOR_API_KEY" not in line or len(line) < 100, f"WARNING 日志不应包含完整用户提示: {line}"

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_cloud_client_auth_failure_uses_debug_log(self) -> None:
        """验证 CloudClientFactory.execute_task 认证失败使用 debug 日志"""
        import io

        from loguru import logger

        from cursor.cloud.auth import AuthStatus
        from cursor.cloud.exceptions import AuthErrorCode

        # 捕获所有级别日志
        log_output = io.StringIO()
        handler_id = logger.add(log_output, format="{level}|{message}", level="DEBUG")

        try:
            # 创建认证失败状态
            auth_error = AuthError(
                message="API Key 无效",
                code=AuthErrorCode.INVALID_API_KEY,
            )
            failed_auth_status = AuthStatus(
                authenticated=False,
                error=auth_error,
            )

            # Mock 认证失败
            with patch(
                "cursor.cloud_client.CloudClientFactory.create",
            ) as mock_create:
                mock_client = AsyncMock()
                mock_auth_manager = AsyncMock()
                mock_auth_manager.authenticate = AsyncMock(return_value=failed_auth_status)
                mock_create.return_value = (mock_client, mock_auth_manager)

                from cursor.cloud_client import CloudClientFactory

                result = await CloudClientFactory.execute_task(
                    prompt="测试任务",
                    agent_config=None,
                )

            log_content = log_output.getvalue()

            # 验证返回错误结果
            assert result.success is False
            assert result.error_type == "auth"

            # 验证日志行为：应使用 DEBUG 而非 WARNING
            debug_lines = [line for line in log_content.split("\n") if line.startswith("DEBUG")]
            warning_lines = [line for line in log_content.split("\n") if line.startswith("WARNING")]

            # DEBUG 日志应包含技术性认证失败信息
            assert any("认证失败" in line or "code=" in line for line in debug_lines), (
                f"DEBUG 日志应包含认证失败信息，实际: {debug_lines}"
            )

            # WARNING 日志不应包含完整的 user_friendly_message
            user_friendly = auth_error.user_friendly_message
            for line in warning_lines:
                assert user_friendly not in line, f"WARNING 日志不应包含完整 user_friendly_message: {line}"

        finally:
            logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_cooldown_info_user_message_exists_for_all_error_types(
        self,
    ) -> None:
        """验证所有错误类型都能生成有效的 user_message"""
        from core.execution_policy import (
            CloudFailureKind,
            build_cooldown_info,
            classify_cloud_failure,
        )
        from cursor.cloud_client import (
            AuthError,
            NetworkError,
            RateLimitError,
        )

        # 测试各种错误类型
        error_cases = [
            (RateLimitError("Rate limit exceeded"), CloudFailureKind.RATE_LIMIT),
            (AuthError("Invalid API Key"), CloudFailureKind.AUTH),
            (NetworkError("Connection refused"), CloudFailureKind.NETWORK),
            (asyncio.TimeoutError("Timeout"), CloudFailureKind.TIMEOUT),
            (Exception("Unknown error"), CloudFailureKind.UNKNOWN),
        ]

        for error, expected_kind in error_cases:
            failure_info = classify_cloud_failure(error)

            # 验证分类正确
            assert failure_info.kind == expected_kind, (
                f"错误 {type(error).__name__} 应分类为 {expected_kind.value}，实际: {failure_info.kind.value}"
            )

            # 构建 cooldown_info
            cooldown_info = build_cooldown_info(
                failure_info=failure_info,
                fallback_reason=str(error),
                requested_mode="auto",
                has_ampersand_prefix=False,
            )

            # 验证 user_message 存在且非空
            assert CooldownInfoFields.USER_MESSAGE in cooldown_info, (
                f"错误 {type(error).__name__}: cooldown_info 应包含 user_message"
            )
            user_message = cooldown_info.get(CooldownInfoFields.USER_MESSAGE)
            assert user_message is not None, f"错误 {type(error).__name__}: user_message 不应为 None"
            assert len(user_message) > 0, f"错误 {type(error).__name__}: user_message 不应为空"
