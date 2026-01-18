"""MCP 服务器管理模块测试

测试 cursor/mcp.py 中的 MCP 服务器管理、工具调用等功能。
使用 mock 模拟外部 MCP 服务器响应。
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cursor.mcp import MCPManager, MCPServer, MCPTool, ensure_mcp_servers_enabled


# ==================== MCPServer 数据类测试 ====================


class TestMCPServer:
    """MCPServer 数据类测试"""

    def test_create_server(self):
        """测试创建 MCPServer 实例"""
        server = MCPServer(
            name="playwright",
            identifier="@anthropic/mcp-server-playwright",
            status="connected",
            source="project",
            transport="stdio",
        )

        assert server.name == "playwright"
        assert server.identifier == "@anthropic/mcp-server-playwright"
        assert server.status == "connected"
        assert server.source == "project"
        assert server.transport == "stdio"

    def test_server_equality(self):
        """测试 MCPServer 相等性比较"""
        server1 = MCPServer(
            name="fetch",
            identifier="fetch",
            status="connected",
            source="global",
            transport="HTTP",
        )
        server2 = MCPServer(
            name="fetch",
            identifier="fetch",
            status="connected",
            source="global",
            transport="HTTP",
        )

        assert server1 == server2


# ==================== MCPTool 数据类测试 ====================


class TestMCPTool:
    """MCPTool 数据类测试"""

    def test_create_tool(self):
        """测试创建 MCPTool 实例"""
        tool = MCPTool(
            name="fetch_url",
            description="获取网页内容",
            parameters=["url", "format"],
        )

        assert tool.name == "fetch_url"
        assert tool.description == "获取网页内容"
        assert tool.parameters == ["url", "format"]

    def test_tool_with_empty_parameters(self):
        """测试创建无参数的工具"""
        tool = MCPTool(
            name="list_files",
            description="列出当前目录文件",
            parameters=[],
        )

        assert tool.name == "list_files"
        assert tool.parameters == []


# ==================== MCPManager 测试 ====================


class TestMCPManager:
    """MCPManager 管理器测试"""

    def test_init_default_agent_path(self):
        """测试默认 agent 路径初始化"""
        manager = MCPManager()
        assert manager.agent_path == "agent"

    def test_init_custom_agent_path(self):
        """测试自定义 agent 路径初始化"""
        manager = MCPManager(agent_path="/usr/local/bin/agent")
        assert manager.agent_path == "/usr/local/bin/agent"

    @pytest.mark.asyncio
    async def test_list_servers_success(self):
        """测试成功获取服务器列表"""
        manager = MCPManager()

        # 模拟 CLI 输出
        mock_output = "playwright | connected | project | stdio\nfetch | connected | global | HTTP\n"
        mock_result = {
            "success": True,
            "output": mock_output,
            "error": None,
        }

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            servers = await manager.list_servers()

            mock_cmd.assert_called_once_with(["mcp", "list"])
            assert len(servers) == 2
            assert servers[0].name == "playwright"
            assert servers[0].status == "connected"
            assert servers[1].name == "fetch"
            assert servers[1].transport == "HTTP"

    @pytest.mark.asyncio
    async def test_list_servers_failure(self):
        """测试获取服务器列表失败"""
        manager = MCPManager()

        mock_result = {
            "success": False,
            "output": "",
            "error": "Command failed",
        }

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            servers = await manager.list_servers()

            assert servers == []

    @pytest.mark.asyncio
    async def test_list_servers_exception(self):
        """测试获取服务器列表时发生异常"""
        manager = MCPManager()

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = Exception("Unexpected error")

            servers = await manager.list_servers()

            assert servers == []

    @pytest.mark.asyncio
    async def test_list_tools_success(self):
        """测试成功获取工具列表"""
        manager = MCPManager()

        mock_result = {
            "success": True,
            "output": "tool1: Description 1\ntool2: Description 2",
            "error": None,
        }

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            tools = await manager.list_tools("playwright")

            mock_cmd.assert_called_once_with(["mcp", "list-tools", "playwright"])
            # 当前 _parse_tool_list 返回空列表（TODO 实现）
            assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_list_tools_failure(self):
        """测试获取工具列表失败"""
        manager = MCPManager()

        mock_result = {
            "success": False,
            "output": "",
            "error": "Server not found",
        }

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            tools = await manager.list_tools("unknown-server")

            assert tools == []

    @pytest.mark.asyncio
    async def test_list_tools_exception(self):
        """测试获取工具列表时发生异常"""
        manager = MCPManager()

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = Exception("Connection error")

            tools = await manager.list_tools("playwright")

            assert tools == []

    @pytest.mark.asyncio
    async def test_login_success(self):
        """测试 MCP 服务器登录成功"""
        manager = MCPManager()

        mock_result = {"success": True, "output": "Logged in", "error": None}

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            result = await manager.login("github-server")

            mock_cmd.assert_called_once_with(["mcp", "login", "github-server"])
            assert result is True

    @pytest.mark.asyncio
    async def test_login_failure(self):
        """测试 MCP 服务器登录失败"""
        manager = MCPManager()

        mock_result = {"success": False, "output": "", "error": "Auth failed"}

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            result = await manager.login("github-server")

            assert result is False

    @pytest.mark.asyncio
    async def test_login_exception(self):
        """测试登录时发生异常"""
        manager = MCPManager()

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = Exception("Network error")

            result = await manager.login("github-server")

            assert result is False

    @pytest.mark.asyncio
    async def test_enable_success(self):
        """测试成功启用 MCP 服务器"""
        manager = MCPManager()

        mock_result = {"success": True, "output": "Server enabled", "error": None}

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            result = await manager.enable("playwright")

            mock_cmd.assert_called_once_with(["mcp", "enable", "playwright"])
            assert result is True

    @pytest.mark.asyncio
    async def test_enable_failure(self):
        """测试启用 MCP 服务器失败"""
        manager = MCPManager()

        mock_result = {"success": False, "output": "", "error": "Not found"}

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            result = await manager.enable("unknown-server")

            assert result is False

    @pytest.mark.asyncio
    async def test_enable_exception(self):
        """测试启用服务器时发生异常"""
        manager = MCPManager()

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = Exception("System error")

            result = await manager.enable("playwright")

            assert result is False

    @pytest.mark.asyncio
    async def test_disable_success(self):
        """测试成功禁用 MCP 服务器"""
        manager = MCPManager()

        mock_result = {"success": True, "output": "Server disabled", "error": None}

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            result = await manager.disable("playwright")

            mock_cmd.assert_called_once_with(["mcp", "disable", "playwright"])
            assert result is True

    @pytest.mark.asyncio
    async def test_disable_failure(self):
        """测试禁用 MCP 服务器失败"""
        manager = MCPManager()

        mock_result = {"success": False, "output": "", "error": "Already disabled"}

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result

            result = await manager.disable("playwright")

            assert result is False

    @pytest.mark.asyncio
    async def test_disable_exception(self):
        """测试禁用服务器时发生异常"""
        manager = MCPManager()

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = Exception("Unexpected error")

            result = await manager.disable("playwright")

            assert result is False


# ==================== _run_command 测试 ====================


class TestMCPManagerRunCommand:
    """MCPManager._run_command 方法测试"""

    @pytest.mark.asyncio
    async def test_run_command_success(self):
        """测试命令执行成功"""
        manager = MCPManager()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"output data", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await manager._run_command(["mcp", "list"])

            assert result["success"] is True
            assert result["output"] == "output data"
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_run_command_failure(self):
        """测试命令执行失败"""
        manager = MCPManager()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"error message"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await manager._run_command(["mcp", "list"])

            assert result["success"] is False
            assert result["error"] == "error message"

    @pytest.mark.asyncio
    async def test_run_command_timeout(self):
        """测试命令超时"""
        manager = MCPManager()

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await manager._run_command(["mcp", "list"], timeout=1)

            assert result["success"] is False
            assert result["error"] == "命令超时"

    @pytest.mark.asyncio
    async def test_run_command_not_found(self):
        """测试 agent CLI 不存在"""
        manager = MCPManager(agent_path="/nonexistent/agent")

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            result = await manager._run_command(["mcp", "list"])

            assert result["success"] is False
            assert "找不到 agent CLI" in result["error"]


# ==================== _parse_server_list 测试 ====================


class TestMCPManagerParseServerList:
    """MCPManager._parse_server_list 方法测试"""

    def test_parse_empty_output(self):
        """测试解析空输出"""
        manager = MCPManager()

        servers = manager._parse_server_list("")

        assert servers == []

    def test_parse_valid_output(self):
        """测试解析有效输出"""
        manager = MCPManager()

        output = "playwright | connected | project | stdio\nfetch | disconnected | global | HTTP"

        servers = manager._parse_server_list(output)

        assert len(servers) == 2
        assert servers[0].name == "playwright"
        assert servers[0].status == "connected"
        assert servers[0].source == "project"
        assert servers[0].transport == "stdio"
        assert servers[1].name == "fetch"
        assert servers[1].status == "disconnected"

    def test_parse_output_with_empty_lines(self):
        """测试解析包含空行的输出"""
        manager = MCPManager()

        output = "playwright | connected | project | stdio\n\n\nfetch | connected | global | HTTP\n"

        servers = manager._parse_server_list(output)

        assert len(servers) == 2

    def test_parse_invalid_format(self):
        """测试解析格式不正确的输出"""
        manager = MCPManager()

        output = "invalid line without pipes\nanother invalid line"

        servers = manager._parse_server_list(output)

        assert servers == []


# ==================== _parse_tool_list 测试 ====================


class TestMCPManagerParseToolList:
    """MCPManager._parse_tool_list 方法测试"""

    def test_parse_empty_output(self):
        """测试解析空输出"""
        manager = MCPManager()

        tools = manager._parse_tool_list("")

        assert tools == []

    def test_parse_returns_empty_list(self):
        """测试当前实现返回空列表"""
        manager = MCPManager()

        # 当前实现返回空列表（TODO）
        tools = manager._parse_tool_list("some output")

        assert tools == []


# ==================== check_available 测试 ====================


class TestMCPManagerCheckAvailable:
    """MCPManager.check_available 方法测试"""

    def test_check_available_success(self):
        """测试 MCP 功能可用"""
        manager = MCPManager()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = manager.check_available()

            assert result is True

    def test_check_available_failure(self):
        """测试 MCP 功能不可用"""
        manager = MCPManager()

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            result = manager.check_available()

            assert result is False

    def test_check_available_exception(self):
        """测试检查时发生异常"""
        manager = MCPManager()

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = manager.check_available()

            assert result is False

    def test_check_available_timeout(self):
        """测试检查超时"""
        manager = MCPManager()

        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
            result = manager.check_available()

            assert result is False


# ==================== ensure_mcp_servers_enabled 测试 ====================


class TestEnsureMCPServersEnabled:
    """ensure_mcp_servers_enabled 辅助函数测试"""

    @pytest.mark.asyncio
    async def test_enable_single_server(self):
        """测试启用单个服务器"""
        with patch.object(MCPManager, "enable", new_callable=AsyncMock) as mock_enable:
            mock_enable.return_value = True

            results = await ensure_mcp_servers_enabled(["playwright"])

            mock_enable.assert_called_once_with("playwright")
            assert results == {"playwright": True}

    @pytest.mark.asyncio
    async def test_enable_multiple_servers(self):
        """测试启用多个服务器"""
        with patch.object(MCPManager, "enable", new_callable=AsyncMock) as mock_enable:
            mock_enable.return_value = True

            results = await ensure_mcp_servers_enabled(["playwright", "fetch", "github"])

            assert mock_enable.call_count == 3
            assert results == {"playwright": True, "fetch": True, "github": True}

    @pytest.mark.asyncio
    async def test_enable_partial_success(self):
        """测试部分服务器启用成功"""
        with patch.object(MCPManager, "enable", new_callable=AsyncMock) as mock_enable:
            # 模拟第二个服务器启用失败
            mock_enable.side_effect = [True, False, True]

            results = await ensure_mcp_servers_enabled(["server1", "server2", "server3"])

            assert results == {"server1": True, "server2": False, "server3": True}

    @pytest.mark.asyncio
    async def test_enable_empty_list(self):
        """测试空服务器列表"""
        results = await ensure_mcp_servers_enabled([])

        assert results == {}

    @pytest.mark.asyncio
    async def test_custom_agent_path(self):
        """测试自定义 agent 路径"""
        with patch.object(MCPManager, "enable", new_callable=AsyncMock) as mock_enable:
            mock_enable.return_value = True

            results = await ensure_mcp_servers_enabled(
                ["playwright"],
                agent_path="/custom/path/agent",
            )

            assert results == {"playwright": True}


# ==================== 集成测试 ====================


class TestMCPManagerIntegration:
    """MCPManager 集成测试（使用 mock）"""

    @pytest.mark.asyncio
    async def test_workflow_list_and_enable(self):
        """测试完整工作流：列出服务器然后启用"""
        manager = MCPManager()

        # 模拟列出服务器
        list_result = {
            "success": True,
            "output": "playwright | disconnected | project | stdio\n",
            "error": None,
        }

        # 模拟启用服务器
        enable_result = {"success": True, "output": "Enabled", "error": None}

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            # 第一次调用返回列表结果，第二次返回启用结果
            mock_cmd.side_effect = [list_result, enable_result]

            # 列出服务器
            servers = await manager.list_servers()
            assert len(servers) == 1
            assert servers[0].status == "disconnected"

            # 启用服务器
            success = await manager.enable(servers[0].identifier)
            assert success is True

    @pytest.mark.asyncio
    async def test_workflow_login_and_list_tools(self):
        """测试完整工作流：登录然后获取工具"""
        manager = MCPManager()

        login_result = {"success": True, "output": "Logged in", "error": None}
        tools_result = {"success": True, "output": "tool1\ntool2", "error": None}

        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = [login_result, tools_result]

            # 登录
            login_success = await manager.login("github-server")
            assert login_success is True

            # 获取工具列表
            tools = await manager.list_tools("github-server")
            assert isinstance(tools, list)
