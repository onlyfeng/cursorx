"""MCP (Model Context Protocol) 服务器管理

通过 agent mcp 命令管理 MCP 服务器
参考: https://cursor.com/cn/docs/cli/mcp
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass

from loguru import logger


@dataclass
class MCPServer:
    """MCP 服务器信息"""

    name: str
    identifier: str
    status: str  # connected, disconnected
    source: str  # project, global
    transport: str  # stdio, HTTP, SSE


@dataclass
class MCPTool:
    """MCP 工具信息"""

    name: str
    description: str
    parameters: list


class MCPManager:
    """MCP 服务器管理器

    CLI 命令:
    - agent mcp list: 列出已配置的服务器
    - agent mcp list-tools <id>: 查看服务器的工具
    - agent mcp login <id>: 身份验证
    - agent mcp enable <id>: 启用服务器
    - agent mcp disable <id>: 禁用服务器
    """

    def __init__(self, agent_path: str = "agent"):
        self.agent_path = agent_path

    async def list_servers(self) -> list[MCPServer]:
        """列出已配置的 MCP 服务器"""
        try:
            result = await self._run_command(["mcp", "list"])
            if not result["success"]:
                logger.warning(f"获取 MCP 服务器列表失败: {result.get('error')}")
                return []

            # 解析输出（格式依赖于 CLI 输出）
            servers = self._parse_server_list(result["output"])
            return servers

        except Exception as e:
            logger.error(f"获取 MCP 服务器列表异常: {e}")
            return []

    async def list_tools(self, identifier: str) -> list[MCPTool]:
        """查看某个 MCP 服务器提供的工具"""
        try:
            result = await self._run_command(["mcp", "list-tools", identifier])
            if not result["success"]:
                logger.warning(f"获取工具列表失败: {result.get('error')}")
                return []

            tools = self._parse_tool_list(result["output"])
            return tools

        except Exception as e:
            logger.error(f"获取工具列表异常: {e}")
            return []

    async def login(self, identifier: str) -> bool:
        """登录 MCP 服务器进行身份验证"""
        try:
            result = await self._run_command(["mcp", "login", identifier])
            return result["success"]
        except Exception as e:
            logger.error(f"MCP 登录异常: {e}")
            return False

    async def enable(self, identifier: str) -> bool:
        """启用 MCP 服务器"""
        try:
            result = await self._run_command(["mcp", "enable", identifier])
            if result["success"]:
                logger.info(f"MCP 服务器已启用: {identifier}")
            return result["success"]
        except Exception as e:
            logger.error(f"启用 MCP 服务器异常: {e}")
            return False

    async def disable(self, identifier: str) -> bool:
        """禁用 MCP 服务器"""
        try:
            result = await self._run_command(["mcp", "disable", identifier])
            if result["success"]:
                logger.info(f"MCP 服务器已禁用: {identifier}")
            return result["success"]
        except Exception as e:
            logger.error(f"禁用 MCP 服务器异常: {e}")
            return False

    async def _run_command(self, args: list[str], timeout: int = 30) -> dict:
        """运行 agent mcp 命令"""
        cmd = [self.agent_path] + args

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return {
                "success": process.returncode == 0,
                "output": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if process.returncode != 0 else None,
            }

        except asyncio.TimeoutError:
            return {"success": False, "error": "命令超时"}
        except FileNotFoundError:
            return {"success": False, "error": f"找不到 agent CLI: {self.agent_path}"}

    def _parse_server_list(self, output: str) -> list[MCPServer]:
        """解析服务器列表输出"""
        servers = []
        # TODO: 根据实际 CLI 输出格式解析
        # 这里是示例解析逻辑
        lines = output.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            # 假设格式: name | status | source | transport
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                servers.append(
                    MCPServer(
                        name=parts[0],
                        identifier=parts[0],
                        status=parts[1],
                        source=parts[2],
                        transport=parts[3],
                    )
                )
        return servers

    def _parse_tool_list(self, output: str) -> list[MCPTool]:
        """解析工具列表输出"""
        tools: list[MCPTool] = []
        # TODO: 根据实际 CLI 输出格式解析
        return tools

    def check_available(self) -> bool:
        """检查 MCP 功能是否可用"""
        try:
            result = subprocess.run(
                [self.agent_path, "mcp", "list"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False


async def ensure_mcp_servers_enabled(
    servers: list[str],
    agent_path: str = "agent",
) -> dict[str, bool]:
    """确保指定的 MCP 服务器已启用

    Args:
        servers: 要启用的服务器标识符列表
        agent_path: agent CLI 路径

    Returns:
        每个服务器的启用状态
    """
    manager = MCPManager(agent_path)
    results = {}

    for server in servers:
        success = await manager.enable(server)
        results[server] = success

    return results
