"""Cursor Cloud Agent 客户端

Cloud Agent 客户端，用于将任务推送到云端执行。

用法:
    from cursor.cloud.client import CursorCloudClient

    client = CursorCloudClient()
    result = await client.execute("& 实现功能")
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger

# 从本地模块导入
from .auth import CloudAuthManager
from .exceptions import (
    AuthError,
    AuthErrorCode,
    NetworkError,
    RateLimitError,
)
from .task import (
    CloudTask,
    CloudTaskOptions,
    TaskStatus,
)

# ========== 数据类 ==========

@dataclass
class CloudAgentResult:
    """云端 Agent 执行结果"""
    success: bool
    task: Optional[CloudTask] = None
    output: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    files_modified: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "task": self.task.to_dict() if self.task else None,
            "output": self.output,
            "error": self.error,
            "duration": self.duration,
            "files_modified": self.files_modified,
        }


# ========== Cloud Agent 客户端 ==========

class CursorCloudClient:
    """Cursor Cloud Agent 客户端

    用于将任务推送到云端执行。支持:
    - 使用 & 前缀将消息推送到云端
    - 查询任务状态
    - 获取任务结果
    - 取消任务

    参考: https://cursor.com/cn/docs/cloud-agent

    用法:
        client = CursorCloudClient()

        # 方式1: 直接提交任务
        result = await client.submit_task("实现用户登录功能")

        # 方式2: 使用 & 前缀（自动检测）
        result = await client.execute("& 实现用户登录功能")

        # 查询任务状态
        task = await client.get_task_status(task_id)

        # 等待任务完成
        result = await client.wait_for_completion(task_id)
    """

    # Cloud Agent 前缀
    CLOUD_PREFIX = "&"

    def __init__(
        self,
        api_base: str = "https://api.cursor.com",
        agents_endpoint: str = "/v1/agents",
        auth_manager: Optional[CloudAuthManager] = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.agents_endpoint = agents_endpoint
        self.auth_manager = auth_manager or CloudAuthManager()
        self._agent_path = self._find_agent_executable()

        # 任务缓存
        self._tasks: dict[str, CloudTask] = {}

    def _find_agent_executable(self) -> str:
        """查找 agent 可执行文件"""
        import shutil
        possible_paths = [
            shutil.which("agent"),
            "/usr/local/bin/agent",
            os.path.expanduser("~/.local/bin/agent"),
            os.path.expanduser("~/.cursor/bin/agent"),
        ]

        for path in possible_paths:
            if path and os.path.isfile(path):
                return path

        return "agent"

    @staticmethod
    def is_cloud_request(prompt: str) -> bool:
        """检测是否是云端请求（以 & 开头）

        边界情况处理:
        - None 或空字符串返回 False
        - 只有 & 的字符串返回 False（无实际内容）
        - 只有空白字符返回 False
        - 多个 & 开头只识别第一个

        Args:
            prompt: 任务 prompt

        Returns:
            是否为云端请求
        """
        # 处理 None 和非字符串类型
        if not prompt or not isinstance(prompt, str):
            return False

        stripped = prompt.strip()

        # 空字符串
        if not stripped:
            return False

        # 检查是否以 & 开头
        if not stripped.startswith(CursorCloudClient.CLOUD_PREFIX):
            return False

        # 确保 & 后面有实际内容（不只是空白）
        content_after_prefix = stripped[len(CursorCloudClient.CLOUD_PREFIX):].strip()
        return len(content_after_prefix) > 0

    @staticmethod
    def strip_cloud_prefix(prompt: str) -> str:
        """移除云端前缀"""
        stripped = prompt.strip()
        if stripped.startswith(CursorCloudClient.CLOUD_PREFIX):
            return stripped[len(CursorCloudClient.CLOUD_PREFIX):].strip()
        return stripped

    async def execute(
        self,
        prompt: str,
        options: Optional[CloudTaskOptions] = None,
        wait: bool = True,
        timeout: Optional[int] = None,
        session_id: Optional[str] = None,
        switch_to_cloud: bool = False,
    ) -> CloudAgentResult:
        """执行任务（自动检测是否推送到云端）

        如果 prompt 以 & 开头，将任务推送到云端执行。
        否则使用本地 Agent 执行。

        支持会话切换:
        - 提供 session_id 可恢复已有会话
        - switch_to_cloud=True 可将本地会话推送到云端

        Args:
            prompt: 任务 prompt（可以带 & 前缀）
            options: 任务选项
            wait: 是否等待任务完成
            timeout: 超时时间（秒）
            session_id: 可选的会话 ID（用于恢复会话）
            switch_to_cloud: 是否将会话切换到云端执行

        Returns:
            执行结果
        """
        # 处理会话切换到云端的情况
        if switch_to_cloud and session_id:
            return await self.push_to_cloud(session_id, prompt, options)

        # 处理恢复云端会话的情况
        if session_id and not switch_to_cloud:
            # 检查是否是云端会话（从缓存或云端获取）
            cached_task = self._tasks.get(session_id)
            if cached_task or session_id.startswith("cloud-"):
                return await self.resume_from_cloud(
                    session_id,
                    local=True,
                    prompt=prompt if prompt and not self.is_cloud_request(prompt) else None,
                    options=options,
                )

        if self.is_cloud_request(prompt):
            # 移除 & 前缀，提交到云端
            clean_prompt = self.strip_cloud_prefix(prompt)

            # 如果有 session_id，使用恢复模式并推送到云端
            if session_id:
                return await self.push_to_cloud(session_id, clean_prompt, options)

            return await self.submit_and_wait(clean_prompt, options, timeout) if wait else await self.submit_task(clean_prompt, options)
        else:
            # 本地执行（使用现有的 CursorAgentClient）
            from cursor.client import CursorAgentClient, CursorAgentConfig

            config = CursorAgentConfig(
                model=options.model if options else None,
                working_directory=options.working_directory if options else ".",
                force_write=options.allow_write if options else True,
                timeout=timeout or (options.timeout if options else 300),
            )

            client = CursorAgentClient(config)

            # 如果提供了 session_id，使用恢复模式
            if session_id:
                result = await client.execute(prompt, session_id=session_id)
            else:
                result = await client.execute(prompt)

            return CloudAgentResult(
                success=result.success,
                output=result.output,
                error=result.error,
                duration=result.duration,
                files_modified=result.files_modified,
            )

    async def submit_task(
        self,
        prompt: str,
        options: Optional[CloudTaskOptions] = None,
    ) -> CloudAgentResult:
        """提交任务到云端（不等待完成）

        使用 agent -b (background) 模式提交任务。
        支持增强的错误处理和友好的错误提示。

        Args:
            prompt: 任务 prompt（不含 & 前缀）
            options: 任务选项

        Returns:
            包含 task_id 的结果
        """
        # 确保已认证
        auth_status = await self.auth_manager.authenticate()
        if not auth_status.authenticated:
            error_msg = auth_status.error.user_friendly_message if auth_status.error else "未认证"
            logger.warning(f"提交任务失败: {error_msg}")
            return CloudAgentResult(
                success=False,
                error=error_msg,
            )

        options = options or CloudTaskOptions()

        try:
            # 构建命令：使用 -b (background) 模式
            cmd = [self._agent_path, "-b", "-p", prompt]

            if options.model:
                cmd.extend(["--model", options.model])

            # 输出格式使用 JSON 以便解析 task_id
            cmd.extend(["--output-format", "json"])

            env = os.environ.copy()
            api_key = self.auth_manager.get_api_key()
            if api_key:
                env["CURSOR_API_KEY"] = api_key

            # 设置工作目录
            cwd = options.working_directory or os.getcwd()

            logger.info(f"提交云端任务: {prompt[:50]}...")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60,  # 提交任务应该很快
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if process.returncode != 0:
                # 分析错误类型
                error_text = error_output or output
                error_lower = error_text.lower()

                # 认证错误检测
                if "unauthorized" in error_lower or "401" in error_lower:
                    auth_error = AuthError(
                        message=error_text or "认证失败",
                        code=AuthErrorCode.INVALID_API_KEY,
                    )
                    logger.warning(auth_error.user_friendly_message)
                    return CloudAgentResult(
                        success=False,
                        error=auth_error.user_friendly_message,
                    )

                if "forbidden" in error_lower or "403" in error_lower:
                    auth_error = AuthError(
                        message=error_text or "权限不足",
                        code=AuthErrorCode.INSUFFICIENT_PERMISSIONS,
                    )
                    logger.warning(auth_error.user_friendly_message)
                    return CloudAgentResult(
                        success=False,
                        error=auth_error.user_friendly_message,
                    )

                # 限流错误检测
                if "rate limit" in error_lower or "429" in error_lower:
                    rate_error = RateLimitError(message=error_text)
                    logger.warning(rate_error.user_friendly_message)
                    return CloudAgentResult(
                        success=False,
                        error=rate_error.user_friendly_message,
                    )

                # 通用错误
                logger.error(f"提交失败 (exit_code={process.returncode}): {error_text[:200]}")
                return CloudAgentResult(
                    success=False,
                    error=error_text or f"提交失败 (exit code: {process.returncode})",
                )

            # 解析返回的 task_id
            task_id = self._parse_task_id(output)
            if not task_id:
                # 如果没有 task_id，可能是直接返回了结果（小任务）
                task_id = f"local-{os.urandom(8).hex()}"

            # 创建任务记录
            task = CloudTask(
                task_id=task_id,
                status=TaskStatus.QUEUED,
                prompt=prompt,
                options=options,
            )

            self._tasks[task_id] = task
            logger.info(f"任务已提交: task_id={task_id}")

            return CloudAgentResult(
                success=True,
                task=task,
                output=output,
            )

        except asyncio.TimeoutError:
            timeout_error = NetworkError(
                message="提交任务超时",
                error_type="timeout",
                retry_after=5.0,
            )
            logger.warning(timeout_error.user_friendly_message)
            return CloudAgentResult(
                success=False,
                error=timeout_error.user_friendly_message,
            )
        except FileNotFoundError:
            not_found_error = NetworkError(
                message=f"找不到 agent CLI: {self._agent_path}",
                error_type="not_found",
                details={
                    "path": self._agent_path,
                    "hint": "请安装 Cursor CLI: curl https://cursor.com/install -fsS | bash",
                },
            )
            logger.error(not_found_error.message)
            return CloudAgentResult(
                success=False,
                error=f"{not_found_error.message}\n提示: 请安装 Cursor CLI",
            )
        except OSError as e:
            os_error = NetworkError.from_exception(e, context="提交任务")
            logger.error(f"提交任务 OS 错误: {os_error}")
            return CloudAgentResult(
                success=False,
                error=os_error.user_friendly_message,
            )
        except Exception as e:
            logger.error(f"提交云端任务失败: {e}", exc_info=True)
            return CloudAgentResult(
                success=False,
                error=str(e),
            )

    def _parse_task_id(self, output: str) -> Optional[str]:
        """从输出中解析 task_id"""
        try:
            # 尝试解析 JSON 输出
            data = json.loads(output)
            return data.get("task_id") or data.get("session_id") or data.get("id")
        except json.JSONDecodeError:
            pass

        # 尝试从文本中提取
        import re

        # 匹配 UUID 格式
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        match = re.search(uuid_pattern, output, re.IGNORECASE)
        if match:
            return match.group(0)

        # 匹配 session_id 格式
        session_pattern = r'session[_-]?id[:\s]+([a-zA-Z0-9_-]+)'
        match = re.search(session_pattern, output, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    async def get_task_status(self, task_id: str) -> Optional[CloudTask]:
        """获取任务状态

        Args:
            task_id: 任务 ID

        Returns:
            任务对象，如果不存在返回 None
        """
        # 首先检查本地缓存
        if task_id in self._tasks:
            task = self._tasks[task_id]
            # 如果任务还在运行，尝试刷新状态
            if task.is_running:
                await self._refresh_task_status(task)
            return task

        # 尝试从云端获取
        return await self._fetch_task_from_cloud(task_id)

    async def _refresh_task_status(self, task: CloudTask) -> None:
        """刷新任务状态"""
        try:
            # 使用 agent ls 查询会话状态
            env = os.environ.copy()
            api_key = self.auth_manager.get_api_key()
            if api_key:
                env["CURSOR_API_KEY"] = api_key

            process = await asyncio.create_subprocess_exec(
                self._agent_path, "ls",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=30,
            )

            output = stdout.decode("utf-8", errors="replace")

            # 查找任务状态（简化解析）
            if task.task_id in output:
                if "completed" in output.lower() or "done" in output.lower():
                    task.status = TaskStatus.COMPLETED
                elif "failed" in output.lower() or "error" in output.lower():
                    task.status = TaskStatus.FAILED
                elif "running" in output.lower():
                    task.status = TaskStatus.RUNNING

        except Exception as e:
            logger.debug(f"刷新任务状态失败: {e}")

    async def _fetch_task_from_cloud(self, task_id: str) -> Optional[CloudTask]:
        """从云端获取任务信息"""
        # 预留：实际 API 调用
        # 当前实现依赖 agent CLI
        return None

    async def wait_for_completion(
        self,
        task_id: str,
        timeout: Optional[int] = None,
        poll_interval: float = 2.0,
        on_progress: Optional[Callable[[CloudTask], None]] = None,
    ) -> CloudAgentResult:
        """等待任务完成

        Args:
            task_id: 任务 ID
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
            on_progress: 进度回调

        Returns:
            执行结果
        """
        timeout = timeout or 600
        start_time = time.time()

        while True:
            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return CloudAgentResult(
                    success=False,
                    error=f"等待任务完成超时 ({timeout}s)",
                )

            # 获取任务状态
            task = await self.get_task_status(task_id)
            if not task:
                return CloudAgentResult(
                    success=False,
                    error=f"任务不存在: {task_id}",
                )

            # 回调进度
            if on_progress:
                on_progress(task)

            # 检查是否完成
            if task.is_completed:
                return CloudAgentResult(
                    success=task.is_success,
                    task=task,
                    output=task.output or "",
                    error=task.error,
                    duration=task.duration or elapsed,
                    files_modified=task.files_modified,
                )

            # 等待下一次轮询
            await asyncio.sleep(poll_interval)

    async def submit_and_wait(
        self,
        prompt: str,
        options: Optional[CloudTaskOptions] = None,
        timeout: Optional[int] = None,
        on_progress: Optional[Callable[[CloudTask], None]] = None,
    ) -> CloudAgentResult:
        """提交任务并等待完成

        Args:
            prompt: 任务 prompt
            options: 任务选项
            timeout: 超时时间
            on_progress: 进度回调

        Returns:
            执行结果
        """
        # 提交任务
        submit_result = await self.submit_task(prompt, options)
        if not submit_result.success or not submit_result.task:
            return submit_result

        # 等待完成
        return await self.wait_for_completion(
            submit_result.task.task_id,
            timeout=timeout or (options.timeout if options else 600),
            on_progress=on_progress,
        )

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务

        Args:
            task_id: 任务 ID

        Returns:
            是否成功取消
        """
        task = self._tasks.get(task_id)
        if task:
            if task.is_running:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                logger.info(f"任务已取消: {task_id}")
                return True
            else:
                logger.warning(f"任务已结束，无法取消: {task_id}")
                return False

        logger.warning(f"任务不存在: {task_id}")
        return False

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 50,
    ) -> list[CloudTask]:
        """列出任务

        Args:
            status: 过滤状态
            limit: 最大返回数量

        Returns:
            任务列表
        """
        tasks = list(self._tasks.values())

        # 过滤状态
        if status:
            tasks = [t for t in tasks if t.status == status]

        # 按创建时间倒序
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]

    def get_cached_task(self, task_id: str) -> Optional[CloudTask]:
        """获取缓存的任务（同步方法）"""
        return self._tasks.get(task_id)

    async def push_to_cloud(
        self,
        session_id: str,
        prompt: Optional[str] = None,
        options: Optional[CloudTaskOptions] = None,
    ) -> CloudAgentResult:
        """将现有会话推送到云端继续执行

        将本地会话上传到云端，使其可以在云端继续运行。
        支持在本地处理到一半时切换到云端后台执行。

        Args:
            session_id: 本地会话 ID
            prompt: 可选的附加 prompt（在云端继续时使用）
            options: 任务选项

        Returns:
            包含云端任务信息的结果
        """
        # 确保已认证
        auth_status = await self.auth_manager.authenticate()
        if not auth_status.authenticated:
            error_msg = auth_status.error.user_friendly_message if auth_status.error else "未认证"
            logger.warning(f"推送到云端失败: {error_msg}")
            return CloudAgentResult(
                success=False,
                error=error_msg,
            )

        options = options or CloudTaskOptions()

        try:
            # 构建命令: 使用 --resume 恢复会话，然后用 -b 推送到后台
            cmd = [self._agent_path, "-b", "--resume", session_id]

            # 如果有附加 prompt，添加到命令中
            if prompt:
                cmd.extend(["-p", prompt])

            if options.model:
                cmd.extend(["--model", options.model])

            cmd.extend(["--output-format", "json"])

            env = os.environ.copy()
            api_key = self.auth_manager.get_api_key()
            if api_key:
                env["CURSOR_API_KEY"] = api_key

            cwd = options.working_directory or os.getcwd()

            logger.info(f"推送会话到云端: session_id={session_id}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60,
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if process.returncode != 0:
                error_text = error_output or output
                logger.error(f"推送失败 (exit_code={process.returncode}): {error_text[:200]}")
                return CloudAgentResult(
                    success=False,
                    error=error_text or f"推送失败 (exit code: {process.returncode})",
                )

            # 解析返回的云端 task_id
            cloud_task_id = self._parse_task_id(output)
            if not cloud_task_id:
                cloud_task_id = f"cloud-{session_id}"

            # 创建云端任务记录
            task = CloudTask(
                task_id=cloud_task_id,
                status=TaskStatus.RUNNING,
                prompt=prompt or f"[Resumed from {session_id}]",
                options=options,
            )

            self._tasks[cloud_task_id] = task
            logger.info(f"会话已推送到云端: cloud_task_id={cloud_task_id}")

            return CloudAgentResult(
                success=True,
                task=task,
                output=output,
            )

        except asyncio.TimeoutError:
            timeout_error = NetworkError(
                message="推送到云端超时",
                error_type="timeout",
                retry_after=5.0,
            )
            logger.warning(timeout_error.user_friendly_message)
            return CloudAgentResult(
                success=False,
                error=timeout_error.user_friendly_message,
            )
        except FileNotFoundError:
            not_found_error = NetworkError(
                message=f"找不到 agent CLI: {self._agent_path}",
                error_type="not_found",
            )
            logger.error(not_found_error.message)
            return CloudAgentResult(
                success=False,
                error=f"{not_found_error.message}\n提示: 请安装 Cursor CLI",
            )
        except Exception as e:
            logger.error(f"推送到云端失败: {e}", exc_info=True)
            return CloudAgentResult(
                success=False,
                error=str(e),
            )

    async def resume_from_cloud(
        self,
        task_id: str,
        local: bool = True,
        prompt: Optional[str] = None,
        options: Optional[CloudTaskOptions] = None,
    ) -> CloudAgentResult:
        """从云端恢复会话到本地

        获取云端会话状态，并可选择在本地继续执行。

        Args:
            task_id: 云端任务 ID
            local: 是否在本地继续执行（True=本地，False=仅获取状态）
            prompt: 可选的附加 prompt
            options: 任务选项

        Returns:
            执行结果或任务状态
        """
        # 确保已认证
        auth_status = await self.auth_manager.authenticate()
        if not auth_status.authenticated:
            error_msg = auth_status.error.user_friendly_message if auth_status.error else "未认证"
            logger.warning(f"从云端恢复失败: {error_msg}")
            return CloudAgentResult(
                success=False,
                error=error_msg,
            )

        options = options or CloudTaskOptions()

        try:
            # 构建命令: 使用 --resume 恢复云端会话
            cmd = [self._agent_path, "--resume", task_id]

            # 如果不在本地继续，只获取状态
            if not local:
                cmd.append("--output-format")
                cmd.append("json")
            else:
                # 本地继续执行，使用 -p 模式
                cmd.append("-p")
                if prompt:
                    cmd.append(prompt)
                else:
                    cmd.append("")  # 空 prompt，仅恢复上下文

            if options.model:
                cmd.extend(["--model", options.model])

            env = os.environ.copy()
            api_key = self.auth_manager.get_api_key()
            if api_key:
                env["CURSOR_API_KEY"] = api_key

            cwd = options.working_directory or os.getcwd()
            timeout = options.timeout or 300

            logger.info(f"从云端恢复会话: task_id={task_id}, local={local}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if process.returncode != 0:
                error_text = error_output or output
                error_lower = error_text.lower()

                # 会话不存在检测
                if "not found" in error_lower or "does not exist" in error_lower:
                    logger.warning(f"云端会话不存在: {task_id}")
                    return CloudAgentResult(
                        success=False,
                        error=f"云端会话不存在: {task_id}",
                    )

                logger.error(f"恢复失败 (exit_code={process.returncode}): {error_text[:200]}")
                return CloudAgentResult(
                    success=False,
                    error=error_text or f"恢复失败 (exit code: {process.returncode})",
                )

            # 更新或创建任务记录
            task = self._tasks.get(task_id)
            if not task:
                task = CloudTask(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED if local else TaskStatus.RUNNING,
                    prompt=prompt or "[Resumed from cloud]",
                    options=options,
                )
                self._tasks[task_id] = task
            else:
                if local:
                    task.status = TaskStatus.COMPLETED
                task.output = output

            # 提取修改的文件（从输出解析）
            files_modified = self._extract_modified_files(output)

            logger.info(f"会话已从云端恢复: task_id={task_id}")

            return CloudAgentResult(
                success=True,
                task=task,
                output=output,
                files_modified=files_modified,
            )

        except asyncio.TimeoutError:
            timeout_error = NetworkError(
                message=f"从云端恢复超时 ({options.timeout or 300}s)",
                error_type="timeout",
            )
            logger.warning(timeout_error.user_friendly_message)
            return CloudAgentResult(
                success=False,
                error=timeout_error.user_friendly_message,
            )
        except FileNotFoundError:
            not_found_error = NetworkError(
                message=f"找不到 agent CLI: {self._agent_path}",
                error_type="not_found",
            )
            logger.error(not_found_error.message)
            return CloudAgentResult(
                success=False,
                error=f"{not_found_error.message}\n提示: 请安装 Cursor CLI",
            )
        except Exception as e:
            logger.error(f"从云端恢复失败: {e}", exc_info=True)
            return CloudAgentResult(
                success=False,
                error=str(e),
            )

    def _extract_modified_files(self, output: str) -> list[str]:
        """从输出中提取修改的文件列表"""
        import re

        files = []

        # 尝试从 JSON 输出解析
        try:
            data = json.loads(output)
            if "files_modified" in data:
                return data["files_modified"]
        except json.JSONDecodeError:
            pass

        # 从文本输出中匹配文件路径
        # 匹配常见的文件修改模式
        patterns = [
            r'(?:wrote|created|modified|edited)\s+["\']?([^\s"\']+\.\w+)["\']?',
            r'file[:\s]+["\']?([^\s"\']+\.\w+)["\']?',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            files.extend(matches)

        return list(set(files))  # 去重


# ========== 导出 ==========

__all__ = [
    "CloudAgentResult",
    "CursorCloudClient",
]
