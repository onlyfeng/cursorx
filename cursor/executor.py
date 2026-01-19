"""Agent Executor 抽象层

提供统一的 Agent 执行接口，支持多种执行模式:
- cli: 通过本地 Cursor CLI 执行
- cloud: 通过 Cloud API 执行
- auto: 自动选择，Cloud 不可用时回退到 CLI

用法:
    from cursor.executor import AgentExecutorFactory, ExecutionMode

    # 创建 Executor
    executor = AgentExecutorFactory.create(mode=ExecutionMode.AUTO)

    # 执行任务
    result = await executor.execute(
        prompt="分析代码结构",
        context={"files": ["main.py"]},
    )
"""
import asyncio
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, Field

from cursor.client import CursorAgentClient, CursorAgentConfig, CursorAgentResult
from cursor.cloud_client import AuthStatus, CloudAuthConfig, CloudAuthManager

# 导入 Cloud Client 相关类
from cursor.cloud.client import CloudAgentResult, CursorCloudClient
from cursor.cloud.task import CloudTaskOptions, TaskStatus


class ExecutionMode(str, Enum):
    """执行模式"""
    CLI = "cli"       # 本地 CLI 执行（完整 agent 模式）
    CLOUD = "cloud"   # Cloud API 执行
    AUTO = "auto"     # 自动选择（Cloud 优先，不可用时回退到 CLI）
    PLAN = "plan"     # 规划模式（只分析不执行，对应 --mode plan）
    ASK = "ask"       # 问答模式（仅回答问题，不修改文件，对应 --mode ask）


class AgentResult(BaseModel):
    """统一的 Agent 执行结果"""
    success: bool
    output: str = ""
    error: Optional[str] = None
    exit_code: int = 0
    duration: float = 0.0  # 秒
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # 执行信息
    executor_type: str = ""  # cli, cloud
    files_modified: list[str] = Field(default_factory=list)
    session_id: Optional[str] = None

    # 原始结果（用于调试）
    raw_result: Optional[dict[str, Any]] = None

    @classmethod
    def from_cli_result(cls, result: CursorAgentResult) -> "AgentResult":
        """从 CLI 执行结果转换"""
        return cls(
            success=result.success,
            output=result.output,
            error=result.error,
            exit_code=result.exit_code,
            duration=result.duration,
            started_at=result.started_at,
            completed_at=result.completed_at,
            executor_type="cli",
            files_modified=result.files_modified,
        )

    @classmethod
    def from_cloud_result(
        cls,
        success: bool,
        output: str,
        error: Optional[str] = None,
        duration: float = 0.0,
        session_id: Optional[str] = None,
        raw_result: Optional[dict[str, Any]] = None,
    ) -> "AgentResult":
        """从 Cloud API 结果转换"""
        now = datetime.now()
        return cls(
            success=success,
            output=output,
            error=error,
            duration=duration,
            started_at=now,
            completed_at=now,
            executor_type="cloud",
            session_id=session_id,
            raw_result=raw_result,
        )


@runtime_checkable
class AgentExecutor(Protocol):
    """Agent 执行器协议（抽象基类）

    定义了所有执行器必须实现的接口
    """

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> AgentResult:
        """执行 Agent 任务

        Args:
            prompt: 给 Agent 的指令/提示
            context: 上下文信息（文件列表、任务信息等）
            working_directory: 工作目录
            timeout: 超时时间（秒）

        Returns:
            执行结果
        """
        ...

    @abstractmethod
    async def check_available(self) -> bool:
        """检查执行器是否可用

        Returns:
            是否可用
        """
        ...

    @property
    @abstractmethod
    def executor_type(self) -> str:
        """执行器类型标识"""
        ...


class CLIAgentExecutor:
    """CLI Agent 执行器

    封装 CursorAgentClient，通过本地 Cursor CLI 执行任务
    支持 --mode 参数指定执行模式（agent/plan/ask）
    """

    def __init__(
        self,
        config: Optional[CursorAgentConfig] = None,
        mode: Optional[str] = None,
    ):
        """初始化 CLI 执行器

        Args:
            config: Cursor Agent 配置
            mode: CLI 工作模式（plan/ask/agent），会覆盖 config 中的 mode
        """
        self._config = config or CursorAgentConfig()
        # 如果显式指定了 mode，覆盖配置中的 mode
        if mode is not None:
            self._config = self._config.model_copy(update={"mode": mode})
        self._client = CursorAgentClient(self._config)
        self._available: Optional[bool] = None

    @property
    def executor_type(self) -> str:
        return "cli"

    @property
    def config(self) -> CursorAgentConfig:
        return self._config

    @property
    def client(self) -> CursorAgentClient:
        return self._client

    @property
    def cli_mode(self) -> Optional[str]:
        """返回当前 CLI 工作模式（plan/ask/agent）"""
        return self._config.mode

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> AgentResult:
        """通过 CLI 执行任务"""
        cli_result = await self._client.execute(
            instruction=prompt,
            context=context,
            working_directory=working_directory,
            timeout=timeout,
        )
        return AgentResult.from_cli_result(cli_result)

    async def execute_with_retry(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> AgentResult:
        """带重试的执行"""
        cli_result = await self._client.execute_with_retry(
            instruction=prompt,
            context=context,
            working_directory=working_directory,
            max_retries=max_retries,
        )
        return AgentResult.from_cli_result(cli_result)

    async def check_available(self) -> bool:
        """检查 CLI 是否可用"""
        if self._available is None:
            self._available = self._client.check_agent_available()
        return self._available

    def check_available_sync(self) -> bool:
        """同步版本：检查 CLI 是否可用"""
        if self._available is None:
            self._available = self._client.check_agent_available()
        return self._available


class PlanAgentExecutor:
    """规划模式 Agent 执行器

    使用 --mode plan 参数执行，只分析不执行
    适合用于 Planner Agent，生成任务计划

    特点：
    - 使用规划模式（--mode plan）
    - 只分析和规划，不修改任何文件
    - 适合生成任务计划、分析代码结构等场景
    """

    def __init__(self, config: Optional[CursorAgentConfig] = None):
        """初始化规划模式执行器

        Args:
            config: Cursor Agent 配置（mode 会被强制设为 plan）
        """
        base_config = config or CursorAgentConfig()
        # 强制设置为规划模式，不允许修改文件
        self._config = base_config.model_copy(update={
            "mode": "plan",
            "force_write": False,  # 规划模式不修改文件
        })
        self._cli_executor = CLIAgentExecutor(config=self._config)

    @property
    def executor_type(self) -> str:
        return "plan"

    @property
    def config(self) -> CursorAgentConfig:
        return self._config

    @property
    def client(self) -> CursorAgentClient:
        return self._cli_executor.client

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> AgentResult:
        """使用规划模式执行任务"""
        result = await self._cli_executor.execute(
            prompt=prompt,
            context=context,
            working_directory=working_directory,
            timeout=timeout,
        )
        # 更新 executor_type 为 plan
        result.executor_type = "plan"
        return result

    async def check_available(self) -> bool:
        """检查执行器是否可用"""
        return await self._cli_executor.check_available()

    def check_available_sync(self) -> bool:
        """同步版本：检查执行器是否可用"""
        return self._cli_executor.check_available_sync()


class AskAgentExecutor:
    """问答模式 Agent 执行器

    使用 --mode ask 参数执行，仅回答问题，不修改文件
    适合用于咨询场景，代码解释等

    特点：
    - 使用问答模式（--mode ask）
    - 仅回答问题和提供建议，不修改文件
    - 适合代码解释、问题咨询等场景
    """

    def __init__(self, config: Optional[CursorAgentConfig] = None):
        """初始化问答模式执行器

        Args:
            config: Cursor Agent 配置（mode 会被强制设为 ask）
        """
        base_config = config or CursorAgentConfig()
        # 强制设置为问答模式，不允许修改文件
        self._config = base_config.model_copy(update={
            "mode": "ask",
            "force_write": False,  # 问答模式不修改文件
        })
        self._cli_executor = CLIAgentExecutor(config=self._config)

    @property
    def executor_type(self) -> str:
        return "ask"

    @property
    def config(self) -> CursorAgentConfig:
        return self._config

    @property
    def client(self) -> CursorAgentClient:
        return self._cli_executor.client

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> AgentResult:
        """使用问答模式执行任务"""
        result = await self._cli_executor.execute(
            prompt=prompt,
            context=context,
            working_directory=working_directory,
            timeout=timeout,
        )
        # 更新 executor_type 为 ask
        result.executor_type = "ask"
        return result

    async def check_available(self) -> bool:
        """检查执行器是否可用"""
        return await self._cli_executor.check_available()

    def check_available_sync(self) -> bool:
        """同步版本：检查执行器是否可用"""
        return self._cli_executor.check_available_sync()


class CloudAgentExecutor:
    """Cloud Agent 执行器

    通过 Cursor Cloud API 执行任务。
    使用 CursorCloudClient 实现后台任务提交、轮询和恢复功能。

    特点:
    - 支持 & 前缀自动识别云端请求
    - 使用 -b (background) 模式提交后台任务
    - 支持任务状态轮询和结果获取
    - 支持会话恢复功能
    """

    def __init__(
        self,
        auth_config: Optional[CloudAuthConfig] = None,
        agent_config: Optional[CursorAgentConfig] = None,
        cloud_client: Optional[CursorCloudClient] = None,
    ):
        """初始化 Cloud 执行器

        Args:
            auth_config: Cloud 认证配置
            agent_config: Agent 配置（用于模型、超时等设置）
            cloud_client: 可选的 CursorCloudClient 实例（用于测试注入）
        """
        self._auth_config = auth_config or CloudAuthConfig()
        self._agent_config = agent_config or CursorAgentConfig()
        self._auth_manager = CloudAuthManager(self._auth_config)
        self._available: Optional[bool] = None
        self._auth_status: Optional[AuthStatus] = None

        # 初始化 Cloud Client
        if cloud_client is not None:
            self._cloud_client = cloud_client
        else:
            self._cloud_client = CursorCloudClient(
                auth_manager=self._auth_manager,
            )

    @property
    def executor_type(self) -> str:
        return "cloud"

    @property
    def auth_manager(self) -> CloudAuthManager:
        return self._auth_manager

    @property
    def agent_config(self) -> CursorAgentConfig:
        return self._agent_config

    @property
    def cloud_client(self) -> CursorCloudClient:
        """获取 Cloud Client 实例"""
        return self._cloud_client

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
        background: bool = True,
        session_id: Optional[str] = None,
    ) -> AgentResult:
        """通过 Cloud API 执行任务

        使用 CursorCloudClient 提交任务到云端执行。
        默认使用后台模式 (-b) 提交任务并轮询结果。

        Args:
            prompt: 任务提示（可带 & 前缀）
            context: 上下文信息
            working_directory: 工作目录
            timeout: 超时时间（秒）
            background: 是否使用后台模式（默认 True）
            session_id: 可选的会话 ID（用于恢复会话）

        Returns:
            执行结果
        """
        started_at = datetime.now()
        timeout_sec = timeout or self._agent_config.timeout

        try:
            # 验证认证状态
            if not self._auth_status or not self._auth_status.authenticated:
                self._auth_status = await self._auth_manager.authenticate()

            if not self._auth_status.authenticated:
                error_msg = self._auth_status.error or "未认证"
                return AgentResult(
                    success=False,
                    error=f"Cloud 认证失败: {error_msg}",
                    executor_type="cloud",
                    started_at=started_at,
                    completed_at=datetime.now(),
                )

            # 构建任务选项
            task_options = CloudTaskOptions(
                model=self._agent_config.model,
                working_directory=working_directory or ".",
                allow_write=self._agent_config.force_write,
                timeout=timeout_sec,
            )

            # 使用 CursorCloudClient 执行任务
            result = await self._execute_via_cloud_client(
                prompt=prompt,
                options=task_options,
                timeout=timeout_sec,
                session_id=session_id,
            )

            return result

        except asyncio.TimeoutError:
            return AgentResult(
                success=False,
                error=f"Cloud API 执行超时 ({timeout_sec}s)",
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Cloud API 执行异常: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
            )

    async def _execute_via_cloud_client(
        self,
        prompt: str,
        options: CloudTaskOptions,
        timeout: int,
        session_id: Optional[str] = None,
    ) -> AgentResult:
        """通过 CursorCloudClient 执行任务

        使用 Cloud Client 的 execute 方法：
        1. 如果 prompt 以 & 开头，自动识别为云端请求
        2. 使用后台模式提交任务
        3. 轮询等待任务完成
        4. 返回执行结果
        """
        started_at = datetime.now()

        try:
            # 使用 CursorCloudClient 执行
            # 默认等待任务完成（wait=True）
            cloud_result: CloudAgentResult = await self._cloud_client.execute(
                prompt=prompt,
                options=options,
                wait=True,
                timeout=timeout,
                session_id=session_id,
            )

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            # 转换为 AgentResult
            return AgentResult.from_cloud_result(
                success=cloud_result.success,
                output=cloud_result.output,
                error=cloud_result.error,
                duration=duration,
                session_id=cloud_result.task.task_id if cloud_result.task else None,
                raw_result=cloud_result.to_dict(),
            )

        except asyncio.TimeoutError:
            logger.error(f"Cloud Client 执行超时 ({timeout}s)")
            return AgentResult(
                success=False,
                error=f"Cloud API 执行超时 ({timeout}s)",
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
            )
        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            logger.error(f"Cloud Client 执行失败: {error_msg}")
            return AgentResult(
                success=False,
                error=error_msg,
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
            )

    async def submit_background_task(
        self,
        prompt: str,
        options: Optional[CloudTaskOptions] = None,
    ) -> AgentResult:
        """提交后台任务（不等待完成）

        使用 -b (background) 模式提交任务，立即返回任务 ID。
        后续可通过 get_task_status 或 wait_for_task 获取结果。

        Args:
            prompt: 任务提示
            options: 任务选项

        Returns:
            包含 task_id 的结果（可用于后续查询）
        """
        started_at = datetime.now()

        try:
            if not self._auth_status or not self._auth_status.authenticated:
                self._auth_status = await self._auth_manager.authenticate()

            if not self._auth_status.authenticated:
                return AgentResult(
                    success=False,
                    error="Cloud 认证失败",
                    executor_type="cloud",
                    started_at=started_at,
                    completed_at=datetime.now(),
                )

            # 提交任务（不等待完成）
            cloud_result = await self._cloud_client.submit_task(prompt, options)

            return AgentResult(
                success=cloud_result.success,
                output=cloud_result.output,
                error=cloud_result.error,
                executor_type="cloud",
                session_id=cloud_result.task.task_id if cloud_result.task else None,
                started_at=started_at,
                completed_at=datetime.now(),
                raw_result=cloud_result.to_dict(),
            )

        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
            )

    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[int] = None,
    ) -> AgentResult:
        """等待后台任务完成

        Args:
            task_id: 任务 ID
            timeout: 超时时间（秒）

        Returns:
            任务执行结果
        """
        started_at = datetime.now()
        timeout_sec = timeout or self._agent_config.timeout

        try:
            cloud_result = await self._cloud_client.wait_for_completion(
                task_id=task_id,
                timeout=timeout_sec,
            )

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            return AgentResult.from_cloud_result(
                success=cloud_result.success,
                output=cloud_result.output,
                error=cloud_result.error,
                duration=duration,
                session_id=task_id,
                raw_result=cloud_result.to_dict(),
            )

        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
            )

    async def resume_session(
        self,
        session_id: str,
        prompt: Optional[str] = None,
    ) -> AgentResult:
        """恢复云端会话

        Args:
            session_id: 会话 ID
            prompt: 可选的附加提示

        Returns:
            执行结果
        """
        started_at = datetime.now()

        try:
            cloud_result = await self._cloud_client.resume_from_cloud(
                task_id=session_id,
                local=True,
                prompt=prompt,
            )

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            return AgentResult.from_cloud_result(
                success=cloud_result.success,
                output=cloud_result.output,
                error=cloud_result.error,
                duration=duration,
                session_id=session_id,
                raw_result=cloud_result.to_dict(),
            )

        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
            )

    async def check_available(self) -> bool:
        """检查 Cloud API 是否可用

        检查条件:
        1. 认证状态有效
        2. Cloud Client 可用
        """
        if self._available is not None:
            return self._available

        try:
            # 验证认证
            self._auth_status = await self._auth_manager.authenticate()
            if not self._auth_status.authenticated:
                self._available = False
                return False

            # Cloud API 可用（认证成功即可用）
            self._available = True
            return True

        except Exception as e:
            logger.debug(f"Cloud API 可用性检查失败: {e}")
            self._available = False
            return False

    def check_available_sync(self) -> bool:
        """同步版本：检查 Cloud API 是否可用"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已在异步上下文中，返回缓存值或 False
                return self._available if self._available is not None else False
            return loop.run_until_complete(self.check_available())
        except RuntimeError:
            return False

    def reset_availability_cache(self) -> None:
        """重置可用性缓存，下次检查时重新验证"""
        self._available = None
        self._auth_status = None


class AutoAgentExecutor:
    """自动选择执行器

    优先使用 Cloud API，不可用时自动回退到 CLI
    """

    def __init__(
        self,
        cli_config: Optional[CursorAgentConfig] = None,
        cloud_auth_config: Optional[CloudAuthConfig] = None,
    ):
        """初始化自动执行器

        Args:
            cli_config: CLI 执行器配置
            cloud_auth_config: Cloud 认证配置
        """
        self._cli_executor = CLIAgentExecutor(cli_config)
        self._cloud_executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=cli_config,
        )
        self._preferred_executor: Optional[AgentExecutor] = None

    @property
    def executor_type(self) -> str:
        return "auto"

    @property
    def cli_executor(self) -> CLIAgentExecutor:
        return self._cli_executor

    @property
    def cloud_executor(self) -> CloudAgentExecutor:
        return self._cloud_executor

    async def _select_executor(self) -> AgentExecutor:
        """选择可用的执行器

        优先级:
        1. Cloud API（如果可用）
        2. CLI（回退选项）
        """
        # 尝试 Cloud
        if await self._cloud_executor.check_available():
            logger.debug("使用 Cloud API 执行器")
            return self._cloud_executor

        # 回退到 CLI
        if await self._cli_executor.check_available():
            logger.debug("Cloud 不可用，回退到 CLI 执行器")
            return self._cli_executor

        # 两者都不可用，仍然返回 CLI（可能会失败）
        logger.warning("Cloud 和 CLI 均不可用，尝试使用 CLI")
        return self._cli_executor

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> AgentResult:
        """自动选择执行器并执行任务"""
        # 选择执行器
        if self._preferred_executor is None:
            self._preferred_executor = await self._select_executor()

        executor = self._preferred_executor

        # 执行任务
        result = await executor.execute(
            prompt=prompt,
            context=context,
            working_directory=working_directory,
            timeout=timeout,
        )

        # 如果 Cloud 失败，尝试回退到 CLI
        if not result.success and executor.executor_type == "cloud":
            logger.info("Cloud 执行失败，尝试回退到 CLI")
            self._preferred_executor = self._cli_executor
            result = await self._cli_executor.execute(
                prompt=prompt,
                context=context,
                working_directory=working_directory,
                timeout=timeout,
            )

        return result

    async def check_available(self) -> bool:
        """检查是否有可用的执行器"""
        cloud_ok = await self._cloud_executor.check_available()
        cli_ok = await self._cli_executor.check_available()
        return cloud_ok or cli_ok

    def reset_preference(self) -> None:
        """重置执行器偏好，下次执行时重新选择"""
        self._preferred_executor = None


class AgentExecutorFactory:
    """Agent 执行器工厂

    根据配置创建对应的执行器实例

    用法:
        # 创建 CLI 执行器
        executor = AgentExecutorFactory.create(mode=ExecutionMode.CLI)

        # 创建自动选择执行器
        executor = AgentExecutorFactory.create(mode=ExecutionMode.AUTO)

        # 使用配置创建
        config = CursorAgentConfig(model="gpt-5.2-high")
        executor = AgentExecutorFactory.create(
            mode=ExecutionMode.CLI,
            cli_config=config,
        )
    """

    @staticmethod
    def create(
        mode: ExecutionMode = ExecutionMode.AUTO,
        cli_config: Optional[CursorAgentConfig] = None,
        cloud_auth_config: Optional[CloudAuthConfig] = None,
    ) -> AgentExecutor:
        """创建执行器实例

        Args:
            mode: 执行模式
            cli_config: CLI 执行器配置
            cloud_auth_config: Cloud 认证配置

        Returns:
            对应的执行器实例

        支持的模式:
            - CLI: 本地 CLI 执行（完整 agent 模式）
            - CLOUD: Cloud API 执行
            - AUTO: 自动选择（Cloud 优先，不可用时回退到 CLI）
            - PLAN: 规划模式（只分析不执行，对应 --mode plan）
            - ASK: 问答模式（仅回答问题，不修改文件，对应 --mode ask）
        """
        if mode == ExecutionMode.CLI:
            return CLIAgentExecutor(cli_config)
        elif mode == ExecutionMode.CLOUD:
            return CloudAgentExecutor(
                auth_config=cloud_auth_config,
                agent_config=cli_config,
            )
        elif mode == ExecutionMode.AUTO:
            return AutoAgentExecutor(
                cli_config=cli_config,
                cloud_auth_config=cloud_auth_config,
            )
        elif mode == ExecutionMode.PLAN:
            return PlanAgentExecutor(cli_config)
        elif mode == ExecutionMode.ASK:
            return AskAgentExecutor(cli_config)
        else:
            raise ValueError(f"未知的执行模式: {mode}")

    @staticmethod
    def create_from_config(
        config: CursorAgentConfig,
        cloud_auth_config: Optional[CloudAuthConfig] = None,
    ) -> AgentExecutor:
        """从配置创建执行器

        根据 config.execution_mode 自动选择执行模式

        Args:
            config: Cursor Agent 配置
            cloud_auth_config: Cloud 认证配置

        Returns:
            对应的执行器实例
        """
        mode = getattr(config, 'execution_mode', ExecutionMode.AUTO)

        # 确保 mode 是 ExecutionMode 类型
        if isinstance(mode, str):
            mode = ExecutionMode(mode)

        return AgentExecutorFactory.create(
            mode=mode,
            cli_config=config,
            cloud_auth_config=cloud_auth_config,
        )

    @staticmethod
    async def create_best_available(
        cli_config: Optional[CursorAgentConfig] = None,
        cloud_auth_config: Optional[CloudAuthConfig] = None,
    ) -> AgentExecutor:
        """创建最佳可用执行器

        检查可用性后返回最佳执行器:
        1. Cloud（如果可用）
        2. CLI（如果可用）
        3. CLI（即使不可用，作为最后手段）

        Args:
            cli_config: CLI 执行器配置
            cloud_auth_config: Cloud 认证配置

        Returns:
            最佳可用的执行器实例
        """
        # 尝试 Cloud
        cloud_executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=cli_config,
        )
        if await cloud_executor.check_available():
            logger.info("选择 Cloud API 执行器")
            return cloud_executor

        # 回退到 CLI
        cli_executor = CLIAgentExecutor(cli_config)
        if await cli_executor.check_available():
            logger.info("选择 CLI 执行器")
            return cli_executor

        # 都不可用，返回 CLI 并记录警告
        logger.warning("没有可用的执行器，返回 CLI 执行器（可能会失败）")
        return cli_executor


# ========== 便捷函数 ==========

async def execute_agent(
    prompt: str,
    context: Optional[dict[str, Any]] = None,
    mode: ExecutionMode = ExecutionMode.AUTO,
    config: Optional[CursorAgentConfig] = None,
) -> AgentResult:
    """便捷函数：执行 Agent 任务

    Args:
        prompt: 给 Agent 的指令
        context: 上下文信息
        mode: 执行模式
        config: Agent 配置

    Returns:
        执行结果
    """
    executor = AgentExecutorFactory.create(mode=mode, cli_config=config)
    return await executor.execute(prompt=prompt, context=context)


def execute_agent_sync(
    prompt: str,
    context: Optional[dict[str, Any]] = None,
    mode: ExecutionMode = ExecutionMode.CLI,
    config: Optional[CursorAgentConfig] = None,
) -> AgentResult:
    """同步版本：执行 Agent 任务"""
    return asyncio.get_event_loop().run_until_complete(
        execute_agent(prompt=prompt, context=context, mode=mode, config=config)
    )
