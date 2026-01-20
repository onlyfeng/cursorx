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
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, Field

from cursor.client import CursorAgentClient, CursorAgentConfig, CursorAgentResult
from cursor.cloud_client import (
    AuthStatus,
    CloudAuthConfig,
    CloudAuthManager,
    CloudClientFactory,
)

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
        """从 CLI 执行结果转换

        透传 CursorAgentResult 中的所有关键字段，包括:
        - session_id: 从 stream-json system/init 事件提取的会话 ID
        - files_modified: 写入/创建的文件列表
        - files_edited: 编辑/修改的文件列表 (合并到 files_modified)
        """
        # 合并 files_modified 和 files_edited 到 files_modified 列表
        all_files = list(result.files_modified)
        # 添加 files_edited 中不在 files_modified 中的文件
        files_edited = getattr(result, 'files_edited', []) or []
        for f in files_edited:
            if f not in all_files:
                all_files.append(f)

        return cls(
            success=result.success,
            output=result.output,
            error=result.error,
            exit_code=result.exit_code,
            duration=result.duration,
            started_at=result.started_at,
            completed_at=result.completed_at,
            executor_type="cli",
            files_modified=all_files,
            session_id=result.session_id,
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
        files_modified: Optional[list[str]] = None,
    ) -> "AgentResult":
        """从 Cloud API 结果转换

        Args:
            success: 是否成功
            output: 输出内容
            error: 错误信息
            duration: 执行时长（秒）
            session_id: 会话 ID
            raw_result: 原始结果字典
            files_modified: 修改的文件列表

        Returns:
            AgentResult 实例
        """
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
            files_modified=files_modified or [],
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
    - 使用 CloudClientFactory 统一认证配置优先级

    配置来源优先级（从高到低）：
    1. agent_config.api_key
    2. auth_config.api_key
    3. 环境变量 CURSOR_API_KEY
    """

    def __init__(
        self,
        auth_config: Optional[CloudAuthConfig] = None,
        agent_config: Optional[CursorAgentConfig] = None,
        cloud_client: Optional[CursorCloudClient] = None,
    ):
        """初始化 Cloud 执行器

        使用 CloudClientFactory 统一创建认证管理器和客户端，
        确保配置来源优先级一致。

        Args:
            auth_config: Cloud 认证配置
            agent_config: Agent 配置（用于模型、超时等设置）
            cloud_client: 可选的 CursorCloudClient 实例（用于测试注入）
        """
        self._auth_config = auth_config or CloudAuthConfig()
        self._agent_config = agent_config or CursorAgentConfig()
        self._available: Optional[bool] = None
        self._auth_status: Optional[AuthStatus] = None

        # 使用 CloudClientFactory 统一创建认证管理器和客户端
        # 配置来源优先级: agent_config.api_key > auth_config.api_key > 环境变量
        if cloud_client is not None:
            self._cloud_client = cloud_client
            # 使用工厂创建认证管理器以保持一致性
            self._auth_manager = CloudClientFactory.create_auth_manager(
                agent_config=self._agent_config,
                auth_config=self._auth_config,
            )
        else:
            self._cloud_client, self._auth_manager = CloudClientFactory.create(
                agent_config=self._agent_config,
                auth_config=self._auth_config,
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
        background: bool = False,
        session_id: Optional[str] = None,
        force_cloud: bool = True,
    ) -> AgentResult:
        """通过 Cloud API 执行任务

        使用 CloudClientFactory.execute_task() 统一执行入口。
        此方法与 CursorAgentClient._execute_via_cloud() 使用相同的工厂方法，
        确保两条 Cloud 执行路径在 allow_write/timeout/session_id 恢复能力上行为一致。

        配置来源优先级:
        1. 显式参数（timeout, working_directory 等）
        2. agent_config 中的对应值
        3. auth_config 中的对应值
        4. 环境变量 CURSOR_API_KEY

        强制云端逻辑:
        - CloudAgentExecutor 默认 force_cloud=True（因为这是 Cloud 执行器）
        - 若 prompt 不以 & 开头，会自动按云端模式执行（不修改 prompt 本身）
        - 确保不重复添加 & 前缀

        执行模式行为:
        - background=False（默认）: 前台模式，等待任务完成，返回完整结果
        - background=True: 后台模式，立即返回 session_id，不等待完成（Cloud Relay 语义）

        默认行为约定（由 run.py 控制传递）:
        - 当用户任务以 '&' 前缀触发 Cloud 模式时，run.py 传递 background=True
        - 当显式 --execution-mode cloud 且未指定 --cloud-background 时，run.py 传递 background=False

        Args:
            prompt: 任务提示（可带 & 前缀）
            context: 上下文信息（暂不使用，保持接口兼容）
            working_directory: 工作目录
            timeout: 超时时间（秒）
            background: 是否使用后台模式（默认 False，等待任务完成）
            session_id: 可选的会话 ID（用于恢复会话）
            force_cloud: 强制云端执行（默认 True，自动为非 & 开头的 prompt 启用云端模式）

        Returns:
            执行结果:
            - background=False: 完整的执行结果，包含 output/files_modified 等
            - background=True: success 表示提交是否成功，session_id 用于后续查询/恢复
        """
        started_at = datetime.now()
        timeout_sec = timeout or self._agent_config.timeout

        # 根据 background 参数决定 wait 行为
        # background=True 时不等待（wait=False），立即返回 task 元信息
        # background=False 时等待完成（wait=True），返回完整结果
        wait_for_completion = not background

        try:
            # 使用 CloudClientFactory.execute_task() 统一执行入口
            # 配置来源优先级与 CursorAgentClient._execute_via_cloud() 一致
            # force_cloud=True 时，即使 prompt 不以 & 开头也会使用云端执行
            # allow_write 由 agent_config.force_write 控制，保持只读/可写语义
            cloud_result = await CloudClientFactory.execute_task(
                prompt=prompt,
                agent_config=self._agent_config,
                auth_config=self._auth_config,
                working_directory=working_directory or ".",
                timeout=timeout_sec,
                allow_write=self._agent_config.force_write,
                session_id=session_id,
                wait=wait_for_completion,
                force_cloud=force_cloud,
            )

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            # 根据 wait 模式构造返回结果
            if wait_for_completion:
                # 前台模式：返回完整结果，包含 output/files_modified
                return AgentResult.from_cloud_result(
                    success=cloud_result.success,
                    output=cloud_result.output,
                    error=cloud_result.error,
                    duration=duration,
                    session_id=cloud_result.task.task_id if cloud_result.task else None,
                    raw_result=cloud_result.to_dict(),
                    files_modified=cloud_result.files_modified,
                )
            else:
                # 后台模式：仅返回任务元信息
                # success 表示任务提交是否成功，而非任务本身是否完成
                # output/files_modified 为空（任务尚未完成）
                # raw_result 包含 task 元信息，供调用方获取更多细节
                task_id = cloud_result.task.task_id if cloud_result.task else None
                return AgentResult.from_cloud_result(
                    success=cloud_result.success,
                    output=cloud_result.output or "",  # 后台模式可能只有初始响应
                    error=cloud_result.error,
                    duration=duration,
                    session_id=task_id,
                    raw_result=cloud_result.to_dict(),
                    files_modified=[],  # 后台模式任务尚未完成，无文件修改信息
                )

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
            options: 任务选项（如果为 None，使用 agent_config 构建默认选项）

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

            # 如果未提供选项，使用工厂构建默认选项
            if options is None:
                options = CloudClientFactory.build_task_options(
                    agent_config=self._agent_config,
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

        使用 CloudClientFactory.resume_session() 统一入口。
        此方法与 CursorAgentClient 的会话恢复行为一致。

        Args:
            session_id: 会话 ID
            prompt: 可选的附加提示

        Returns:
            执行结果
        """
        started_at = datetime.now()

        try:
            # 使用 CloudClientFactory.resume_session() 统一入口
            cloud_result = await CloudClientFactory.resume_session(
                session_id=session_id,
                prompt=prompt,
                agent_config=self._agent_config,
                auth_config=self._auth_config,
                local=True,
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
                files_modified=cloud_result.files_modified,
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

    优先使用 Cloud API，不可用时自动回退到 CLI。
    支持 Cloud 失败后的冷却策略，避免频繁重试导致的抖动。
    """

    # 默认冷却时间（秒）
    DEFAULT_COOLDOWN_SECONDS: int = 300  # 5 分钟

    def __init__(
        self,
        cli_config: Optional[CursorAgentConfig] = None,
        cloud_auth_config: Optional[CloudAuthConfig] = None,
        cloud_cooldown_seconds: Optional[int] = None,
        enable_cooldown: bool = True,
    ):
        """初始化自动执行器

        Args:
            cli_config: CLI 执行器配置
            cloud_auth_config: Cloud 认证配置
            cloud_cooldown_seconds: Cloud 失败后的冷却时间（秒），默认 300 秒
            enable_cooldown: 是否启用冷却策略，默认启用
        """
        self._cli_executor = CLIAgentExecutor(cli_config)
        self._cloud_executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=cli_config,
        )
        self._preferred_executor: Optional[AgentExecutor] = None
        # Cloud 冷却策略
        self._enable_cooldown = enable_cooldown
        self._cloud_cooldown_seconds = (
            cloud_cooldown_seconds
            if cloud_cooldown_seconds is not None
            else self.DEFAULT_COOLDOWN_SECONDS
        )
        self._cloud_cooldown_until: Optional[datetime] = None
        self._cloud_failure_count: int = 0

    @property
    def executor_type(self) -> str:
        return "auto"

    @property
    def cli_executor(self) -> CLIAgentExecutor:
        return self._cli_executor

    @property
    def cloud_executor(self) -> CloudAgentExecutor:
        return self._cloud_executor

    @property
    def cloud_cooldown_seconds(self) -> int:
        """获取 Cloud 冷却时间（秒）"""
        return self._cloud_cooldown_seconds

    @property
    def is_cloud_in_cooldown(self) -> bool:
        """检查 Cloud 是否处于冷却期"""
        if not self._enable_cooldown:
            return False
        if self._cloud_cooldown_until is None:
            return False
        return datetime.now() < self._cloud_cooldown_until

    @property
    def cloud_cooldown_remaining(self) -> Optional[float]:
        """获取 Cloud 冷却剩余时间（秒），未在冷却期返回 None"""
        if not self.is_cloud_in_cooldown:
            return None
        remaining = (self._cloud_cooldown_until - datetime.now()).total_seconds()
        return max(0.0, remaining)

    def _start_cloud_cooldown(self) -> None:
        """开始 Cloud 冷却期"""
        if not self._enable_cooldown:
            return
        self._cloud_failure_count += 1
        self._cloud_cooldown_until = datetime.now() + timedelta(
            seconds=self._cloud_cooldown_seconds
        )
        logger.info(
            f"Cloud 执行失败（第 {self._cloud_failure_count} 次），"
            f"进入冷却期 {self._cloud_cooldown_seconds} 秒"
        )

    def _reset_cloud_cooldown(self) -> None:
        """重置 Cloud 冷却状态（Cloud 成功时调用）"""
        if self._cloud_cooldown_until is not None or self._cloud_failure_count > 0:
            logger.debug("Cloud 执行成功，重置冷却状态")
        self._cloud_cooldown_until = None
        self._cloud_failure_count = 0

    def _check_cooldown_expired(self) -> bool:
        """检查冷却期是否已过期，如果过期则重置并返回 True"""
        if not self._enable_cooldown:
            return True
        if self._cloud_cooldown_until is None:
            return True
        if datetime.now() >= self._cloud_cooldown_until:
            logger.info("Cloud 冷却期已结束，重新尝试 Cloud")
            self._cloud_cooldown_until = None
            # 重置 Cloud 执行器的可用性缓存，以便重新检查
            self._cloud_executor.reset_availability_cache()
            return True
        return False

    async def _select_executor(self) -> AgentExecutor:
        """选择可用的执行器

        优先级:
        1. Cloud API（如果可用且不在冷却期）
        2. CLI（回退选项）

        冷却策略:
        - Cloud 失败后进入冷却期，冷却期内不尝试 Cloud
        - 冷却期结束后重新检查 Cloud 可用性
        """
        # 检查冷却期是否已过期
        cooldown_expired = self._check_cooldown_expired()

        # 如果不在冷却期，尝试 Cloud
        if cooldown_expired and not self.is_cloud_in_cooldown:
            if await self._cloud_executor.check_available():
                logger.debug("使用 Cloud API 执行器")
                return self._cloud_executor
        elif self.is_cloud_in_cooldown:
            remaining = self.cloud_cooldown_remaining
            logger.debug(
                f"Cloud 处于冷却期，剩余 {remaining:.1f} 秒，使用 CLI 执行器"
            )

        # 回退到 CLI
        if await self._cli_executor.check_available():
            logger.debug("回退到 CLI 执行器")
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
        """自动选择执行器并执行任务

        冷却策略:
        - Cloud 失败后进入冷却期，冷却期内优先使用 CLI
        - 冷却期结束后重新尝试 Cloud
        - Cloud 成功时重置冷却状态
        """
        # 检查是否需要重新选择执行器
        # 情况1: 首次执行，没有偏好执行器
        # 情况2: 当前偏好是 CLI，但冷却期已结束，需要重新尝试 Cloud
        should_reselect = self._preferred_executor is None
        if (
            not should_reselect
            and self._preferred_executor is not None
            and self._preferred_executor.executor_type == "cli"
            and self._check_cooldown_expired()
            and self._cloud_cooldown_until is None  # 冷却已重置
        ):
            # 冷却期结束，重新选择（可能切换回 Cloud）
            should_reselect = True
            logger.debug("冷却期已结束，重新选择执行器")

        if should_reselect:
            self._preferred_executor = await self._select_executor()

        executor = self._preferred_executor

        # 执行任务
        result = await executor.execute(
            prompt=prompt,
            context=context,
            working_directory=working_directory,
            timeout=timeout,
        )

        # 处理 Cloud 执行结果
        if executor.executor_type == "cloud":
            if result.success:
                # Cloud 成功，重置冷却状态
                self._reset_cloud_cooldown()
            else:
                # Cloud 失败，启动冷却并回退到 CLI
                logger.info("Cloud 执行失败，尝试回退到 CLI")
                self._start_cloud_cooldown()
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

    def reset_cooldown(self) -> None:
        """手动重置 Cloud 冷却状态

        调用此方法后，下次执行将重新尝试 Cloud（如果可用）
        """
        self._cloud_cooldown_until = None
        self._cloud_failure_count = 0
        self._cloud_executor.reset_availability_cache()
        # 重置偏好，以便重新选择
        self._preferred_executor = None
        logger.debug("手动重置 Cloud 冷却状态")


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
