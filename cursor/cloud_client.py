"""Cursor Cloud 客户端入口模块

统一入口，重新导出所有 cloud 子模块的公共 API:
- 异常类和错误处理工具 (cloud.exceptions)
- 认证管理 (cloud.auth)
- 任务管理 (cloud.task)
- Cloud Agent 客户端 (cloud.client)
- CloudClientFactory (统一的 Cloud Client 构造工厂)

配置来源优先级（从高到低）：
1. 显式参数 api_key
2. agent_config.api_key (如果提供)
3. auth_config.api_key (如果提供)
4. 环境变量 CURSOR_API_KEY
5. 环境变量 CURSOR_CLOUD_API_KEY（备选）
6. config.yaml 中的 cloud_agent.api_key
"""
import os
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

# 导入统一配置构建函数
from core.config import build_cloud_client_config

# 导入异常类和错误处理工具
# 导入认证模块
from .cloud.auth import (
    AuthStatus,
    AuthToken,
    CloudAuthConfig,
    CloudAuthManager,
    get_api_key,
    require_auth,
    verify_auth,
    verify_auth_sync,
)

# 导入 Cloud Agent 客户端
from .cloud.client import (
    CloudAgentResult,
    CursorCloudClient,
)
from .cloud.exceptions import (
    AuthError,
    AuthErrorCode,
    CloudAgentError,
    NetworkError,
    RateLimitError,
    TaskError,
    handle_http_error,
)
from .cloud.retry import (
    RetryConfig,
    retry_async,
    with_retry,
)

# 导入任务管理模块
from .cloud.task import (
    CloudTask,
    CloudTaskClient,
    CloudTaskOptions,
    TaskResult,
    TaskStatus,
)

# 导入流式事件类型
from .streaming import (
    StreamEvent,
    StreamEventType,
    ToolCallInfo,
    parse_stream_event,
)

# 别名（统一命名风格）
CloudAgentClient = CursorCloudClient
CloudAgentConfig = CloudAuthConfig


# ========== CloudClientFactory ==========


class CloudClientFactory:
    """Cloud Client 工厂类

    统一封装 Cloud 认证与客户端构造逻辑，确保两条 Cloud 执行路径行为一致。

    配置来源优先级（从高到低）：
    1. 显式参数 api_key
    2. agent_config.api_key (如果提供)
    3. auth_config.api_key (如果提供)
    4. 环境变量 CURSOR_API_KEY

    用法：
        # 基本用法
        client, auth_manager = CloudClientFactory.create()

        # 带配置
        from cursor.client import CursorAgentConfig
        agent_config = CursorAgentConfig(api_key="explicit-key")
        client, auth_manager = CloudClientFactory.create(agent_config=agent_config)

        # 使用 CloudTaskOptions
        options = CloudClientFactory.build_task_options(
            agent_config=agent_config,
            working_directory="/path",
            timeout=600,
        )
    """

    @staticmethod
    def resolve_api_key(
        explicit_api_key: Optional[str] = None,
        agent_config: Optional[Any] = None,
        auth_config: Optional[CloudAuthConfig] = None,
    ) -> Optional[str]:
        """解析 API Key，遵循统一优先级

        优先级：
        1. explicit_api_key（显式参数）
        2. agent_config.api_key（Agent 配置）
        3. auth_config.api_key（认证配置）
        4. 环境变量 CURSOR_API_KEY
        5. 环境变量 CURSOR_CLOUD_API_KEY（备选）
        6. config.yaml 中的 cloud_agent.api_key

        Args:
            explicit_api_key: 显式传入的 API Key
            agent_config: CursorAgentConfig 实例（可选）
            auth_config: CloudAuthConfig 实例（可选）

        Returns:
            解析后的 API Key，如果都没有则返回 None
        """
        # 1. 显式参数最高优先级
        if explicit_api_key:
            logger.debug("使用显式传入的 API Key")
            return explicit_api_key

        # 2. agent_config.api_key
        if agent_config and hasattr(agent_config, 'api_key') and agent_config.api_key:
            logger.debug("使用 agent_config.api_key")
            return agent_config.api_key

        # 3. auth_config.api_key
        if auth_config and auth_config.api_key:
            logger.debug("使用 auth_config.api_key")
            return auth_config.api_key

        # 4. 环境变量 CURSOR_API_KEY
        env_key = os.environ.get("CURSOR_API_KEY")
        if env_key:
            logger.debug("使用环境变量 CURSOR_API_KEY")
            return env_key

        # 5. 环境变量 CURSOR_CLOUD_API_KEY（备选）
        env_cloud_key = os.environ.get("CURSOR_CLOUD_API_KEY")
        if env_cloud_key:
            logger.debug("使用环境变量 CURSOR_CLOUD_API_KEY")
            return env_cloud_key

        # 6. 从 config.yaml 获取（通过 build_cloud_client_config）
        # 注意：这里只检查 config.yaml，不再递归调用环境变量
        try:
            cloud_config = build_cloud_client_config()
            yaml_api_key = cloud_config.get("api_key")
            # 如果 yaml_api_key 来自环境变量，则已在上面处理过，这里只取 config.yaml 的值
            # build_cloud_client_config 已处理优先级，这里直接使用其结果
            if yaml_api_key:
                logger.debug("使用 config.yaml 中的 cloud_agent.api_key")
                return yaml_api_key
        except Exception:
            pass

        return None

    @staticmethod
    def create_auth_config(
        explicit_api_key: Optional[str] = None,
        agent_config: Optional[Any] = None,
        auth_config: Optional[CloudAuthConfig] = None,
        auth_timeout: Optional[int] = None,
        base_url: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> CloudAuthConfig:
        """创建统一的 CloudAuthConfig

        遵循配置来源优先级解析 API Key，并从 config.yaml 获取默认配置。

        配置来源优先级（从高到低）：
        1. 显式参数（explicit_api_key, auth_timeout, base_url, max_retries）
        2. agent_config.api_key
        3. auth_config 中的对应值
        4. 环境变量 CURSOR_API_KEY
        5. 环境变量 CURSOR_CLOUD_API_KEY（备选）
        6. config.yaml 中的 cloud_agent 配置

        Args:
            explicit_api_key: 显式传入的 API Key
            agent_config: CursorAgentConfig 实例（可选）
            auth_config: 基础 CloudAuthConfig 实例（可选）
            auth_timeout: 认证超时时间（可选）
            base_url: API 基础 URL（可选）
            max_retries: 最大重试次数（可选）

        Returns:
            配置好的 CloudAuthConfig 实例
        """
        # 从 config.yaml 获取默认配置
        cloud_config = build_cloud_client_config()

        # 解析 API Key
        resolved_api_key = CloudClientFactory.resolve_api_key(
            explicit_api_key=explicit_api_key,
            agent_config=agent_config,
            auth_config=auth_config,
        )

        # 解析其他配置项，遵循优先级
        resolved_auth_timeout = (
            auth_timeout
            if auth_timeout is not None
            else (auth_config.auth_timeout if auth_config else None)
        )
        if resolved_auth_timeout is None:
            resolved_auth_timeout = cloud_config.get("auth_timeout", 30)

        resolved_base_url = (
            base_url
            if base_url is not None
            else (auth_config.api_base_url if auth_config else None)
        )
        if resolved_base_url is None:
            resolved_base_url = cloud_config.get("base_url", "https://api.cursor.com")

        resolved_max_retries = (
            max_retries
            if max_retries is not None
            else (auth_config.max_retries if auth_config else None)
        )
        if resolved_max_retries is None:
            resolved_max_retries = cloud_config.get("max_retries", 3)

        # 创建新的配置，合并参数
        if auth_config:
            # 基于现有配置创建，覆盖解析后的值
            new_config = auth_config.model_copy(update={
                "api_key": resolved_api_key,
                "auth_timeout": resolved_auth_timeout,
                "api_base_url": resolved_base_url,
                "max_retries": resolved_max_retries,
            })
            return new_config
        else:
            # 创建新配置
            return CloudAuthConfig(
                api_key=resolved_api_key,
                auth_timeout=resolved_auth_timeout,
                api_base_url=resolved_base_url,
                max_retries=resolved_max_retries,
            )

    @staticmethod
    def create_auth_manager(
        explicit_api_key: Optional[str] = None,
        agent_config: Optional[Any] = None,
        auth_config: Optional[CloudAuthConfig] = None,
        auth_timeout: Optional[int] = None,
    ) -> CloudAuthManager:
        """创建统一的 CloudAuthManager

        Args:
            explicit_api_key: 显式传入的 API Key
            agent_config: CursorAgentConfig 实例（可选）
            auth_config: CloudAuthConfig 实例（可选）
            auth_timeout: 认证超时时间（可选）

        Returns:
            配置好的 CloudAuthManager 实例
        """
        unified_config = CloudClientFactory.create_auth_config(
            explicit_api_key=explicit_api_key,
            agent_config=agent_config,
            auth_config=auth_config,
            auth_timeout=auth_timeout,
        )
        return CloudAuthManager(config=unified_config)

    @staticmethod
    def create(
        explicit_api_key: Optional[str] = None,
        agent_config: Optional[Any] = None,
        auth_config: Optional[CloudAuthConfig] = None,
        auth_timeout: Optional[int] = None,
        api_base: Optional[str] = None,
        agents_endpoint: str = "/v1/agents",
    ) -> tuple[CursorCloudClient, CloudAuthManager]:
        """创建 CursorCloudClient 和 CloudAuthManager

        统一的工厂方法，确保认证配置来源优先级一致。
        从 config.yaml 获取默认的 api_base/auth_timeout/max_retries 等配置。

        配置来源优先级（从高到低）：
        1. 显式参数（explicit_api_key, auth_timeout, api_base）
        2. agent_config 中的对应值
        3. auth_config 中的对应值
        4. 环境变量 CURSOR_API_KEY / CURSOR_CLOUD_API_KEY
        5. config.yaml 中的 cloud_agent 配置
        6. 代码默认值

        Args:
            explicit_api_key: 显式传入的 API Key
            agent_config: CursorAgentConfig 实例（可选）
            auth_config: CloudAuthConfig 实例（可选）
            auth_timeout: 认证超时时间（可选）
            api_base: API 基础 URL（可选，默认从 config.yaml 获取）
            agents_endpoint: Agents 端点

        Returns:
            (CursorCloudClient, CloudAuthManager) 元组
        """
        # 从 config.yaml 获取默认配置
        cloud_config = build_cloud_client_config()

        # 解析 api_base，遵循优先级
        resolved_api_base = api_base
        if resolved_api_base is None and auth_config:
            resolved_api_base = auth_config.api_base_url
        if resolved_api_base is None:
            resolved_api_base = cloud_config.get("base_url", "https://api.cursor.com")

        auth_manager = CloudClientFactory.create_auth_manager(
            explicit_api_key=explicit_api_key,
            agent_config=agent_config,
            auth_config=auth_config,
            auth_timeout=auth_timeout,
        )

        cloud_client = CursorCloudClient(
            api_base=resolved_api_base,
            agents_endpoint=agents_endpoint,
            auth_manager=auth_manager,
        )

        return cloud_client, auth_manager

    @staticmethod
    def build_task_options(
        agent_config: Optional[Any] = None,
        model: Optional[str] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
        allow_write: Optional[bool] = None,
    ) -> CloudTaskOptions:
        """构建 CloudTaskOptions

        从 agent_config 和显式参数构建统一的任务选项。
        显式参数优先级高于 agent_config。

        Args:
            agent_config: CursorAgentConfig 实例（可选）
            model: 模型名称（覆盖 agent_config.model）
            working_directory: 工作目录（覆盖 agent_config.working_directory）
            timeout: 超时时间（覆盖 agent_config.timeout）
            allow_write: 是否允许写入（覆盖 agent_config.force_write）

        Returns:
            配置好的 CloudTaskOptions 实例
        """
        # 从 agent_config 提取默认值
        default_model = None
        default_working_directory = "."
        default_timeout = 300
        default_allow_write = False

        if agent_config:
            default_model = getattr(agent_config, 'model', None)
            default_working_directory = getattr(agent_config, 'working_directory', ".")
            default_timeout = getattr(agent_config, 'timeout', 300)
            default_allow_write = getattr(agent_config, 'force_write', False)

        return CloudTaskOptions(
            model=model or default_model,
            working_directory=working_directory or default_working_directory,
            timeout=timeout or default_timeout,
            allow_write=allow_write if allow_write is not None else default_allow_write,
        )

    @staticmethod
    async def execute_task(
        prompt: str,
        agent_config: Optional[Any] = None,
        auth_config: Optional[CloudAuthConfig] = None,
        explicit_api_key: Optional[str] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
        allow_write: Optional[bool] = None,
        session_id: Optional[str] = None,
        wait: bool = True,
        force_cloud: bool = False,
    ) -> CloudAgentResult:
        """统一的 Cloud 任务执行入口

        封装完整的 Cloud 执行流程，确保两条执行路径
        （CursorAgentClient._execute_via_cloud 和 CloudAgentExecutor.execute）
        行为一致。

        配置来源优先级:
        1. 显式参数（explicit_api_key, timeout, allow_write 等）
        2. agent_config 中的对应值
        3. auth_config 中的对应值
        4. 环境变量 CURSOR_API_KEY

        Args:
            prompt: 任务 prompt
            agent_config: CursorAgentConfig 实例（可选）
            auth_config: CloudAuthConfig 实例（可选）
            explicit_api_key: 显式传入的 API Key（最高优先级）
            working_directory: 工作目录
            timeout: 超时时间（秒）
            allow_write: 是否允许写入（对应 --force 参数）
            session_id: 可选的会话 ID（用于恢复会话）
            wait: 是否等待任务完成（默认 True）
            force_cloud: 强制云端执行（自动为非云端请求添加 & 前缀，默认 False）

        Returns:
            CloudAgentResult 执行结果，包含 success/output/error/files_modified/session_id
        """
        # 创建客户端和认证管理器
        cloud_client, auth_manager = CloudClientFactory.create(
            explicit_api_key=explicit_api_key,
            agent_config=agent_config,
            auth_config=auth_config,
        )

        # 验证认证
        auth_status = await auth_manager.authenticate()
        if not auth_status.authenticated:
            error_msg = auth_status.error.user_friendly_message if auth_status.error else "未认证"
            logger.warning(f"Cloud 认证失败: {error_msg}")
            return CloudAgentResult(
                success=False,
                error=error_msg,
            )

        # 构建任务选项
        task_options = CloudClientFactory.build_task_options(
            agent_config=agent_config,
            working_directory=working_directory,
            timeout=timeout,
            allow_write=allow_write,
        )

        # 执行任务（透传 force_cloud 参数）
        return await cloud_client.execute(
            prompt=prompt,
            options=task_options,
            wait=wait,
            timeout=timeout or task_options.timeout,
            session_id=session_id,
            force_cloud=force_cloud,
        )

    @staticmethod
    async def resume_session(
        session_id: str,
        prompt: Optional[str] = None,
        agent_config: Optional[Any] = None,
        auth_config: Optional[CloudAuthConfig] = None,
        explicit_api_key: Optional[str] = None,
        local: bool = True,
    ) -> CloudAgentResult:
        """恢复云端会话

        统一的会话恢复入口，确保两条执行路径行为一致。

        Args:
            session_id: 会话 ID
            prompt: 可选的附加 prompt
            agent_config: CursorAgentConfig 实例（可选）
            auth_config: CloudAuthConfig 实例（可选）
            explicit_api_key: 显式传入的 API Key
            local: 是否在本地继续执行（True=本地，False=仅获取状态）

        Returns:
            CloudAgentResult 执行结果
        """
        # 创建客户端和认证管理器
        cloud_client, auth_manager = CloudClientFactory.create(
            explicit_api_key=explicit_api_key,
            agent_config=agent_config,
            auth_config=auth_config,
        )

        # 验证认证
        auth_status = await auth_manager.authenticate()
        if not auth_status.authenticated:
            error_msg = auth_status.error.user_friendly_message if auth_status.error else "未认证"
            logger.warning(f"Cloud 认证失败: {error_msg}")
            return CloudAgentResult(
                success=False,
                error=error_msg,
            )

        # 构建任务选项
        task_options = CloudClientFactory.build_task_options(agent_config=agent_config)

        # 恢复会话
        return await cloud_client.resume_from_cloud(
            task_id=session_id,
            local=local,
            prompt=prompt,
            options=task_options,
        )


__all__ = [
    # 异常类
    "CloudAgentError",
    "RateLimitError",
    "NetworkError",
    "TaskError",
    "AuthErrorCode",
    "AuthError",
    # 重试工具
    "RetryConfig",
    "with_retry",
    "retry_async",
    "handle_http_error",
    # 认证管理
    "CloudAuthManager",
    "CloudAuthConfig",
    "AuthToken",
    "AuthStatus",
    "get_api_key",
    "verify_auth",
    "verify_auth_sync",
    "require_auth",
    # 任务管理
    "TaskStatus",
    "TaskResult",
    "CloudTaskClient",
    # Cloud Agent
    "CursorCloudClient",
    "CloudTaskOptions",
    "CloudTask",
    "CloudAgentResult",
    # 工厂类（统一的 Cloud Client 构造和执行入口）
    "CloudClientFactory",
    # 流式事件
    "StreamEvent",
    "StreamEventType",
    "ToolCallInfo",
    "parse_stream_event",
    # 别名
    "CloudAgentClient",
    "CloudAgentConfig",
]
