"""Cursor Cloud 认证模块

管理 Cursor API 的认证状态，支持:
- 环境变量 CURSOR_API_KEY 读取 API Key
- 配置文件读取 API Key
- OAuth 流程支持（为 Web/移动端准备）
- Token 刷新机制
- 认证失败友好错误提示
"""
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger
from pydantic import BaseModel, Field

# 从 exceptions 模块导入异常类
from .exceptions import (
    AuthError,
    AuthErrorCode,
)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ========== 数据类 ==========

@dataclass
class AuthToken:
    """认证 Token"""
    access_token: str
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """检查 Token 是否已过期"""
        if self.expires_at is None:
            return False
        # 提前 5 分钟视为过期，预留刷新时间
        return datetime.now() >= self.expires_at - timedelta(minutes=5)

    @property
    def expires_in_seconds(self) -> Optional[int]:
        """返回 Token 剩余有效时间（秒）"""
        if self.expires_at is None:
            return None
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, int(remaining))

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthToken":
        """从字典创建"""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])
        elif data.get("expires_in"):
            expires_at = datetime.now() + timedelta(seconds=data["expires_in"])

        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
        )


@dataclass
class AuthStatus:
    """认证状态"""
    authenticated: bool = False
    user_id: Optional[str] = None
    email: Optional[str] = None
    plan: Optional[str] = None  # free, pro, business
    token: Optional[AuthToken] = None
    last_verified: Optional[datetime] = None
    error: Optional[AuthError] = None

    @property
    def needs_refresh(self) -> bool:
        """是否需要刷新 Token"""
        if not self.authenticated or not self.token:
            return False
        return self.token.is_expired

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "authenticated": self.authenticated,
            "user_id": self.user_id,
            "email": self.email,
            "plan": self.plan,
            "token": self.token.to_dict() if self.token else None,
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "error": str(self.error) if self.error else None,
        }


# ========== 配置类 ==========

class CloudAuthConfig(BaseModel):
    """Cloud 认证配置"""
    # API Key（优先从环境变量读取）
    api_key: Optional[str] = None

    # 配置文件路径
    config_file: str = "config.yaml"

    # 全局配置文件路径
    global_config_file: str = Field(
        default_factory=lambda: str(Path.home() / ".cursor" / "config.json")
    )

    # CLI 配置文件路径
    cli_config_file: str = Field(
        default_factory=lambda: str(Path.home() / ".cursor" / "cli-config.json")
    )

    # API 端点
    api_base_url: str = "https://api.cursor.com"
    auth_endpoint: str = "/v1/auth/status"
    refresh_endpoint: str = "/v1/auth/refresh"

    # OAuth 配置（可选）
    oauth_client_id: Optional[str] = None
    oauth_redirect_uri: str = "http://localhost:8765/callback"
    oauth_scope: str = "read write"

    # 缓存配置
    cache_token: bool = True
    token_cache_file: str = Field(
        default_factory=lambda: str(Path.home() / ".cursor" / "token_cache.json")
    )

    # 超时配置
    auth_timeout: int = 30  # 秒
    refresh_before_expiry: int = 300  # 提前 5 分钟刷新

    # 指数退避重试配置
    max_retries: int = 3           # 最大重试次数
    base_delay: float = 1.0        # 基础延迟时间（秒）
    max_delay: float = 60.0        # 最大延迟时间（秒）
    exponential_base: float = 2.0  # 指数基数
    retry_jitter: bool = True      # 是否添加随机抖动


# ========== 认证管理器 ==========

class CloudAuthManager:
    """Cloud 认证管理器

    管理 Cursor API 的认证状态，支持:
    - 从环境变量 CURSOR_API_KEY 读取 API Key
    - 从配置文件读取 API Key
    - OAuth 流程支持（为 Web/移动端准备）
    - Token 刷新机制
    - 认证状态缓存

    用法:
        auth_manager = CloudAuthManager()
        status = await auth_manager.authenticate()
        if status.authenticated:
            api_key = auth_manager.get_api_key()
    """

    def __init__(self, config: Optional[CloudAuthConfig] = None):
        self.config = config or CloudAuthConfig()
        self._status = AuthStatus()
        self._agent_path = self._find_agent_executable()
        self._token_refresh_lock: Optional[asyncio.Lock] = None
        self._on_auth_change_callbacks: list[Callable[[AuthStatus], None]] = []

    async def _get_token_refresh_lock(self) -> asyncio.Lock:
        """延迟创建锁，避免无事件循环时报错。"""
        if self._token_refresh_lock is None:
            self._token_refresh_lock = asyncio.Lock()
        return self._token_refresh_lock

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

    @property
    def status(self) -> AuthStatus:
        """当前认证状态"""
        return self._status

    @property
    def is_authenticated(self) -> bool:
        """是否已认证"""
        return self._status.authenticated

    def on_auth_change(self, callback: Callable[[AuthStatus], None]) -> None:
        """注册认证状态变化回调"""
        self._on_auth_change_callbacks.append(callback)

    def _notify_auth_change(self) -> None:
        """通知认证状态变化"""
        for callback in self._on_auth_change_callbacks:
            try:
                callback(self._status)
            except Exception as e:
                logger.warning(f"认证状态回调执行失败: {e}")

    def get_api_key(self) -> Optional[str]:
        """获取 API Key

        优先级（与 CloudClientFactory.resolve_api_key 保持一致）:
        1. 配置中直接设置的 api_key（显式参数传入）
        2. 环境变量 CURSOR_API_KEY
        3. 环境变量 CURSOR_CLOUD_API_KEY（备选）
        4. 项目配置文件 (config.yaml)
           - YAML 路径优先级: cloud_agent.api_key > agent_cli.api_key > auth.api_key
        5. 全局配置文件 (~/.cursor/config.json)
        6. CLI 配置文件 (~/.cursor/cli-config.json)

        注意: 此优先级与 CloudClientFactory.resolve_api_key() 保持一致，
        确保两条 Cloud 执行路径（CursorAgentClient._execute_via_cloud 和
        CloudAgentExecutor.execute）的认证行为统一。
        """
        # 1. 配置中直接设置（优先级最高）
        if self.config.api_key:
            logger.debug("从配置对象获取 API Key（显式传入）")
            return self.config.api_key

        # 2. 环境变量 CURSOR_API_KEY
        env_key = os.environ.get("CURSOR_API_KEY")
        if env_key:
            logger.debug("从环境变量 CURSOR_API_KEY 获取 API Key")
            return env_key

        # 3. 环境变量 CURSOR_CLOUD_API_KEY（备选）
        env_cloud_key = os.environ.get("CURSOR_CLOUD_API_KEY")
        if env_cloud_key:
            logger.debug("从环境变量 CURSOR_CLOUD_API_KEY 获取 API Key")
            return env_cloud_key

        # 4. 项目配置文件
        project_key = self._read_api_key_from_yaml(self.config.config_file)
        if project_key:
            logger.debug(f"从项目配置文件 {self.config.config_file} 获取 API Key")
            return project_key

        # 5. 全局配置文件
        global_key = self._read_api_key_from_json(self.config.global_config_file)
        if global_key:
            logger.debug("从全局配置文件获取 API Key")
            return global_key

        # 6. CLI 配置文件
        cli_key = self._read_api_key_from_json(self.config.cli_config_file)
        if cli_key:
            logger.debug("从 CLI 配置文件获取 API Key")
            return cli_key

        return None

    def _read_api_key_from_yaml(self, file_path: str) -> Optional[str]:
        """从 YAML 配置文件读取 API Key

        YAML 路径优先级（从高到低）：
        1. cloud_agent.api_key（推荐，与 CloudClientFactory.resolve_api_key 一致）
        2. agent_cli.api_key（旧路径，向后兼容）
        3. auth.api_key（旧路径，向后兼容）

        注意: 此优先级与 CloudClientFactory.resolve_api_key() 中对 config.yaml
        的读取保持一致，确保两条 Cloud 执行路径的认证行为统一。
        """
        if not YAML_AVAILABLE:
            return None

        try:
            path = Path(file_path)
            if not path.exists():
                return None

            with open(path) as f:
                config = yaml.safe_load(f)

            if not config:
                return None

            # 尝试多种路径，按优先级顺序
            # 1. cloud_agent.api_key（推荐路径，优先级最高）
            if "cloud_agent" in config:
                api_key = config["cloud_agent"].get("api_key")
                if api_key:
                    logger.debug("从 YAML 路径 cloud_agent.api_key 读取 API Key")
                    return api_key

            # 2. agent_cli.api_key（旧路径，向后兼容）
            if "agent_cli" in config:
                api_key = config["agent_cli"].get("api_key")
                if api_key:
                    logger.debug("从 YAML 路径 agent_cli.api_key 读取 API Key（旧路径）")
                    return api_key

            # 3. auth.api_key（旧路径，向后兼容）
            if "auth" in config:
                api_key = config["auth"].get("api_key")
                if api_key:
                    logger.debug("从 YAML 路径 auth.api_key 读取 API Key（旧路径）")
                    return api_key

            return None
        except Exception as e:
            logger.debug(f"读取 YAML 配置失败: {e}")
            return None

    def _read_api_key_from_json(self, file_path: str) -> Optional[str]:
        """从 JSON 配置文件读取 API Key"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None

            with open(path) as f:
                config = json.load(f)

            # 尝试多种路径
            if "api_key" in config:
                return config["api_key"]
            if "apiKey" in config:
                return config["apiKey"]
            if "auth" in config and "api_key" in config["auth"]:
                return config["auth"]["api_key"]

            return None
        except Exception as e:
            logger.debug(f"读取 JSON 配置失败: {e}")
            return None

    async def authenticate(self, force: bool = False) -> AuthStatus:
        """验证 API Key 有效性

        Args:
            force: 强制重新验证，忽略缓存

        Returns:
            认证状态
        """
        # 如果已认证且不强制刷新，检查是否需要刷新 Token
        if not force and self._status.authenticated:
            if self._status.needs_refresh:
                return await self.refresh_token()
            return self._status

        api_key = self.get_api_key()

        if not api_key:
            self._status = AuthStatus(
                authenticated=False,
                error=AuthError(
                    "未找到 API Key",
                    AuthErrorCode.CONFIG_NOT_FOUND,
                ),
            )
            logger.warning(self._status.error.user_friendly_message)
            return self._status

        # 使用 agent CLI 验证认证状态
        try:
            status = await self._verify_with_cli()
            if status.authenticated:
                # 创建 Token（API Key 模式下不会过期）
                status.token = AuthToken(
                    access_token=api_key,
                    token_type="Bearer",
                )
                status.last_verified = datetime.now()

                # 缓存 Token
                if self.config.cache_token:
                    self._save_token_cache(status)

            self._status = status
            self._notify_auth_change()
            return self._status

        except AuthError as e:
            # 保留原始的 AuthError，包括其 error code
            logger.error(f"认证失败: {e}")
            self._status = AuthStatus(
                authenticated=False,
                error=e,
            )
            return self._status
        except Exception as e:
            logger.error(f"认证失败: {e}")
            self._status = AuthStatus(
                authenticated=False,
                error=AuthError(str(e), AuthErrorCode.UNKNOWN),
            )
            return self._status

    async def _verify_with_cli(self) -> AuthStatus:
        """使用 agent CLI 验证认证状态"""
        try:
            env = os.environ.copy()
            api_key = self.get_api_key()
            if api_key:
                env["CURSOR_API_KEY"] = api_key

            process = await asyncio.create_subprocess_exec(
                self._agent_path, "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.auth_timeout,
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if process.returncode == 0:
                # 解析认证信息
                return self._parse_auth_output(output)
            else:
                # 认证失败
                error_code = self._detect_error_code(error_output or output)
                raise AuthError(
                    error_output or output or "认证失败",
                    error_code,
                )

        except FileNotFoundError:
            raise AuthError(
                "找不到 agent CLI，请先安装: curl https://cursor.com/install -fsS | bash",
                AuthErrorCode.CONFIG_NOT_FOUND,
            )
        except asyncio.TimeoutError:
            raise AuthError(
                "认证请求超时",
                AuthErrorCode.NETWORK_ERROR,
            )

    def _parse_auth_output(self, output: str) -> AuthStatus:
        """解析 agent status 输出"""
        status = AuthStatus(authenticated=True)

        # 尝试从输出中提取信息
        output_lower = output.lower()

        # 检查是否认证成功
        if "not authenticated" in output_lower or "not logged in" in output_lower:
            status.authenticated = False
            status.error = AuthError(
                "未登录",
                AuthErrorCode.INVALID_API_KEY,
            )
            return status

        # 提取邮箱
        for line in output.split("\n"):
            line = line.strip()
            if "email" in line.lower() or "@" in line:
                # 简单提取邮箱
                parts = line.split()
                for part in parts:
                    if "@" in part:
                        status.email = part.strip(":<>(),")
                        break

            # 提取用户 ID
            if "user" in line.lower() and "id" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    status.user_id = parts[-1].strip()

            # 提取计划
            if "plan" in line.lower():
                if "pro" in line.lower():
                    status.plan = "pro"
                elif "business" in line.lower():
                    status.plan = "business"
                else:
                    status.plan = "free"

        return status

    def _detect_error_code(self, error_output: str) -> AuthErrorCode:
        """从错误输出检测错误类型"""
        error_lower = error_output.lower()

        if "invalid" in error_lower and "key" in error_lower:
            return AuthErrorCode.INVALID_API_KEY
        if "expired" in error_lower:
            return AuthErrorCode.EXPIRED_TOKEN
        if "rate limit" in error_lower or "too many" in error_lower:
            return AuthErrorCode.RATE_LIMITED
        if "permission" in error_lower or "unauthorized" in error_lower:
            return AuthErrorCode.INSUFFICIENT_PERMISSIONS
        if "network" in error_lower or "connection" in error_lower:
            return AuthErrorCode.NETWORK_ERROR

        return AuthErrorCode.UNKNOWN

    async def refresh_token(self) -> AuthStatus:
        """刷新 Token

        对于使用 OAuth 的场景，刷新 access_token
        对于 API Key 场景，重新验证
        """
        lock = await self._get_token_refresh_lock()
        async with lock:
            logger.info("正在刷新认证 Token...")

            # 如果有 refresh_token，使用 OAuth 刷新
            if self._status.token and self._status.token.refresh_token:
                try:
                    return await self._oauth_refresh()
                except AuthError as e:
                    logger.warning(f"OAuth 刷新失败: {e}")
                    # 回退到重新认证

            # 否则重新验证 API Key
            return await self.authenticate(force=True)

    async def _oauth_refresh(self) -> AuthStatus:
        """使用 refresh_token 刷新 OAuth Token"""
        if not self._status.token or not self._status.token.refresh_token:
            raise AuthError(
                "没有可用的 refresh_token",
                AuthErrorCode.REFRESH_FAILED,
            )

        # OAuth 刷新逻辑（预留接口）
        # 实际实现需要调用 OAuth 刷新端点
        raise AuthError(
            "OAuth 刷新尚未实现",
            AuthErrorCode.REFRESH_FAILED,
        )

    def _save_token_cache(self, status: AuthStatus) -> None:
        """保存 Token 缓存"""
        try:
            cache_path = Path(self.config.token_cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_path, "w") as f:
                json.dump(status.to_dict(), f, indent=2)

            # 设置文件权限为仅用户可读
            os.chmod(cache_path, 0o600)

        except Exception as e:
            logger.debug(f"保存 Token 缓存失败: {e}")

    def _load_token_cache(self) -> Optional[AuthStatus]:
        """加载 Token 缓存"""
        try:
            cache_path = Path(self.config.token_cache_file)
            if not cache_path.exists():
                return None

            with open(cache_path) as f:
                data = json.load(f)

            status = AuthStatus(
                authenticated=data.get("authenticated", False),
                user_id=data.get("user_id"),
                email=data.get("email"),
                plan=data.get("plan"),
            )

            if data.get("token"):
                status.token = AuthToken.from_dict(data["token"])

            if data.get("last_verified"):
                status.last_verified = datetime.fromisoformat(data["last_verified"])

            return status

        except Exception as e:
            logger.debug(f"加载 Token 缓存失败: {e}")
            return None

    def clear_cache(self) -> None:
        """清除 Token 缓存"""
        try:
            cache_path = Path(self.config.token_cache_file)
            if cache_path.exists():
                cache_path.unlink()
                logger.info("Token 缓存已清除")
        except Exception as e:
            logger.warning(f"清除缓存失败: {e}")

    async def logout(self) -> bool:
        """登出"""
        try:
            # 调用 agent logout
            process = await asyncio.create_subprocess_exec(
                self._agent_path, "logout",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.auth_timeout,
            )

            # 清除状态和缓存
            self._status = AuthStatus()
            self.clear_cache()
            self._notify_auth_change()

            logger.info("已登出")
            return True

        except Exception as e:
            logger.error(f"登出失败: {e}")
            return False

    # ========== OAuth 流程支持 ==========

    async def start_oauth_flow(
        self,
        on_url_ready: Optional[Callable[[str], None]] = None,
    ) -> AuthStatus:
        """启动 OAuth 认证流程（为 Web/移动端准备）

        Args:
            on_url_ready: 当授权 URL 准备好时的回调，用于在浏览器中打开

        Returns:
            认证状态
        """
        if not self.config.oauth_client_id:
            raise AuthError(
                "未配置 OAuth client_id",
                AuthErrorCode.OAUTH_FAILED,
            )

        # 使用 agent login 启动 OAuth 流程
        try:
            process = await asyncio.create_subprocess_exec(
                self._agent_path, "login",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # 读取输出，查找授权 URL
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120,  # OAuth 流程可能需要更长时间
            )

            output = stdout.decode("utf-8", errors="replace")

            if process.returncode == 0:
                # 登录成功，验证状态
                return await self.authenticate(force=True)
            else:
                raise AuthError(
                    stderr.decode("utf-8", errors="replace") or "OAuth 认证失败",
                    AuthErrorCode.OAUTH_FAILED,
                )

        except asyncio.TimeoutError:
            raise AuthError(
                "OAuth 认证超时",
                AuthErrorCode.OAUTH_FAILED,
            )

    def get_authorization_url(self) -> str:
        """生成 OAuth 授权 URL（用于自定义 OAuth 流程）"""
        if not self.config.oauth_client_id:
            raise AuthError(
                "未配置 OAuth client_id",
                AuthErrorCode.OAUTH_FAILED,
            )

        import secrets
        state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.config.oauth_client_id,
            "redirect_uri": self.config.oauth_redirect_uri,
            "response_type": "code",
            "scope": self.config.oauth_scope,
            "state": state,
        }

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.config.api_base_url}/oauth/authorize?{query}"

    async def exchange_code_for_token(self, code: str) -> AuthToken:
        """用授权码换取 Token（OAuth 流程）"""
        # 预留接口，实际实现需要调用 OAuth Token 端点
        raise AuthError(
            "OAuth token 交换尚未实现",
            AuthErrorCode.OAUTH_FAILED,
        )

    # ========== 同步版本 ==========

    def authenticate_sync(self, force: bool = False) -> AuthStatus:
        """同步版本的认证验证"""
        return asyncio.get_event_loop().run_until_complete(
            self.authenticate(force=force)
        )

    def refresh_token_sync(self) -> AuthStatus:
        """同步版本的 Token 刷新"""
        return asyncio.get_event_loop().run_until_complete(
            self.refresh_token()
        )

    # ========== 上下文管理器 ==========

    async def __aenter__(self) -> "CloudAuthManager":
        """异步上下文管理器入口"""
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器退出"""
        pass


# ========== 便捷函数 ==========

def get_api_key() -> Optional[str]:
    """快速获取 API Key"""
    manager = CloudAuthManager()
    return manager.get_api_key()


async def verify_auth() -> AuthStatus:
    """快速验证认证状态"""
    manager = CloudAuthManager()
    return await manager.authenticate()


def verify_auth_sync() -> AuthStatus:
    """同步版本：快速验证认证状态"""
    manager = CloudAuthManager()
    return manager.authenticate_sync()


# ========== 装饰器 ==========

def require_auth(func: Callable) -> Callable:
    """装饰器：要求认证

    用法:
        @require_auth
        async def my_api_call():
            ...
    """
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        manager = CloudAuthManager()
        status = await manager.authenticate()

        if not status.authenticated:
            raise status.error or AuthError(
                "需要认证",
                AuthErrorCode.INVALID_API_KEY,
            )

        return await func(*args, **kwargs)

    return wrapper


# ========== 导出 ==========

__all__ = [
    # 数据类
    "AuthToken",
    "AuthStatus",
    # 配置类
    "CloudAuthConfig",
    # 认证管理器
    "CloudAuthManager",
    # 便捷函数
    "get_api_key",
    "verify_auth",
    "verify_auth_sync",
    "require_auth",
]
