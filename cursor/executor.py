"""Agent Executor 抽象层

提供统一的 Agent 执行接口，支持多种执行模式:
- cli: 本地 Cursor CLI 执行
- cloud: 强制 Cloud API 执行
- auto: Cloud 优先，失败时按错误类型冷却后回退到 CLI

================================================================================
职责边界定义
================================================================================

本模块 (cursor/executor.py) 是执行编排层，职责如下：

1. 【编排】选择执行器、管理冷却状态、执行回退策略
2. 【日志】可使用 logger 打印技术性日志（DEBUG/INFO 级别）
3. 【元数据】将冷却信息、错误分类附加到 AgentResult.cooldown_info
4. 【不打印用户提示】用户可见的多行提示由入口脚本打印

与其他层的职责划分：

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  层级                │ 职责                     │ 打印行为              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  core/execution_    │ 决策、分类、构建消息字符串  │ 只返回元数据，不打印     │
    │  policy.py          │                          │                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  cursor/executor.py │ 编排执行流程、管理冷却状态   │ logger.info/debug      │
    │  (本模块)           │                          │ 技术性日志可打印         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  cursor/client.py   │ 与 CLI 进程/Cloud API 交互 │ 【需改造】避免多行提示   │
    │                     │                          │ 见下方"改造点"说明       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  入口脚本           │ 解析参数、打印用户可见消息  │ 打印 cooldown_info 中    │
    │  (run.py 等)        │                          │ 的 user_message         │
    └─────────────────────────────────────────────────────────────────────────┘

示例 - 执行器层的正确行为：

    # ✓ 使用 logger 打印技术性日志
    logger.info(f"Cloud 执行失败，错误类型: {error_type}，冷却 {remaining}s")

    # ✓ 使用权威函数构建 cooldown_info，由入口脚本决定是否打印
    from core.execution_policy import build_cooldown_info, classify_cloud_failure
    failure_info = classify_cloud_failure(error)
    result.cooldown_info = build_cooldown_info(
        failure_info=failure_info,
        fallback_reason=failure_info.message,
        requested_mode="auto",
        has_ampersand_prefix=has_ampersand_prefix,
        mode_source=mode_source,  # "cli" 或 "config"
    )

    # ✗ 不应直接 print() 多行用户提示
    # ✗ 不应手动构建 cooldown_info dict（必须使用 build_cooldown_info）

================================================================================
三类失败及冷却策略（与 core/execution_policy.py 对齐）
================================================================================

本模块使用 classify_cloud_failure() 分类错误，并根据类型应用冷却策略：

【类型 1】配置/认证失败 (NO_KEY, AUTH, CLOUD_DISABLED)
    - 冷却: auth_cooldown_seconds=600 (10分钟)
    - 恢复: 需要配置变化 (config_hash 检测)
    - 示例: API Key 无效 → 进入 10 分钟冷却，修改 config 后提前结束

【类型 2】可重试失败 (RATE_LIMIT, TIMEOUT, NETWORK, SERVICE)
    - 冷却: 按类型自适应
        - RATE_LIMIT: retry_after 夹逼 [30s, 300s]
        - TIMEOUT: 60s
        - NETWORK: 120s
        - SERVICE: 30s
    - 恢复: 冷却期结束后自动重试 Cloud
    - 示例: 429 Rate Limit → 使用 retry_after=60s 冷却 → 回退 CLI

【类型 3】资源/配额失败 (QUOTA, UNKNOWN)
    - 冷却: unknown_cooldown_seconds=300 (5分钟)
    - 恢复: 需要用户检查账户或联系支持
    - 示例: 配额耗尽 → 提示用户检查账户

================================================================================
cursor/client.py 改造状态（已完成）
================================================================================

以下违反职责边界的点已完成改造：

1. 【第 496-499 行】API Key 缺失警告 ✓ 已修复
   原: logger.warning("检测到 Cloud 请求 (& 前缀) 但未配置 API Key...")
   现: logger.debug("检测到 Cloud 请求 (& 前缀) 但未配置 API Key...")
   改造: 降级为 DEBUG 日志，用户提示通过 CursorAgentResult.failure_kind 结构化返回

2. 【第 657-678 行】回退消息打印 ✓ 已修复
   原: logger.warning(fallback_msg)
   现: logger.info(f"Cloud 执行失败，回退到 CLI: {failure_info.message}")
   改造: 降级为 INFO 日志（技术性），用户友好消息通过 cooldown_info[CooldownInfoFields.USER_MESSAGE] 返回

3. 【第 974-975 行】安装说明打印 ✓ 已修复
   原: logger.info("请先安装: curl https://cursor.com/install -fsS | bash")
   现: logger.debug("agent CLI 未安装，请执行: ...")
   改造: 降级为 DEBUG 日志，安装指引由入口脚本根据错误类型决定是否显示

改造原则（已遵循）:
- 库层 (client.py) 只做技术日志 (DEBUG/INFO 级别)，返回结构化错误信息
- 用户可见消息通过 cooldown_info[CooldownInfoFields.USER_MESSAGE] 传递给入口脚本
- 入口脚本根据 --quiet / --verbose 等参数决定打印策略

防回归测试:
- tests/test_run.py::TestPrintWarningMockVerification::test_library_layer_no_user_message_print
- tests/test_run.py::TestCloudFallbackUserMessageDedup
- tests/test_self_iterate.py::TestSelfIterateCloudFallbackUserMessageDedup

================================================================================
Cloud/Auto 语义统一说明
================================================================================

- cloud_enabled: 控制 '&' 前缀的自动检测，False 时 & 前缀视为普通字符
- execution_mode=auto: Cloud 优先，按错误类型自适应冷却后回退到 CLI
  冷却策略: RateLimitError 使用 retry_after, AuthError 需配置变化, 其他中等冷却
- force_write: 独立于 auto_commit，由用户显式控制 (--force)
- auto_commit: 需显式开启 (--auto-commit)，不影响 force_write 语义

配置 API Key 的三种方式:
  1. export CURSOR_API_KEY=your_key
  2. --cloud-api-key your_key
  3. agent login

用法:
    from cursor.executor import AgentExecutorFactory, ExecutionMode

    executor = AgentExecutorFactory.create(mode=ExecutionMode.AUTO)
    result = await executor.execute(prompt="分析代码结构")

    # 入口脚本打印用户消息（优先使用稳定字段 kind/reason/user_message）
    if result.cooldown_info and result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE):
        print(result.cooldown_info[CooldownInfoFields.USER_MESSAGE])

配置优先级: 显式参数 > agent_config > auth_config > CURSOR_API_KEY > config.yaml
"""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, Field

from core.cloud_utils import strip_cloud_prefix as _strip_cloud_prefix_util
from core.config import build_cloud_client_config, build_cooldown_config
from core.execution_policy import (
    CloudFailureInfo,
    CloudFailureKind,
    build_cooldown_info,
    build_cooldown_info_from_metadata,
    classify_cloud_failure,
)
from core.output_contract import CooldownInfoFields
from cursor.client import CursorAgentClient, CursorAgentConfig, CursorAgentResult

# 导入 Cloud Client 相关类
from cursor.cloud.client import CursorCloudClient
from cursor.cloud.task import CloudTaskOptions
from cursor.cloud_client import (
    AuthStatus,
    CloudAuthConfig,
    CloudAuthManager,
    CloudClientFactory,
)

# ========== Cloud 执行策略 ==========


# [已废弃] CloudErrorType 已被 CloudFailureKind 替代
# 保留此类以提供向后兼容，新代码请使用 CloudFailureKind
class CloudErrorType(str, Enum):
    """[已废弃] Cloud 错误类型分类

    请使用 core.execution_policy.CloudFailureKind 替代。
    此枚举保留用于向后兼容。
    """

    RATE_LIMIT = "rate_limit"  # 速率限制错误
    AUTH = "auth"  # 认证错误
    NETWORK = "network"  # 网络错误
    TIMEOUT = "timeout"  # 超时错误
    UNKNOWN = "unknown"  # 未知错误
    # 新增类型（映射到 CloudFailureKind）
    NO_KEY = "no_key"  # 未配置 API Key


# CloudFailureKind -> CloudErrorType 映射（向后兼容）
_FAILURE_KIND_TO_ERROR_TYPE: dict[CloudFailureKind, CloudErrorType] = {
    CloudFailureKind.NO_KEY: CloudErrorType.NO_KEY,
    CloudFailureKind.AUTH: CloudErrorType.AUTH,
    CloudFailureKind.RATE_LIMIT: CloudErrorType.RATE_LIMIT,
    CloudFailureKind.TIMEOUT: CloudErrorType.TIMEOUT,
    CloudFailureKind.NETWORK: CloudErrorType.NETWORK,
    CloudFailureKind.SERVICE: CloudErrorType.NETWORK,  # SERVICE 映射到 NETWORK
    CloudFailureKind.QUOTA: CloudErrorType.AUTH,  # QUOTA 映射到 AUTH
    CloudFailureKind.CLOUD_DISABLED: CloudErrorType.AUTH,  # CLOUD_DISABLED 映射到 AUTH
    CloudFailureKind.UNKNOWN: CloudErrorType.UNKNOWN,
}


def _failure_kind_to_error_type(kind: CloudFailureKind) -> CloudErrorType:
    """将 CloudFailureKind 转换为 CloudErrorType（向后兼容）"""
    return _FAILURE_KIND_TO_ERROR_TYPE.get(kind, CloudErrorType.UNKNOWN)


@dataclass
class CooldownConfig:
    """冷却配置

    按错误类型定义不同的冷却策略。
    默认值与 core/config.py 中的 DEFAULT_COOLDOWN_* 常量保持同步。

    使用 from_config() 工厂方法从 config.yaml 加载配置。
    """

    # RateLimitError: 使用 retry_after 或默认值，最小/最大夹逼
    rate_limit_min_seconds: int = 30
    rate_limit_default_seconds: int = 60
    rate_limit_max_seconds: int = 300

    # AuthError: 更长冷却，直到配置变化才重试
    auth_cooldown_seconds: int = 600  # 10 分钟
    auth_require_config_change: bool = True  # 需要配置变化才能重试

    # Network/Timeout: 中等冷却
    network_cooldown_seconds: int = 120  # 2 分钟
    timeout_cooldown_seconds: int = 60  # 1 分钟

    # 未知错误: 默认冷却
    unknown_cooldown_seconds: int = 300  # 5 分钟

    @classmethod
    def from_config(
        cls,
        overrides: Optional[dict[str, Any]] = None,
    ) -> "CooldownConfig":
        """从 config.yaml 加载冷却配置

        使用 core.config.build_cooldown_config() 读取 config.yaml 中的
        cloud_agent.cooldown 配置，并应用覆盖项。

        优先级: overrides (CLI) > config.yaml > DEFAULT_COOLDOWN_* 常量

        Args:
            overrides: CLI 或调用方传入的覆盖配置

        Returns:
            CooldownConfig 实例

        示例:
            >>> # 使用 config.yaml 默认配置
            >>> config = CooldownConfig.from_config()

            >>> # 覆盖特定配置
            >>> config = CooldownConfig.from_config(
            ...     overrides={"auth_cooldown_seconds": 1200}
            ... )
        """
        config_dict = build_cooldown_config(overrides=overrides)
        return cls(**config_dict)


@dataclass
class CooldownState:
    """冷却状态

    跟踪当前的冷却状态和错误信息。
    使用 failure_kind (CloudFailureKind) 替代已废弃的 error_type。
    """

    failure_kind: Optional[CloudFailureKind] = None  # 使用 CloudFailureKind 替代 CloudErrorType
    cooldown_until: Optional[datetime] = None
    failure_count: int = 0
    last_error_message: Optional[str] = None
    retry_after_hint: Optional[int] = None  # RateLimitError 的 retry_after 提示
    config_hash: Optional[str] = None  # 用于检测配置变化（AuthError/NO_KEY）

    # ========== 向后兼容属性 ==========
    @property
    def error_type(self) -> Optional[CloudErrorType]:
        """[已废弃] 向后兼容：获取错误类型

        请使用 failure_kind 替代。
        """
        if self.failure_kind is None:
            return None
        return _failure_kind_to_error_type(self.failure_kind)

    @error_type.setter
    def error_type(self, value: Optional[CloudErrorType]) -> None:
        """[已废弃] 向后兼容：设置错误类型"""
        if value is None:
            self.failure_kind = None
        else:
            # 反向映射 CloudErrorType -> CloudFailureKind
            error_type_to_kind = {
                CloudErrorType.NO_KEY: CloudFailureKind.NO_KEY,
                CloudErrorType.AUTH: CloudFailureKind.AUTH,
                CloudErrorType.RATE_LIMIT: CloudFailureKind.RATE_LIMIT,
                CloudErrorType.TIMEOUT: CloudFailureKind.TIMEOUT,
                CloudErrorType.NETWORK: CloudFailureKind.NETWORK,
                CloudErrorType.UNKNOWN: CloudFailureKind.UNKNOWN,
            }
            self.failure_kind = error_type_to_kind.get(value, CloudFailureKind.UNKNOWN)

    def is_in_cooldown(self) -> bool:
        """检查是否处于冷却期"""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until

    def get_remaining_seconds(self) -> Optional[float]:
        """获取剩余冷却秒数"""
        if not self.is_in_cooldown():
            return None
        assert self.cooldown_until is not None
        remaining = (self.cooldown_until - datetime.now()).total_seconds()
        return max(0.0, remaining)

    def reset(self) -> None:
        """重置冷却状态"""
        self.failure_kind = None
        self.cooldown_until = None
        self.failure_count = 0
        self.last_error_message = None
        self.retry_after_hint = None


@dataclass
class CooldownMetadata:
    """冷却元数据

    供上层打印提示使用。
    使用 failure_kind (CloudFailureKind) 替代已废弃的 error_type。
    """

    in_cooldown: bool = False
    failure_kind: Optional[CloudFailureKind] = None  # 使用 CloudFailureKind 替代 CloudErrorType
    remaining_seconds: Optional[float] = None
    failure_count: int = 0
    last_error_message: Optional[str] = None
    can_retry_with_config_change: bool = False  # AuthError/NO_KEY 时配置变化可重试

    # ========== 向后兼容属性 ==========
    @property
    def error_type(self) -> Optional[CloudErrorType]:
        """[已废弃] 向后兼容：获取错误类型

        请使用 failure_kind 替代。
        """
        if self.failure_kind is None:
            return None
        return _failure_kind_to_error_type(self.failure_kind)

    @property
    def kind(self) -> Optional[str]:
        """获取错误类型字符串（用于 cooldown_info 输出）

        返回 CloudFailureKind 的值，如 "no_key", "auth", "rate_limit" 等。
        """
        if self.failure_kind is None:
            return None
        return self.failure_kind.value


class CloudExecutionPolicy:
    """Cloud 执行策略

    负责：
    1. 判断是否应该尝试 Cloud（基于 cloud_enabled 和冷却状态）
    2. 为 CLI 回退清理 prompt（剥离 & 前缀）
    3. 管理按错误类型自适应的冷却逻辑
    """

    def __init__(self, config: CooldownConfig | None = None):
        """初始化策略

        Args:
            config: 冷却配置，默认使用 CooldownConfig 默认值
        """
        self._config = config or CooldownConfig()
        self._state = CooldownState()

    @property
    def cooldown_config(self) -> CooldownConfig:
        return self._config

    @property
    def cooldown_state(self) -> CooldownState:
        return self._state

    def should_try_cloud(
        self,
        cloud_enabled: bool,
        current_config_hash: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """判断是否应该尝试 Cloud

        Args:
            cloud_enabled: 配置中 cloud_enabled 是否为 True
            current_config_hash: 当前配置的哈希值（用于检测 AuthError/NO_KEY 配置变化）

        Returns:
            (should_try, reason): 是否应该尝试，以及不尝试的原因
        """
        # 1. cloud_enabled 为 False 时直接返回 False
        if not cloud_enabled:
            return False, "cloud_enabled=False"

        # 2. 检查冷却状态
        if self._state.is_in_cooldown():
            remaining = self._state.get_remaining_seconds()
            failure_kind = self._state.failure_kind

            # AuthError/NO_KEY 特殊处理：如果配置变化，可以提前结束冷却
            if (
                failure_kind in (CloudFailureKind.AUTH, CloudFailureKind.NO_KEY)
                and self._config.auth_require_config_change
                and current_config_hash is not None
                and self._state.config_hash is not None
                and current_config_hash != self._state.config_hash
            ):
                kind_name = failure_kind.value if failure_kind else "unknown"
                logger.info(f"检测到配置变化，提前结束 {kind_name} 冷却期")
                self._state.reset()
                return True, None

            kind_value = failure_kind.value if failure_kind else "unknown"
            reason = f"Cloud 处于 {kind_value} 冷却期，剩余 {remaining:.1f} 秒"
            return False, reason

        return True, None

    @staticmethod
    def sanitize_prompt_for_cli_fallback(prompt: str) -> str:
        """为 CLI 回退清理 prompt

        剥离 & 前缀，确保回退到 CLI 时不会再次触发 Cloud 路由。

        Args:
            prompt: 原始 prompt（可能带 & 前缀）

        Returns:
            清理后的 prompt
        """
        return _strip_cloud_prefix_util(prompt)

    def classify_error(self, error: Exception) -> CloudErrorType:
        """[已废弃] 将异常分类为错误类型

        .. deprecated::
            此方法已废弃。请使用 `core.execution_policy.classify_cloud_failure()`
            获取 `CloudFailureInfo`，然后通过 `AutoAgentExecutor._map_cloud_failure_to_error_type()`
            映射到 `CloudErrorType`。

            新代码示例::

                from core.execution_policy import classify_cloud_failure
                failure_info = classify_cloud_failure(error)
                # failure_info.kind 包含分类结果
                # failure_info.retry_after 包含重试建议

        Args:
            error: 捕获的异常

        Returns:
            错误类型枚举
        """
        # 使用 classify_cloud_failure 进行分类，然后映射到 CloudErrorType
        failure_info = classify_cloud_failure(error)

        # 映射 CloudFailureKind 到 CloudErrorType
        kind_to_error_type = {
            CloudFailureKind.AUTH: CloudErrorType.AUTH,
            CloudFailureKind.NO_KEY: CloudErrorType.AUTH,
            CloudFailureKind.RATE_LIMIT: CloudErrorType.RATE_LIMIT,
            CloudFailureKind.TIMEOUT: CloudErrorType.TIMEOUT,
            CloudFailureKind.NETWORK: CloudErrorType.NETWORK,
            CloudFailureKind.SERVICE: CloudErrorType.NETWORK,
            CloudFailureKind.QUOTA: CloudErrorType.AUTH,
            CloudFailureKind.UNKNOWN: CloudErrorType.UNKNOWN,
        }
        return kind_to_error_type.get(failure_info.kind, CloudErrorType.UNKNOWN)

    def start_cooldown(
        self,
        error: Exception,
        config_hash: Optional[str] = None,
    ) -> CooldownMetadata:
        """[已废弃] 开始冷却期

        .. deprecated::
            此方法已废弃。请使用 `AutoAgentExecutor._start_cooldown_from_failure_info()`
            结合 `core.execution_policy.classify_cloud_failure()` 获取更精确的错误分类。

            新代码示例::

                from core.execution_policy import classify_cloud_failure
                failure_info = classify_cloud_failure(error_or_result)
                cooldown_meta = self._start_cooldown_from_failure_info(
                    failure_info=failure_info,
                    config_hash=config_hash,
                )

        根据错误类型设置不同的冷却时间。

        Args:
            error: 导致冷却的异常
            config_hash: 当前配置哈希（用于 AuthError/NO_KEY）

        Returns:
            冷却元数据，供上层使用
        """
        # 使用 classify_cloud_failure 进行分类
        failure_info = classify_cloud_failure(error)

        # 直接使用 failure_kind（CloudFailureKind）
        self._state.failure_kind = failure_info.kind
        self._state.failure_count += 1
        self._state.last_error_message = failure_info.message
        self._state.config_hash = config_hash

        # 使用 failure_info.retry_after 计算冷却时间
        cooldown_seconds = self._calculate_cooldown_seconds_from_kind(failure_info)

        self._state.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)

        # 记录日志
        logger.warning(
            f"Cloud 执行失败（第 {self._state.failure_count} 次），"
            f"错误类型: {failure_info.kind.value}，"
            f"进入冷却期 {cooldown_seconds} 秒"
        )

        return self.get_cooldown_metadata()

    def _calculate_cooldown_seconds(
        self,
        error: Exception,
        error_type: CloudErrorType,
    ) -> int:
        """[已废弃] 计算冷却秒数

        .. deprecated::
            此方法已废弃。请使用 `_calculate_cooldown_seconds_from_failure_info()`
            结合 `CloudFailureInfo` 获取更精确的冷却时间。

        Args:
            error: 异常实例
            error_type: 错误类型

        Returns:
            冷却秒数
        """
        # 使用 classify_cloud_failure 进行分类，然后调用新方法
        failure_info = classify_cloud_failure(error)
        return self._calculate_cooldown_seconds_from_failure_info(failure_info, error_type)

    def _calculate_cooldown_seconds_from_failure_info(
        self,
        failure_info: CloudFailureInfo,
        error_type: CloudErrorType,
    ) -> int:
        """[已废弃] 根据 CloudFailureInfo 计算冷却秒数

        .. deprecated::
            此方法已废弃。请使用 `_calculate_cooldown_seconds_from_kind()` 替代。

        使用 failure_info.retry_after 作为速率限制的冷却时间依据。

        Args:
            failure_info: Cloud 错误分类信息
            error_type: 错误类型（已废弃，实际使用 failure_info.kind）

        Returns:
            冷却秒数
        """
        # 直接委托给新方法
        return self._calculate_cooldown_seconds_from_kind(failure_info)

    def _calculate_cooldown_seconds_from_kind(
        self,
        failure_info: CloudFailureInfo,
    ) -> int:
        """根据 CloudFailureInfo.kind 计算冷却秒数

        使用 failure_info.retry_after 作为速率限制的冷却时间依据。
        支持 NO_KEY 作为独立的错误类型处理。

        Args:
            failure_info: Cloud 错误分类信息

        Returns:
            冷却秒数
        """
        config = self._config
        kind = failure_info.kind

        if kind == CloudFailureKind.RATE_LIMIT:
            # 使用 failure_info.retry_after（如果有），否则使用默认值
            retry_after = failure_info.retry_after

            if retry_after is not None:
                self._state.retry_after_hint = retry_after
                # 最小/最大夹逼
                clamped = max(config.rate_limit_min_seconds, min(retry_after, config.rate_limit_max_seconds))
                logger.debug(f"RateLimitError retry_after={retry_after}s, 夹逼后={clamped}s")
                return clamped
            return config.rate_limit_default_seconds

        elif kind in (CloudFailureKind.AUTH, CloudFailureKind.NO_KEY, CloudFailureKind.QUOTA):
            # AUTH/NO_KEY/QUOTA 都使用 auth_cooldown_seconds
            # NO_KEY 需要配置 API Key 才能继续，与 AUTH 类似
            return config.auth_cooldown_seconds

        elif kind in (CloudFailureKind.NETWORK, CloudFailureKind.SERVICE):
            # NETWORK 和 SERVICE 错误使用相同的冷却时间
            return config.network_cooldown_seconds

        elif kind == CloudFailureKind.TIMEOUT:
            return config.timeout_cooldown_seconds

        else:
            # UNKNOWN, CLOUD_DISABLED 等
            return config.unknown_cooldown_seconds

    def get_cooldown_metadata(self) -> CooldownMetadata:
        """获取冷却元数据

        供上层打印提示使用。

        Returns:
            冷却元数据
        """
        in_cooldown = self._state.is_in_cooldown()

        # 直接使用 CooldownState.failure_kind
        failure_kind = self._state.failure_kind

        return CooldownMetadata(
            in_cooldown=in_cooldown,
            failure_kind=failure_kind,
            remaining_seconds=self._state.get_remaining_seconds() if in_cooldown else None,
            failure_count=self._state.failure_count,
            last_error_message=self._state.last_error_message,
            can_retry_with_config_change=(
                # AUTH 和 NO_KEY 都可以通过配置变化重试
                failure_kind in (CloudFailureKind.AUTH, CloudFailureKind.NO_KEY)
                and self._config.auth_require_config_change
            ),
        )

    def reset(self) -> None:
        """重置冷却状态（Cloud 成功时调用）"""
        if self._state.failure_count > 0:
            logger.debug("Cloud 执行成功，重置冷却状态")
        self._state.reset()

    def check_cooldown_expired(self) -> bool:
        """检查冷却期是否已过期

        如果过期则重置并返回 True。

        Returns:
            是否已过期
        """
        if self._state.cooldown_until is None:
            return True
        if datetime.now() >= self._state.cooldown_until:
            kind_value = self._state.failure_kind.value if self._state.failure_kind else "unknown"
            logger.info(f"Cloud {kind_value} 冷却期已结束，可重新尝试")
            # 只重置时间，保留错误计数用于指数退避
            self._state.cooldown_until = None
            return True
        return False


class ExecutionMode(str, Enum):
    """执行模式"""

    CLI = "cli"  # 本地 CLI 执行（完整 agent 模式）
    CLOUD = "cloud"  # Cloud API 执行
    AUTO = "auto"  # 自动选择（Cloud 优先，不可用时回退到 CLI）
    PLAN = "plan"  # 规划模式（只分析不执行，对应 --mode plan）
    ASK = "ask"  # 问答模式（仅回答问题，不修改文件，对应 --mode ask）


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

    # 结构化错误字段（从 CloudAgentResult 透传）
    error_type: Optional[str] = None  # auth, rate_limit, timeout, network, etc.
    retry_after: Optional[float] = None  # 建议重试等待时间（秒）

    # failure_kind: 使用 CloudFailureKind 枚举值表示的错误类型
    # 与 error_type 的区别:
    # - error_type: 原始结构化错误字符串（来自 CloudAgentResult）
    # - failure_kind: 经过 classify_cloud_failure() 分类后的标准化类型
    #   可选值: no_key, auth, rate_limit, timeout, network, quota, service, unknown, cloud_disabled
    failure_kind: Optional[str] = None

    # 原始结果（用于调试）
    raw_result: Optional[dict[str, Any]] = None

    # 冷却元数据（供上层打印提示）
    cooldown_info: Optional[dict[str, Any]] = None

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
        files_edited = getattr(result, "files_edited", []) or []
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
        error_type: Optional[str] = None,
        retry_after: Optional[float] = None,
        failure_kind: Optional[str] = None,
    ) -> "AgentResult":
        """从 Cloud API 结果转换

        透传 CloudAgentResult 中的结构化错误字段，以便上层判断错误类型:
        - error_type: 结构化错误类型 (auth, rate_limit, timeout, network, etc.)
        - retry_after: 建议重试等待时间（秒），主要用于限流错误
        - failure_kind: 经过 classify_cloud_failure() 分类后的标准化错误类型

        Args:
            success: 是否成功
            output: 输出内容
            error: 错误信息
            duration: 执行时长（秒）
            session_id: 会话 ID
            raw_result: 原始结果字典
            files_modified: 修改的文件列表
            error_type: 结构化错误类型 (auth, rate_limit, timeout, etc.)
            retry_after: 建议重试等待时间（秒）
            failure_kind: 分类后的错误类型 (no_key, auth, rate_limit, etc.)

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
            error_type=error_type,
            retry_after=retry_after,
            failure_kind=failure_kind,
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
    支持 --mode 参数指定执行模式（plan/ask）；agent 为默认模式，不传 --mode
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
        self._config = base_config.model_copy(
            update={
                "mode": "plan",
                "force_write": False,  # 规划模式不修改文件
            }
        )
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
        self._config = base_config.model_copy(
            update={
                "mode": "ask",
                "force_write": False,  # 问答模式不修改文件
            }
        )
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

    通过 Cursor Cloud API 执行任务，支持后台任务提交和会话恢复。

    Cloud/Auto 语义:
    - cloud_enabled 控制 '&' 前缀的自动检测
    - force_write 由 --force 控制，独立于 auto_commit
    - 失败时按错误类型冷却（RateLimitError/AuthError/Network/Timeout）

    配置 API Key 的三种方式:
      1. export CURSOR_API_KEY=your_key
      2. --cloud-api-key your_key
      3. agent login

    配置优先级: 显式参数 > agent_config > auth_config > CURSOR_API_KEY > config.yaml
    """

    def __init__(
        self,
        auth_config: Optional[CloudAuthConfig] = None,
        agent_config: Optional[CursorAgentConfig] = None,
        cloud_client: Optional[CursorCloudClient] = None,
    ):
        """初始化 Cloud 执行器

        配置优先级: agent_config > auth_config > CURSOR_API_KEY > config.yaml

        Args:
            auth_config: Cloud 认证配置
            agent_config: Agent 配置（模型、超时等）
            cloud_client: CursorCloudClient 实例（用于测试注入）
        """
        # 从 config.yaml 获取默认配置
        cloud_config = build_cloud_client_config()

        # 如果未提供 auth_config，则使用 config.yaml 中的配置创建
        if auth_config is None:
            self._auth_config = CloudAuthConfig(
                api_key=cloud_config.get("api_key"),
                api_base_url=cloud_config.get("base_url", "https://api.cursor.com"),
                auth_timeout=cloud_config.get("auth_timeout", 30),
                max_retries=cloud_config.get("max_retries", 3),
            )
        else:
            self._auth_config = auth_config

        self._agent_config = agent_config or CursorAgentConfig()
        self._available: Optional[bool] = None
        self._auth_status: Optional[AuthStatus] = None

        # 使用 CloudClientFactory 统一创建认证管理器和客户端
        # 配置来源优先级: agent_config.api_key > auth_config.api_key > 环境变量 > config.yaml
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

        Cloud/Auto 语义:
        - force_write 由 agent_config.force_write 控制，独立于 auto_commit
        - background=False: 前台模式，等待任务完成
        - background=True: 后台模式 (Cloud Relay)，立即返回 session_id

        配置优先级: 显式参数 > agent_config > auth_config > CURSOR_API_KEY

        Args:
            prompt: 任务提示（可带 & 前缀）
            context: 上下文信息
            working_directory: 工作目录
            timeout: 超时时间（秒）
            background: 后台模式（默认 False）
            session_id: 会话 ID（用于恢复会话）
            force_cloud: 强制云端执行（默认 True）

        Returns:
            background=False 时返回完整结果; background=True 时返回 session_id 用于后续恢复
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
                # 透传结构化错误字段（error_type, retry_after, failure_kind）
                # 使用 classify_cloud_failure 计算 failure_kind
                failure_kind_value = None
                if not cloud_result.success:
                    failure_info = classify_cloud_failure(
                        cloud_result.to_dict() if hasattr(cloud_result, "to_dict") else cloud_result.error
                    )
                    failure_kind_value = failure_info.kind.value
                return AgentResult.from_cloud_result(
                    success=cloud_result.success,
                    output=cloud_result.output,
                    error=cloud_result.error,
                    duration=duration,
                    session_id=cloud_result.task.task_id if cloud_result.task else None,
                    raw_result=cloud_result.to_dict(),
                    files_modified=cloud_result.files_modified,
                    error_type=cloud_result.error_type,
                    retry_after=cloud_result.retry_after,
                    failure_kind=failure_kind_value,
                )
            else:
                # 后台模式：仅返回任务元信息
                # success 表示任务提交是否成功，而非任务本身是否完成
                # output/files_modified 为空（任务尚未完成）
                # raw_result 包含 task 元信息，供调用方获取更多细节
                # 透传结构化错误字段（error_type, retry_after, failure_kind）
                task_id = cloud_result.task.task_id if cloud_result.task else None
                failure_kind_value = None
                if not cloud_result.success:
                    failure_info = classify_cloud_failure(
                        cloud_result.to_dict() if hasattr(cloud_result, "to_dict") else cloud_result.error
                    )
                    failure_kind_value = failure_info.kind.value
                return AgentResult.from_cloud_result(
                    success=cloud_result.success,
                    output=cloud_result.output or "",  # 后台模式可能只有初始响应
                    error=cloud_result.error,
                    duration=duration,
                    session_id=task_id,
                    raw_result=cloud_result.to_dict(),
                    files_modified=[],  # 后台模式任务尚未完成，无文件修改信息
                    error_type=cloud_result.error_type,
                    retry_after=cloud_result.retry_after,
                    failure_kind=failure_kind_value,
                )

        except asyncio.TimeoutError as e:
            failure_info = classify_cloud_failure(e)
            return AgentResult(
                success=False,
                error=f"Cloud API 执行超时 ({timeout_sec}s)",
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
                failure_kind=failure_info.kind.value,
            )
        except Exception as e:
            logger.error(f"Cloud API 执行异常: {e}")
            failure_info = classify_cloud_failure(e)
            return AgentResult(
                success=False,
                error=str(e),
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
                failure_kind=failure_info.kind.value,
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
                    failure_kind=CloudFailureKind.AUTH.value,
                )

            # 如果未提供选项，使用工厂构建默认选项
            if options is None:
                options = CloudClientFactory.build_task_options(
                    agent_config=self._agent_config,
                )

            # 提交任务（不等待完成）
            cloud_result = await self._cloud_client.submit_task(prompt, options)

            # 计算 failure_kind
            failure_kind_value = None
            if not cloud_result.success:
                failure_info = classify_cloud_failure(
                    cloud_result.to_dict() if hasattr(cloud_result, "to_dict") else cloud_result.error
                )
                failure_kind_value = failure_info.kind.value

            return AgentResult(
                success=cloud_result.success,
                output=cloud_result.output,
                error=cloud_result.error,
                executor_type="cloud",
                session_id=cloud_result.task.task_id if cloud_result.task else None,
                started_at=started_at,
                completed_at=datetime.now(),
                raw_result=cloud_result.to_dict(),
                failure_kind=failure_kind_value,
            )

        except Exception as e:
            failure_info = classify_cloud_failure(e)
            return AgentResult(
                success=False,
                error=str(e),
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
                failure_kind=failure_info.kind.value,
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

            # 计算 failure_kind
            failure_kind_value = None
            if not cloud_result.success:
                failure_info = classify_cloud_failure(
                    cloud_result.to_dict() if hasattr(cloud_result, "to_dict") else cloud_result.error
                )
                failure_kind_value = failure_info.kind.value

            return AgentResult.from_cloud_result(
                success=cloud_result.success,
                output=cloud_result.output,
                error=cloud_result.error,
                duration=duration,
                session_id=task_id,
                raw_result=cloud_result.to_dict(),
                error_type=cloud_result.error_type,
                retry_after=cloud_result.retry_after,
                failure_kind=failure_kind_value,
            )

        except Exception as e:
            failure_info = classify_cloud_failure(e)
            return AgentResult(
                success=False,
                error=str(e),
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
                failure_kind=failure_info.kind.value,
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

            # 计算 failure_kind
            failure_kind_value = None
            if not cloud_result.success:
                failure_info = classify_cloud_failure(
                    cloud_result.to_dict() if hasattr(cloud_result, "to_dict") else cloud_result.error
                )
                failure_kind_value = failure_info.kind.value

            return AgentResult.from_cloud_result(
                success=cloud_result.success,
                output=cloud_result.output,
                error=cloud_result.error,
                duration=duration,
                session_id=session_id,
                raw_result=cloud_result.to_dict(),
                files_modified=cloud_result.files_modified,
                error_type=cloud_result.error_type,
                retry_after=cloud_result.retry_after,
                failure_kind=failure_kind_value,
            )

        except Exception as e:
            # 计算 failure_kind
            failure_info = classify_cloud_failure(e)
            return AgentResult(
                success=False,
                error=str(e),
                executor_type="cloud",
                started_at=started_at,
                completed_at=datetime.now(),
                failure_kind=failure_info.kind.value,
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
    """自动选择执行器 (execution_mode=auto)

    Cloud 优先，失败时按错误类型自适应冷却后回退到 CLI。

    冷却策略:
    - RateLimitError: 使用 retry_after（30s~300s 夹逼）
    - AuthError: 10 分钟，配置变化可提前结束
    - Network/Timeout: 2 分钟/1 分钟
    - Unknown: 5 分钟

    Cloud/Auto 语义:
    - cloud_enabled 控制 '&' 前缀的自动检测
    - force_write 由 --force 控制，独立于 auto_commit
    - 回退时 sanitize prompt（剥离 & 前缀），避免再次触发 Cloud 路由
    """

    def __init__(
        self,
        cli_config: Optional[CursorAgentConfig] = None,
        cloud_auth_config: Optional[CloudAuthConfig] = None,
        cooldown_config: Optional[CooldownConfig] = None,
        enable_cooldown: bool = True,
        # 向后兼容参数（已废弃，请使用 cooldown_config）
        cloud_cooldown_seconds: Optional[int] = None,
    ):
        """初始化自动执行器

        Args:
            cli_config: CLI 执行器配置
            cloud_auth_config: Cloud 认证配置
            cooldown_config: 冷却配置，默认使用 CooldownConfig 默认值
            enable_cooldown: 是否启用冷却策略，默认启用
            cloud_cooldown_seconds: [已废弃] 请使用 cooldown_config
        """
        self._cli_config = cli_config
        self._cli_executor = CLIAgentExecutor(cli_config)
        self._cloud_executor = CloudAgentExecutor(
            auth_config=cloud_auth_config,
            agent_config=cli_config,
        )
        self._preferred_executor: Optional[AgentExecutor] = None

        # Cloud 执行策略
        self._enable_cooldown = enable_cooldown

        # 处理向后兼容：如果传入了 cloud_cooldown_seconds，转换为 CooldownConfig
        if cloud_cooldown_seconds is not None and cooldown_config is None:
            cooldown_config = CooldownConfig(
                rate_limit_default_seconds=cloud_cooldown_seconds,
                network_cooldown_seconds=cloud_cooldown_seconds,
                timeout_cooldown_seconds=cloud_cooldown_seconds,
                unknown_cooldown_seconds=cloud_cooldown_seconds,
            )
        elif cooldown_config is None:
            # 默认从 config.yaml 加载冷却配置
            cooldown_config = CooldownConfig.from_config()

        self._policy = CloudExecutionPolicy(config=cooldown_config)

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
    def policy(self) -> CloudExecutionPolicy:
        """获取 Cloud 执行策略"""
        return self._policy

    @property
    def is_cloud_in_cooldown(self) -> bool:
        """检查 Cloud 是否处于冷却期"""
        if not self._enable_cooldown:
            return False
        return self._policy.cooldown_state.is_in_cooldown()

    @property
    def cloud_cooldown_remaining(self) -> Optional[float]:
        """获取 Cloud 冷却剩余时间（秒），未在冷却期返回 None"""
        if not self.is_cloud_in_cooldown:
            return None
        return self._policy.cooldown_state.get_remaining_seconds()

    # ========== 向后兼容属性（已废弃） ==========

    @property
    def _cloud_cooldown_until(self) -> Optional[datetime]:
        """[已废弃] 向后兼容：获取冷却截止时间"""
        return self._policy.cooldown_state.cooldown_until

    @_cloud_cooldown_until.setter
    def _cloud_cooldown_until(self, value: Optional[datetime]) -> None:
        """[已废弃] 向后兼容：设置冷却截止时间"""
        self._policy.cooldown_state.cooldown_until = value

    @property
    def _cloud_failure_count(self) -> int:
        """[已废弃] 向后兼容：获取失败次数"""
        return self._policy.cooldown_state.failure_count

    @_cloud_failure_count.setter
    def _cloud_failure_count(self, value: int) -> None:
        """[已废弃] 向后兼容：设置失败次数"""
        self._policy.cooldown_state.failure_count = value

    @property
    def cloud_cooldown_seconds(self) -> int:
        """[已废弃] 向后兼容：获取冷却秒数"""
        # 返回 unknown 类型的默认冷却时间
        return self._policy.cooldown_config.unknown_cooldown_seconds

    def _start_cloud_cooldown(self) -> None:
        """[已废弃] 向后兼容：启动冷却

        使用默认错误类型启动冷却期。
        推荐使用 _policy.start_cooldown(error) 以获得按错误类型自适应的冷却。
        """
        self._policy.start_cooldown(
            error=Exception("Manual cooldown trigger"),
            config_hash=self._get_config_hash(),
        )

    def _reset_cloud_cooldown(self) -> None:
        """[已废弃] 向后兼容：重置冷却

        重置冷却状态，包括冷却时间和失败计数。
        推荐使用 reset_cooldown() 方法。
        """
        self._policy.reset()

    def _check_cooldown_expired(self) -> bool:
        """[已废弃] 向后兼容：检查冷却是否过期

        如果冷却期已过期，重置可用性缓存并返回 True。
        推荐使用 _policy.check_cooldown_expired() 方法。

        Returns:
            是否已过期
        """
        if self._policy.check_cooldown_expired():
            self._cloud_executor.reset_availability_cache()
            return True
        return False

    def _get_config_hash(self) -> Optional[str]:
        """获取当前配置的哈希值（用于检测 AuthError 配置变化）

        基于 CloudClientFactory.resolve_api_key() 的最终解析值计算 hash，
        保证当 key 来源变化（env/CLI/config.yaml/agent_config）时都能触发冷却提前解除。

        优先级（与 CloudClientFactory.resolve_api_key 一致）：
        1. agent_config.api_key
        2. 环境变量 CURSOR_API_KEY
        3. 环境变量 CURSOR_CLOUD_API_KEY
        4. config.yaml 中的 cloud_agent.api_key
        """
        import hashlib

        # 使用 CloudClientFactory.resolve_api_key 获取最终解析的 API Key
        # 这会按优先级检查所有来源，确保 key 变化时能触发冷却解除
        resolved_api_key = CloudClientFactory.resolve_api_key(
            explicit_api_key=None,  # 无显式参数（运行时不传递）
            agent_config=self._cli_config,
            auth_config=None,  # auth_config 由 CloudAgentExecutor 管理
        )

        if resolved_api_key:
            return hashlib.sha256(resolved_api_key.encode()).hexdigest()[:16]
        return None

    def _map_cloud_failure_to_error_type(self, failure_info: CloudFailureInfo) -> CloudErrorType:
        """将 CloudFailureInfo.kind 映射到 CloudErrorType

        Args:
            failure_info: Cloud 错误分类信息

        Returns:
            对应的 CloudErrorType 枚举值
        """
        kind_to_error_type = {
            CloudFailureKind.AUTH: CloudErrorType.AUTH,
            CloudFailureKind.NO_KEY: CloudErrorType.AUTH,  # 无 key 视为认证错误
            CloudFailureKind.RATE_LIMIT: CloudErrorType.RATE_LIMIT,
            CloudFailureKind.TIMEOUT: CloudErrorType.TIMEOUT,
            CloudFailureKind.NETWORK: CloudErrorType.NETWORK,
            CloudFailureKind.SERVICE: CloudErrorType.NETWORK,  # 服务端错误视为网络问题
            CloudFailureKind.QUOTA: CloudErrorType.AUTH,  # 配额问题视为认证类问题
            CloudFailureKind.UNKNOWN: CloudErrorType.UNKNOWN,
        }
        return kind_to_error_type.get(failure_info.kind, CloudErrorType.UNKNOWN)

    def _start_cooldown_from_failure_info(
        self,
        failure_info: CloudFailureInfo,
        config_hash: Optional[str] = None,
    ) -> CooldownMetadata:
        """根据 CloudFailureInfo 启动冷却期

        将 CloudFailureInfo.kind/retry_after/retryable 映射到冷却策略：
        - RateLimitError: 使用 retry_after（30s~300s 夹逼）
        - AuthError/NO_KEY: 10 分钟，需配置变化才能重试
        - Network/Timeout/Service: 使用标准冷却时间

        Args:
            failure_info: Cloud 错误分类信息
            config_hash: 当前配置哈希（用于 AuthError/NO_KEY）

        Returns:
            冷却元数据
        """
        config = self._policy.cooldown_config
        state = self._policy.cooldown_state
        kind = failure_info.kind

        # 更新状态 - 直接使用 failure_kind
        state.failure_kind = kind
        state.failure_count += 1
        state.last_error_message = failure_info.message
        state.config_hash = config_hash

        # 根据 failure_info.kind 计算冷却时间
        cooldown_seconds: int
        if kind == CloudFailureKind.RATE_LIMIT:
            # 使用 retry_after（如果有），否则使用默认值
            if failure_info.retry_after is not None:
                state.retry_after_hint = failure_info.retry_after
                # 最小/最大夹逼
                cooldown_seconds = max(
                    config.rate_limit_min_seconds, min(failure_info.retry_after, config.rate_limit_max_seconds)
                )
                logger.debug(f"RateLimitError retry_after={failure_info.retry_after}s, 夹逼后={cooldown_seconds}s")
            else:
                cooldown_seconds = config.rate_limit_default_seconds
        elif kind in (CloudFailureKind.AUTH, CloudFailureKind.NO_KEY, CloudFailureKind.QUOTA):
            # Auth/NO_KEY/QUOTA 错误使用更长冷却，需要配置变化才能重试
            cooldown_seconds = config.auth_cooldown_seconds
        elif kind in (CloudFailureKind.NETWORK, CloudFailureKind.SERVICE):
            cooldown_seconds = config.network_cooldown_seconds
        elif kind == CloudFailureKind.TIMEOUT:
            cooldown_seconds = config.timeout_cooldown_seconds
        else:
            cooldown_seconds = config.unknown_cooldown_seconds

        state.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)

        # 记录日志 - 使用 kind.value
        logger.warning(
            f"Cloud 执行失败（第 {state.failure_count} 次），错误类型: {kind.value}，进入冷却期 {cooldown_seconds} 秒"
        )

        return self._policy.get_cooldown_metadata()

    def _build_cooldown_info_dict(
        self,
        cooldown_meta: CooldownMetadata,
        failure_info: Optional[CloudFailureInfo] = None,
        fallback_reason: Optional[str] = None,
        has_ampersand_prefix: bool = False,
        mode_source: Optional[str] = None,
    ) -> dict[str, Any]:
        """构建冷却信息字典，包含统一的用户提示消息

        使用 core.execution_policy.build_cooldown_info 权威函数确保字段结构一致。

        输出字段（符合 cooldown_info 契约）：
        - kind: CloudFailureKind 值（no_key/auth/rate_limit 等）
        - user_message: 用户友好消息（非空）
        - retryable: 是否可重试
        - retry_after: 建议重试等待秒数
        - reason/fallback_reason: 回退原因
        - error_type/failure_kind: 兼容字段
        - in_cooldown: 是否处于冷却期
        - remaining_seconds: 冷却剩余秒数
        - failure_count: 连续失败次数
        - message_level: 消息级别（"warning" 或 "info"）

        Args:
            cooldown_meta: 冷却元数据
            failure_info: Cloud 错误分类信息（可选）
            fallback_reason: 回退原因描述（可选）
            has_ampersand_prefix: 原始 prompt 是否包含 '&' 前缀（语法检测层面）
            mode_source: execution_mode 的来源（"cli"/"config"/None）

        Returns:
            符合 cooldown_info 契约的字典
        """
        if failure_info is not None:
            # 使用权威函数构建统一结构
            return build_cooldown_info(
                failure_info=failure_info,
                fallback_reason=fallback_reason,
                requested_mode="auto",
                has_ampersand_prefix=has_ampersand_prefix,
                in_cooldown=cooldown_meta.in_cooldown,
                remaining_seconds=cooldown_meta.remaining_seconds,
                failure_count=cooldown_meta.failure_count,
                mode_source=mode_source,
            )
        else:
            # 当没有 failure_info 时，使用 from_metadata 版本
            return build_cooldown_info_from_metadata(
                failure_kind=cooldown_meta.failure_kind,
                failure_message=cooldown_meta.last_error_message,
                retry_after=None,
                retryable=False,
                fallback_reason=fallback_reason,
                requested_mode="auto",
                has_ampersand_prefix=has_ampersand_prefix,
                in_cooldown=cooldown_meta.in_cooldown,
                remaining_seconds=cooldown_meta.remaining_seconds,
                failure_count=cooldown_meta.failure_count,
                mode_source=mode_source,
            )

    def _is_cloud_enabled(self) -> bool:
        """检查是否应该尝试 Cloud

        在 AUTO 模式下，总是返回 True（尝试 Cloud），
        因为 AUTO 模式的语义是"Cloud 优先，不可用时回退到 CLI"。

        cloud_enabled 配置主要用于控制 `& 前缀` 的自动检测，
        而不是完全禁用 AUTO 模式下的 Cloud 尝试。
        """
        # AUTO 模式下总是尝试 Cloud
        return True

    async def _select_executor(
        self,
        current_config_hash: Optional[str] = None,
    ) -> tuple[AgentExecutor, Optional[str]]:
        """选择可用的执行器

        优先级:
        1. Cloud API（如果可用且不在冷却期且 cloud_enabled=True）
        2. CLI（回退选项）

        冷却策略:
        - Cloud 失败后进入冷却期，冷却期内不尝试 Cloud
        - 冷却期结束后重新检查 Cloud 可用性
        - AuthError 时，配置变化可提前结束冷却

        Returns:
            (executor, skip_reason): 执行器实例和跳过 Cloud 的原因（如果有）
        """
        if not self._enable_cooldown:
            # 冷却禁用时，直接检查 Cloud 可用性
            if await self._cloud_executor.check_available():
                return self._cloud_executor, None
            return self._cli_executor, "Cloud 不可用"

        # 使用 Policy 判断是否应该尝试 Cloud
        should_try, skip_reason = self._policy.should_try_cloud(
            cloud_enabled=self._is_cloud_enabled(),
            current_config_hash=current_config_hash,
        )

        if not should_try:
            logger.debug(f"跳过 Cloud: {skip_reason}")
            if await self._cli_executor.check_available():
                return self._cli_executor, skip_reason
            logger.warning("CLI 也不可用，仍使用 CLI（可能会失败）")
            return self._cli_executor, skip_reason

        # 检查冷却期是否已过期
        if self._policy.check_cooldown_expired():
            # 重置 Cloud 执行器的可用性缓存，以便重新检查
            self._cloud_executor.reset_availability_cache()

        # 尝试 Cloud
        if await self._cloud_executor.check_available():
            logger.debug("使用 Cloud API 执行器")
            return self._cloud_executor, None

        # 回退到 CLI
        if await self._cli_executor.check_available():
            logger.debug("Cloud 不可用，回退到 CLI 执行器")
            return self._cli_executor, "Cloud 可用性检查失败"

        # 两者都不可用，仍然返回 CLI（可能会失败）
        logger.warning("Cloud 和 CLI 均不可用，尝试使用 CLI")
        return self._cli_executor, "Cloud 和 CLI 均不可用"

    async def execute(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> AgentResult:
        """自动选择执行器并执行任务

        冷却策略（按错误类型自适应）：
        - Cloud 失败后根据错误类型进入相应冷却期
        - 冷却期内优先使用 CLI
        - 冷却期结束后重新尝试 Cloud
        - Cloud 成功时重置冷却状态
        - CLI 回退时使用 sanitize_prompt_for_cli_fallback 清理 prompt
        """
        config_hash = self._get_config_hash()

        # 检查是否需要重新选择执行器
        should_reselect = self._preferred_executor is None
        if (
            not should_reselect
            and self._preferred_executor is not None
            and self._preferred_executor.executor_type == "cli"
            and self._policy.check_cooldown_expired()
        ):
            # 冷却期结束，重新选择（可能切换回 Cloud）
            should_reselect = True
            logger.debug("冷却期已结束，重新选择执行器")

        skip_reason: Optional[str] = None
        if should_reselect:
            self._preferred_executor, skip_reason = await self._select_executor(
                current_config_hash=config_hash,
            )

        executor = self._preferred_executor
        if executor is None:
            raise RuntimeError("未找到可用执行器")
        result: AgentResult

        # 执行任务
        try:
            result = await executor.execute(
                prompt=prompt,
                context=context,
                working_directory=working_directory,
                timeout=timeout,
            )
        except Exception as e:
            # 捕获执行异常，可能需要启动冷却
            if executor.executor_type == "cloud":
                # 使用 classify_cloud_failure 分类异常
                failure_info = classify_cloud_failure(e)

                # 使用 CloudFailureInfo 启动冷却
                cooldown_meta = self._start_cooldown_from_failure_info(
                    failure_info=failure_info,
                    config_hash=config_hash,
                )

                remaining_str = f"{cooldown_meta.remaining_seconds:.1f}" if cooldown_meta.remaining_seconds else "0"
                logger.info(f"Cloud 执行异常 ({type(e).__name__})，冷却 {remaining_str} 秒，尝试回退到 CLI")
                self._preferred_executor = self._cli_executor

                # 检测原始 prompt 是否包含 & 前缀（语法检测层面）
                from core.cloud_utils import is_cloud_request

                has_ampersand_prefix = is_cloud_request(prompt)

                # 使用 sanitize_prompt_for_cli_fallback 清理 prompt
                sanitized_prompt = CloudExecutionPolicy.sanitize_prompt_for_cli_fallback(prompt)
                result = await self._cli_executor.execute(
                    prompt=sanitized_prompt,
                    context=context,
                    working_directory=working_directory,
                    timeout=timeout,
                )

                # 使用 _build_cooldown_info_dict 构建统一的冷却信息
                result.cooldown_info = self._build_cooldown_info_dict(
                    cooldown_meta=cooldown_meta,
                    failure_info=failure_info,
                    fallback_reason=str(e),
                    has_ampersand_prefix=has_ampersand_prefix,
                )
                return result
            else:
                raise

        # 处理 Cloud 执行结果
        if executor.executor_type == "cloud":
            if result.success:
                # Cloud 成功，重置冷却状态
                self._policy.reset()
            else:
                # Cloud 失败，统一使用 classify_cloud_failure(result.raw_result or result.error)
                # 优先使用 raw_result（包含结构化错误字段 error_type/retry_after）
                # 回退到 error 消息字符串
                error_source = result.raw_result or result.error or "Cloud execution failed"

                # 使用 classify_cloud_failure 获取 CloudFailureInfo
                # classify_cloud_failure 会自动从 dict/str 中提取 error_type/retry_after
                failure_info = classify_cloud_failure(error_source)

                # 使用 CloudFailureInfo 启动冷却
                cooldown_meta = self._start_cooldown_from_failure_info(
                    failure_info=failure_info,
                    config_hash=config_hash,
                )

                error_type_str = cooldown_meta.error_type.value if cooldown_meta.error_type else "unknown"
                remaining_str = f"{cooldown_meta.remaining_seconds:.1f}" if cooldown_meta.remaining_seconds else "0"
                logger.info(f"Cloud 执行失败，错误类型: {error_type_str}，冷却 {remaining_str} 秒，尝试回退到 CLI")
                self._preferred_executor = self._cli_executor

                # 检测原始 prompt 是否包含 & 前缀（语法检测层面）
                from core.cloud_utils import is_cloud_request

                has_ampersand_prefix = is_cloud_request(prompt)

                # 使用 sanitize_prompt_for_cli_fallback 清理 prompt
                sanitized_prompt = CloudExecutionPolicy.sanitize_prompt_for_cli_fallback(prompt)
                result = await self._cli_executor.execute(
                    prompt=sanitized_prompt,
                    context=context,
                    working_directory=working_directory,
                    timeout=timeout,
                )

                # 使用 _build_cooldown_info_dict 构建统一的冷却信息
                result.cooldown_info = self._build_cooldown_info_dict(
                    cooldown_meta=cooldown_meta,
                    failure_info=failure_info,
                    fallback_reason=failure_info.message,
                    has_ampersand_prefix=has_ampersand_prefix,
                )

        # 如果有跳过原因，添加到元数据
        if skip_reason and result.cooldown_info is None:
            meta = self._policy.get_cooldown_metadata()
            if meta.in_cooldown:
                # 使用权威函数构建统一结构
                # mode_source=None: 执行层无法得知来源，使用默认 info 级别
                # 如果 has_ampersand_prefix=True，会自动升级为 warning
                result.cooldown_info = build_cooldown_info_from_metadata(
                    failure_kind=meta.failure_kind,
                    failure_message=meta.last_error_message,
                    retry_after=None,
                    retryable=False,
                    fallback_reason=skip_reason,
                    requested_mode="auto",
                    has_ampersand_prefix=False,
                    in_cooldown=meta.in_cooldown,
                    remaining_seconds=meta.remaining_seconds,
                    failure_count=meta.failure_count,
                    mode_source=None,  # 保守默认 info
                )
                # 添加 skip_reason 字段（此场景特有）
                result.cooldown_info[CooldownInfoFields.SKIP_REASON] = skip_reason

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
        self._policy.reset()
        self._cloud_executor.reset_availability_cache()
        # 重置偏好，以便重新选择
        self._preferred_executor = None
        logger.debug("手动重置 Cloud 冷却状态")

    def get_cooldown_metadata(self) -> CooldownMetadata:
        """获取当前冷却元数据

        供上层打印提示使用。

        Returns:
            冷却元数据
        """
        return self._policy.get_cooldown_metadata()


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
        cooldown_config: Optional[CooldownConfig] = None,
    ) -> AgentExecutor:
        """创建执行器实例

        Args:
            mode: 执行模式
            cli_config: CLI 执行器配置
            cloud_auth_config: Cloud 认证配置
            cooldown_config: 冷却配置（仅 AUTO 模式使用）

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
            # 如果未提供 cooldown_config，从 config.yaml 加载
            effective_cooldown_config = cooldown_config
            if effective_cooldown_config is None:
                effective_cooldown_config = CooldownConfig.from_config()
            return AutoAgentExecutor(
                cli_config=cli_config,
                cloud_auth_config=cloud_auth_config,
                cooldown_config=effective_cooldown_config,
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
        mode = getattr(config, "execution_mode", ExecutionMode.AUTO)

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
