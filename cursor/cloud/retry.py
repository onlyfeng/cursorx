"""重试策略模块

提供指数退避重试机制，支持:
- 可配置的重试参数
- 自动处理 Retry-After 头
- 异步重试装饰器
"""
import asyncio
import functools
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

# ParamSpec 在 Python 3.10+ 中可用
if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

from loguru import logger

# 类型变量用于泛型装饰器
P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class RetryConfig:
    """重试配置

    Attributes:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_base: 指数基数
        jitter: 是否添加随机抖动
        retry_on: 需要重试的异常类型列表
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    # 默认重试的异常类型，延迟导入避免循环依赖
    retry_on: tuple = field(default_factory=lambda: (
        asyncio.TimeoutError,
        ConnectionError,
    ))

    def calculate_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """计算重试延迟时间

        使用指数退避算法：delay = base_delay * (exponential_base ^ attempt)

        Args:
            attempt: 当前尝试次数（从 0 开始）
            retry_after: Retry-After 头指定的延迟（优先使用）

        Returns:
            延迟时间（秒）
        """
        # 如果有 Retry-After，优先使用
        if retry_after and retry_after > 0:
            delay = min(retry_after, self.max_delay)
            logger.debug(f"使用 Retry-After 延迟: {delay:.1f}s")
            return delay

        # 指数退避计算
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        # 添加随机抖动（0-25%）
        if self.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(0, jitter_range)

        return delay


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """重试装饰器（用于异步函数）

    实现指数退避重试策略，支持 RateLimitError 的 Retry-After 头。

    Args:
        config: 重试配置
        on_retry: 重试回调函数，接收 (attempt, error, delay) 参数

    Returns:
        装饰器函数

    Example:
        @with_retry(RetryConfig(max_retries=5, base_delay=2.0))
        async def call_api():
            ...
    """
    # 从本地 exceptions 模块导入
    from .exceptions import AuthError, AuthErrorCode, CloudAgentError, NetworkError, RateLimitError

    retry_config = config or RetryConfig()

    # 更新默认的 retry_on 以包含云端错误类型
    if retry_config.retry_on == (asyncio.TimeoutError, ConnectionError):
        retry_config = RetryConfig(
            max_retries=retry_config.max_retries,
            base_delay=retry_config.base_delay,
            max_delay=retry_config.max_delay,
            exponential_base=retry_config.exponential_base,
            jitter=retry_config.jitter,
            retry_on=(
                RateLimitError,
                NetworkError,
                asyncio.TimeoutError,
                ConnectionError,
            ),
        )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(retry_config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except retry_config.retry_on as e:
                    last_exception = e

                    # 最后一次尝试失败，不再重试
                    if attempt >= retry_config.max_retries:
                        logger.error(
                            f"重试耗尽 [{func.__name__}]: 已尝试 {attempt + 1} 次, "
                            f"错误: {e}"
                        )
                        raise

                    # 获取 Retry-After（如果是 RateLimitError）
                    retry_after = None
                    if isinstance(e, RateLimitError) or isinstance(e, CloudAgentError):
                        retry_after = e.retry_after

                    # 计算延迟
                    delay = retry_config.calculate_delay(attempt, retry_after)

                    logger.warning(
                        f"重试 [{func.__name__}]: 尝试 {attempt + 1}/{retry_config.max_retries + 1}, "
                        f"延迟 {delay:.1f}s, 错误: {type(e).__name__}: {e}"
                    )

                    # 调用回调
                    if on_retry:
                        try:
                            on_retry(attempt, e, delay)
                        except Exception as cb_error:
                            logger.debug(f"重试回调错误: {cb_error}")

                    # 等待后重试
                    await asyncio.sleep(delay)

                except AuthError as e:
                    # 认证错误不重试（除非是网络相关）
                    if e.code == AuthErrorCode.NETWORK_ERROR:
                        last_exception = e
                        if attempt >= retry_config.max_retries:
                            raise
                        delay = retry_config.calculate_delay(attempt)
                        logger.warning(f"认证网络错误，重试: {e}")
                        await asyncio.sleep(delay)
                    else:
                        raise

            # 不应该到达这里
            if last_exception:
                raise last_exception
            raise RuntimeError("重试逻辑错误")

        return wrapper

    return decorator


async def retry_async(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """异步重试辅助函数

    对于不方便使用装饰器的场景，提供函数式重试接口。

    Args:
        func: 要执行的异步函数（无参数）
        config: 重试配置
        on_retry: 重试回调

    Returns:
        函数执行结果

    Example:
        result = await retry_async(
            lambda: client.fetch(url),
            RetryConfig(max_retries=3),
        )
    """
    retry_config = config or RetryConfig()

    @with_retry(retry_config, on_retry)
    async def _wrapper():
        return await func()

    return await _wrapper()
