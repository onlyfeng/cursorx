"""网页获取器

提供统一的网页获取接口，支持多种获取方式：
1. MCP fetch 方式（优先使用，通过 agent CLI）
2. Playwright 浏览器自动化（支持 JavaScript 渲染，适用于 SPA 等动态页面）
3. 命令行工具后备（curl/lynx）

特性：
- 并发获取能力（asyncio）
- 错误处理和重试机制
- 自动选择最佳获取方式（JS 页面自动优先使用 Playwright）
- Playwright 支持：等待选择器、滚动加载懒加载内容、无头模式等
- URL 安全策略：支持 scheme 白名单、域名白名单、私网拒绝等

================================================================================
副作用控制策略 (Side Effect Control)
================================================================================

详细策略矩阵参见: core/execution_policy.py

**本模块产生的副作用**:
| 操作                    | 副作用类型     | 说明                              |
|-------------------------|----------------|-----------------------------------|
| fetch()                 | 网络请求       | HTTP/HTTPS 请求                   |
| fetch_multiple()        | 网络请求       | 批量 HTTP/HTTPS 请求              |
| _fetch_via_*()          | 进程启动       | 启动 curl/lynx/playwright 进程    |
| _fetch_via_playwright() | 浏览器启动     | 启动 Chromium 浏览器实例          |

**策略行为**:
| 策略        | 行为                                                  |
|-------------|-------------------------------------------------------|
| normal      | 正常执行网络请求                                      |
| skip-online | 禁止网络请求：返回缓存结果或 FetchResult(success=False)|
| dry-run     | 正常执行（网络请求用于分析，不涉及持久化写入）        |
| minimal     | 禁止网络请求：同 skip-online                          |

**实现契约**:
当调用方需要 skip-online 语义时，应：
1. 优先查询本地缓存（由调用方实现）
2. 若需调用 fetcher，fetcher 应识别 skip_online 标记
3. 返回 FetchResult(success=False, content="", error="skip-online mode")
4. 或返回缓存结果（若支持缓存）

注意：skip_online 逻辑应由调用方（如 KnowledgeManager/SelfIterator）在调用前判断，
本模块作为底层执行器，不应持有 skip_online 状态，但应支持通过参数传入。
"""

import asyncio
import ipaddress
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

from loguru import logger

# 全局缓存：避免每个 WebFetcher 实例都做一遍外部依赖探测（会启动子进程，较慢）
_AVAILABLE_METHODS_CACHE: Optional[list["FetchMethod"]] = None
_AVAILABLE_METHODS_LOCK = asyncio.Lock()


def sanitize_url_for_log(url: str, max_path_len: int = 30) -> str:
    """截断 URL 中的敏感部分用于日志输出

    只保留 scheme + host，path 截断到指定长度，移除 query/fragment。
    避免在日志中泄露敏感信息（如 token、session_id 等）。

    Args:
        url: 原始 URL
        max_path_len: path 部分的最大长度（超过则截断并添加 ...）

    Returns:
        截断后的 URL 字符串（安全用于日志输出）

    Examples:
        >>> sanitize_url_for_log("https://example.com/very/long/path/to/resource?token=secret")
        'https://example.com/very/long/path/to/resou...'
        >>> sanitize_url_for_log("https://example.com/short")
        'https://example.com/short'
    """
    if not url:
        return "<empty>"

    try:
        parsed = urlparse(url)
        # 只保留 scheme + host + 截断的 path
        path = parsed.path or "/"
        if len(path) > max_path_len:
            path = path[:max_path_len] + "..."

        # 重建 URL（不包含 query 和 fragment）
        sanitized = f"{parsed.scheme}://{parsed.netloc}{path}"
        return sanitized
    except Exception:
        # 解析失败时返回截断的原始字符串
        if len(url) > 50:
            return url[:50] + "..."
        return url


@dataclass
class UrlRejectionReason:
    """URL 拒绝原因（结构化，供上层汇总）

    提供 machine-readable 的拒绝原因，便于上层进行统计和汇总。

    Attributes:
        url: 被拒绝的 URL（已截断敏感部分）
        reason: 人类可读的拒绝原因
        policy_type: 策略类型代码 (scheme/domain/private_network/ip_address/ipv6/hostname/empty/parse/url_prefix)
        raw_url: 原始 URL（仅在调试时使用，默认不暴露）
    """

    url: str  # 截断后的 URL（安全用于日志）
    reason: str
    policy_type: str
    raw_url: Optional[str] = field(default=None, repr=False)  # 原始 URL，repr 中不显示

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "url": self.url,
            "reason": self.reason,
            "policy_type": self.policy_type,
        }

    @classmethod
    def from_policy_error(cls, error: "UrlPolicyError") -> "UrlRejectionReason":
        """从 UrlPolicyError 创建 UrlRejectionReason"""
        return cls(
            url=sanitize_url_for_log(error.url),
            reason=error.reason,
            policy_type=error.policy_type,
            raw_url=error.url,
        )


class UrlPolicyError(Exception):
    """URL 策略校验错误

    当 URL 不满足安全策略时抛出此异常。
    """

    def __init__(self, url: str, reason: str, policy_type: str = "unknown"):
        """初始化 URL 策略错误

        Args:
            url: 被拒绝的 URL
            reason: 拒绝原因
            policy_type: 策略类型 (scheme/domain/private_network/ip_address/ipv6/hostname/empty/parse/url_prefix)
        """
        self.url = url
        self.reason = reason
        self.policy_type = policy_type
        super().__init__(f"URL 策略拒绝: {sanitize_url_for_log(url)} - {reason}")

    def to_rejection_reason(self) -> UrlRejectionReason:
        """转换为结构化的拒绝原因"""
        return UrlRejectionReason.from_policy_error(self)


@dataclass
class UrlPolicy:
    """URL 安全策略配置

    控制允许访问的 URL 范围，防止 SSRF 和非预期访问。

    使用示例:
    ```python
    # 只允许 HTTPS，禁止私网
    policy = UrlPolicy(
        allowed_schemes=["https"],
        deny_private_networks=True,
    )

    # 只允许特定域名
    policy = UrlPolicy(
        allowed_domains=["docs.cursor.com", "cursor.com"],
    )

    # 只允许特定 URL 前缀
    policy = UrlPolicy(
        allowed_url_prefixes=["https://docs.cursor.com/"],
    )
    ```
    """

    # 允许的 URL scheme，默认 http/https
    # 空列表表示不限制
    allowed_schemes: list[str] = field(default_factory=lambda: ["http", "https"])

    # 允许的域名列表（精确匹配或子域名匹配）
    # 空列表表示不限制
    # 支持通配符: *.example.com 匹配所有子域名
    allowed_domains: list[str] = field(default_factory=list)

    # 允许的 URL 前缀列表
    # 空列表表示不限制
    allowed_url_prefixes: list[str] = field(default_factory=list)

    # 是否拒绝私有网络地址 (localhost, 127.0.0.1, 10.x.x.x, 192.168.x.x 等)
    deny_private_networks: bool = True

    # 是否拒绝 IPv6 地址
    deny_ipv6: bool = False

    # 是否拒绝纯 IP 地址（无域名）
    deny_ip_addresses: bool = False

    def validate(self, url: str) -> None:
        """校验 URL 是否符合策略

        Args:
            url: 待校验的 URL

        Raises:
            UrlPolicyError: URL 不符合策略时抛出
        """
        if not url:
            raise UrlPolicyError(url, "URL 为空", "empty")

        try:
            parsed = urlparse(url)
        except Exception as e:
            raise UrlPolicyError(url, f"URL 解析失败: {e}", "parse") from e

        # 1. 校验 scheme
        self._validate_scheme(url, parsed)

        # 2. 校验 hostname
        hostname = parsed.hostname or ""
        if not hostname:
            raise UrlPolicyError(url, "缺少主机名", "hostname")

        # 3. 校验 IP 相关策略
        self._validate_ip_policies(url, hostname)

        # 4. 校验域名白名单
        self._validate_domain(url, hostname)

        # 5. 校验 URL 前缀白名单
        self._validate_url_prefix(url)

    def _validate_scheme(self, url: str, parsed) -> None:
        """校验 URL scheme"""
        scheme = parsed.scheme.lower()
        if self.allowed_schemes and scheme not in self.allowed_schemes:
            raise UrlPolicyError(
                url,
                f"不允许的协议: {scheme}，允许的协议: {self.allowed_schemes}",
                "scheme",
            )

    def _validate_ip_policies(self, url: str, hostname: str) -> None:
        """校验 IP 相关策略"""
        # 检查是否是 IP 地址
        is_ip = False
        ip_obj = None

        try:
            ip_obj = ipaddress.ip_address(hostname)
            is_ip = True
        except ValueError:
            # 不是 IP 地址，可能是域名
            pass

        # 拒绝纯 IP 地址
        if self.deny_ip_addresses and is_ip:
            raise UrlPolicyError(url, "不允许直接使用 IP 地址", "ip_address")

        # 拒绝 IPv6
        if self.deny_ipv6 and is_ip and ip_obj and ip_obj.version == 6:
            raise UrlPolicyError(url, "不允许 IPv6 地址", "ipv6")

        # 拒绝私有网络
        if self.deny_private_networks:
            self._check_private_network(url, hostname, ip_obj)

    def _check_private_network(
        self, url: str, hostname: str, ip_obj: Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]
    ) -> None:
        """检查是否为私有网络地址"""
        # 特殊主机名
        private_hostnames = ["localhost", "localhost.localdomain"]
        if hostname.lower() in private_hostnames:
            raise UrlPolicyError(url, "不允许访问 localhost", "private_network")

        # IP 地址检查
        if ip_obj:
            if ip_obj.is_private:
                raise UrlPolicyError(url, f"不允许访问私有网络地址: {hostname}", "private_network")
            if ip_obj.is_loopback:
                raise UrlPolicyError(url, f"不允许访问回环地址: {hostname}", "private_network")
            if ip_obj.is_link_local:
                raise UrlPolicyError(url, f"不允许访问链路本地地址: {hostname}", "private_network")
            if ip_obj.is_reserved:
                raise UrlPolicyError(url, f"不允许访问保留地址: {hostname}", "private_network")
            # 检查 0.0.0.0
            if str(ip_obj) == "0.0.0.0":
                raise UrlPolicyError(url, "不允许访问 0.0.0.0", "private_network")
        else:
            # 域名情况：尝试解析并检查
            # 注意：这里不进行 DNS 解析，因为可能会有性能问题
            # 如果需要严格检查，可以启用 DNS 解析
            # 只检查已知的私有域名模式
            if self._is_likely_private_domain(hostname):
                raise UrlPolicyError(url, f"疑似私有网络域名: {hostname}", "private_network")

    def _is_likely_private_domain(self, hostname: str) -> bool:
        """检查域名是否疑似私有网络域名"""
        hostname_lower = hostname.lower()

        # 常见私有域名模式
        private_patterns = [
            r"^localhost$",
            r"^localhost\.\w+$",
            r"^.*\.local$",
            r"^.*\.localhost$",
            r"^.*\.internal$",
            r"^.*\.private$",
            r"^.*\.corp$",
            r"^.*\.lan$",
            r"^.*\.home$",
            r"^192-168-\d+-\d+\..*$",  # 192-168-x-x 形式
            r"^10-\d+-\d+-\d+\..*$",  # 10-x-x-x 形式
        ]

        return any(re.match(pattern, hostname_lower) for pattern in private_patterns)

    def _validate_domain(self, url: str, hostname: str) -> None:
        """校验域名白名单"""
        if not self.allowed_domains:
            return  # 未配置域名白名单，跳过

        hostname_lower = hostname.lower()
        for allowed in self.allowed_domains:
            allowed_lower = allowed.lower()
            if allowed_lower.startswith("*."):
                # 通配符匹配：*.example.com 匹配 sub.example.com
                suffix = allowed_lower[1:]  # .example.com
                if hostname_lower.endswith(suffix) or hostname_lower == allowed_lower[2:]:
                    return
            else:
                # 精确匹配
                if hostname_lower == allowed_lower:
                    return

        raise UrlPolicyError(
            url,
            f"域名不在白名单中: {hostname}，允许的域名: {self.allowed_domains}",
            "domain",
        )

    def _validate_url_prefix(self, url: str) -> None:
        """校验 URL 前缀白名单"""
        if not self.allowed_url_prefixes:
            return  # 未配置 URL 前缀白名单，跳过

        url_lower = url.lower()
        for prefix in self.allowed_url_prefixes:
            if url_lower.startswith(prefix.lower()):
                return

        raise UrlPolicyError(
            url,
            f"URL 前缀不在白名单中，允许的前缀: {self.allowed_url_prefixes}",
            "url_prefix",
        )

    def is_valid(self, url: str) -> bool:
        """检查 URL 是否有效（不抛出异常）

        Args:
            url: 待校验的 URL

        Returns:
            是否有效
        """
        try:
            self.validate(url)
            return True
        except UrlPolicyError:
            return False

    def get_validation_error(self, url: str) -> Optional[str]:
        """获取 URL 校验错误信息

        Args:
            url: 待校验的 URL

        Returns:
            错误信息，如果有效则返回 None
        """
        try:
            self.validate(url)
            return None
        except UrlPolicyError as e:
            return str(e)

    def get_rejection_reason(self, url: str) -> Optional[UrlRejectionReason]:
        """获取 URL 拒绝的结构化原因

        Args:
            url: 待校验的 URL

        Returns:
            结构化的拒绝原因，如果有效则返回 None
        """
        try:
            self.validate(url)
            return None
        except UrlPolicyError as e:
            return e.to_rejection_reason()


# 默认策略：允许 http/https，拒绝私网
DEFAULT_URL_POLICY = UrlPolicy()


class FetchMethod(str, Enum):
    """获取方式"""

    MCP = "mcp"  # MCP fetch（通过 agent CLI）
    CURL = "curl"  # curl 命令
    LYNX = "lynx"  # lynx 命令（纯文本）
    PLAYWRIGHT = "playwright"  # Playwright 浏览器自动化（支持 JS 渲染）
    AUTO = "auto"  # 自动选择


class ContentFormat(str, Enum):
    """内容格式"""

    TEXT = "text"  # 纯文本
    HTML = "html"  # 原始 HTML
    MARKDOWN = "markdown"  # Markdown


@dataclass
class FetchConfig:
    """获取配置"""

    # 获取方式（auto 自动选择）
    method: FetchMethod = FetchMethod.AUTO

    # 期望的内容格式
    content_format: ContentFormat = ContentFormat.TEXT

    # 超时设置（秒）
    timeout: int = 30

    # 重试次数
    max_retries: int = 3

    # 重试间隔（秒）
    retry_delay: float = 1.0

    # 并发限制
    max_concurrent: int = 5

    # agent CLI 路径
    agent_path: str = "agent"

    # 用户代理
    user_agent: str = "Mozilla/5.0 (compatible; WebFetcher/1.0)"

    # Playwright 特定配置
    playwright_headless: bool = True  # 无头模式
    playwright_wait_for_selector: Optional[str] = None  # 等待指定选择器出现
    playwright_scroll_to_bottom: bool = False  # 滚动到页面底部（加载懒加载内容）
    playwright_wait_after_load: float = 0.0  # 页面加载后额外等待时间（秒）
    playwright_js_enabled: bool = True  # 是否启用 JavaScript

    # URL 安全策略配置
    # 为 None 时使用默认策略（允许 http/https，拒绝私网）
    # 设为 UrlPolicy() 可自定义策略
    url_policy: Optional[UrlPolicy] = None

    # 是否启用 URL 策略校验（默认启用）
    enforce_url_policy: bool = True


@dataclass
class FetchResult:
    """获取结果"""

    url: str
    success: bool
    content: str = ""
    error: Optional[str] = None
    method_used: FetchMethod = FetchMethod.AUTO
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    duration: float = 0.0
    retry_count: int = 0

    # 内容质量评分（0.0-1.0，None 表示未评估）
    quality_score: Optional[float] = None

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    # URL 策略拒绝原因（结构化，仅当因策略被拒绝时设置）
    rejection_reason: Optional[UrlRejectionReason] = None


class WebFetcher:
    """网页获取器

    统一的网页获取接口，支持多种获取方式和并发获取。

    使用示例:
    ```python
    fetcher = WebFetcher()

    # 获取单个网页
    result = await fetcher.fetch("https://example.com")

    # 并发获取多个网页
    urls = ["https://example.com", "https://google.com"]
    results = await fetcher.fetch_many(urls)
    ```
    """

    def __init__(self, config: Optional[FetchConfig] = None):
        self.config = config or FetchConfig()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._available_methods: list[FetchMethod] = []
        self._initialized = False
        # URL 策略：优先使用配置中的策略，否则使用默认策略
        self._url_policy = self.config.url_policy or DEFAULT_URL_POLICY

    async def initialize(self) -> None:
        """初始化获取器，检测可用的获取方式"""
        if self._initialized:
            return

        # 复用全局探测结果，减少重复的子进程检测开销
        global _AVAILABLE_METHODS_CACHE
        if _AVAILABLE_METHODS_CACHE is None:
            async with _AVAILABLE_METHODS_LOCK:
                if _AVAILABLE_METHODS_CACHE is None:
                    _AVAILABLE_METHODS_CACHE = await self._detect_available_methods()
        self._available_methods = list(_AVAILABLE_METHODS_CACHE or [])
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._initialized = True

        logger.info(f"WebFetcher 初始化完成，可用方式: {[m.value for m in self._available_methods]}")

    async def _detect_available_methods(self) -> list[FetchMethod]:
        """检测可用的获取方式"""
        methods = []

        # 检测 MCP（通过 agent CLI）
        if await self._check_mcp_available():
            methods.append(FetchMethod.MCP)

        # 检测 Playwright
        if await self._check_playwright_available():
            methods.append(FetchMethod.PLAYWRIGHT)

        # 检测 curl
        if shutil.which("curl"):
            methods.append(FetchMethod.CURL)

        # 检测 lynx
        if shutil.which("lynx"):
            methods.append(FetchMethod.LYNX)

        return methods

    async def _check_mcp_available(self) -> bool:
        """检查 MCP fetch 是否可用"""
        try:
            if not shutil.which(self.config.agent_path):
                return False
            process = await asyncio.create_subprocess_exec(
                self.config.agent_path,
                "mcp",
                "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=10,
            )

            # 检查 fetch 服务器是否在列表中
            output = stdout.decode("utf-8", errors="replace").lower()
            return "fetch" in output and process.returncode == 0

        except Exception as e:
            logger.debug(f"MCP 检测失败: {e}")
            return False

    async def _check_playwright_available(self) -> bool:
        """检查 Playwright 是否可用"""
        try:
            # 尝试导入 playwright
            import importlib.util

            spec = importlib.util.find_spec("playwright")
            if spec is None:
                logger.debug("Playwright 模块未安装")
                return False

            # 检查浏览器是否已安装（通过异步检测）
            process = await asyncio.create_subprocess_exec(
                "python",
                "-c",
                "from playwright.async_api import async_playwright; print('ok')",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=10,
            )

            return process.returncode == 0 and b"ok" in stdout

        except Exception as e:
            logger.debug(f"Playwright 检测失败: {e}")
            return False

    async def fetch(
        self,
        url: str,
        method: Optional[FetchMethod] = None,
        timeout: Optional[int] = None,
    ) -> FetchResult:
        """获取单个网页

        Args:
            url: 目标 URL
            method: 获取方式（None 使用配置的方式）
            timeout: 超时时间（None 使用配置值）

        Returns:
            获取结果
        """
        if not self._initialized:
            await self.initialize()

        # URL 策略校验
        if self.config.enforce_url_policy:
            rejection = self._validate_url_policy(url)
            if rejection:
                return FetchResult(
                    url=url,
                    success=False,
                    error=f"URL 策略拒绝: {rejection.reason}",
                    method_used=FetchMethod.AUTO,
                    rejection_reason=rejection,
                )

        method = method or self.config.method
        timeout = timeout or self.config.timeout

        # 带重试的获取
        return await self._fetch_with_retry(url, method, timeout)

    def _validate_url_policy(self, url: str) -> Optional[UrlRejectionReason]:
        """校验 URL 是否符合策略

        Args:
            url: 待校验的 URL

        Returns:
            结构化的拒绝原因，如果有效则返回 None
        """
        try:
            self._url_policy.validate(url)
            return None
        except UrlPolicyError as e:
            rejection = e.to_rejection_reason()
            logger.warning(
                f"URL 策略拒绝: {rejection.url} | policy_type={rejection.policy_type} | reason={rejection.reason}"
            )
            return rejection

    async def fetch_many(
        self,
        urls: list[str],
        method: Optional[FetchMethod] = None,
    ) -> list[FetchResult]:
        """并发获取多个网页

        Args:
            urls: URL 列表
            method: 获取方式

        Returns:
            获取结果列表（顺序与输入一致）
        """
        if not self._initialized:
            await self.initialize()

        tasks = [self._fetch_with_semaphore(url, method) for url in urls]
        return await asyncio.gather(*tasks)

    async def _fetch_with_semaphore(
        self,
        url: str,
        method: Optional[FetchMethod] = None,
    ) -> FetchResult:
        """带信号量限制的获取"""
        assert self._semaphore is not None
        async with self._semaphore:
            return await self.fetch(url, method)

    async def _fetch_with_retry(
        self,
        url: str,
        method: FetchMethod,
        timeout: int,
    ) -> FetchResult:
        """带重试的获取"""
        last_error = None
        retry_count = 0

        for attempt in range(self.config.max_retries):
            try:
                result = await self._do_fetch(url, method, timeout)
                result.retry_count = retry_count

                if result.success:
                    return result

                last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(f"获取失败 (尝试 {attempt + 1}/{self.config.max_retries}): {url} - {e}")

            retry_count += 1

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (2**attempt))

        return FetchResult(
            url=url,
            success=False,
            error=f"所有重试均失败: {last_error}",
            retry_count=retry_count,
        )

    async def _do_fetch(
        self,
        url: str,
        method: FetchMethod,
        timeout: int,
    ) -> FetchResult:
        """执行实际的获取操作"""
        import time

        start_time = time.time()

        # URL 策略校验（二次保护）
        if self.config.enforce_url_policy:
            rejection = self._validate_url_policy(url)
            if rejection:
                return FetchResult(
                    url=url,
                    success=False,
                    error=f"URL 策略拒绝: {rejection.reason}",
                    method_used=method,
                    rejection_reason=rejection,
                )

        # 选择获取方式
        if method == FetchMethod.AUTO:
            method = self._select_best_method(url)

        # 根据方式执行获取
        if method == FetchMethod.MCP:
            result = await self._fetch_via_mcp(url, timeout)
        elif method == FetchMethod.PLAYWRIGHT:
            result = await self._fetch_via_playwright(url, timeout)
        elif method == FetchMethod.CURL:
            result = await self._fetch_via_curl(url, timeout)
        elif method == FetchMethod.LYNX:
            result = await self._fetch_via_lynx(url, timeout)
        else:
            result = FetchResult(
                url=url,
                success=False,
                error=f"不支持的获取方式: {method}",
            )

        result.duration = time.time() - start_time
        result.method_used = method
        return result

    def _select_best_method(self, url: Optional[str] = None) -> FetchMethod:
        """选择最佳的获取方式

        Args:
            url: 目标 URL，用于判断是否需要 JS 渲染

        Returns:
            最佳的获取方式
        """
        # 检查是否可能需要 JS 渲染的页面
        needs_js = self._might_need_js_rendering(url) if url else False

        if needs_js and FetchMethod.PLAYWRIGHT in self._available_methods:
            # JS 页面优先使用 Playwright
            return FetchMethod.PLAYWRIGHT

        # 普通优先级：MCP > PLAYWRIGHT > CURL > LYNX
        priority = [FetchMethod.MCP, FetchMethod.PLAYWRIGHT, FetchMethod.CURL, FetchMethod.LYNX]

        for method in priority:
            if method in self._available_methods:
                return method

        # 回退到 CURL（即使不确定是否可用）
        return FetchMethod.CURL

    def _might_need_js_rendering(self, url: str) -> bool:
        """判断 URL 是否可能需要 JS 渲染

        基于 URL 模式和常见 SPA 框架特征进行判断
        """
        if not url:
            return False

        url_lower = url.lower()

        # 常见需要 JS 渲染的网站模式
        js_patterns = [
            # SPA 框架常见路径
            "/#/",  # Hash 路由
            "/app/",  # 应用路径
            # 动态内容平台
            "twitter.com",
            "x.com",
            "instagram.com",
            "facebook.com",
            "linkedin.com",
            "medium.com",
            # 技术文档（部分使用 SPA）
            "notion.so",
            "gitbook.io",
            # 电商平台
            "amazon.",
            "taobao.",
            "jd.com",
        ]

        return any(pattern in url_lower for pattern in js_patterns)

    async def _fetch_via_mcp(self, url: str, timeout: int) -> FetchResult:
        """通过 MCP fetch 获取网页

        使用 agent CLI 调用 MCP fetch 工具
        """
        try:
            prompt = f"获取 {url} 的内容"

            process = await asyncio.create_subprocess_exec(
                self.config.agent_path,
                "-p",
                prompt,
                "--output-format",
                "text",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if process.returncode == 0 and output.strip():
                return FetchResult(
                    url=url,
                    success=True,
                    content=output,
                    method_used=FetchMethod.MCP,
                )
            else:
                return FetchResult(
                    url=url,
                    success=False,
                    error=error_output or "MCP 获取失败",
                    method_used=FetchMethod.MCP,
                )

        except asyncio.TimeoutError:
            return FetchResult(
                url=url,
                success=False,
                error=f"MCP 获取超时 ({timeout}s)",
                method_used=FetchMethod.MCP,
            )
        except Exception as e:
            return FetchResult(
                url=url,
                success=False,
                error=f"MCP 获取异常: {e}",
                method_used=FetchMethod.MCP,
            )

    async def _fetch_via_playwright(self, url: str, timeout: int) -> FetchResult:
        """通过 Playwright 获取网页（支持 JS 渲染）

        Playwright 可以完整渲染 JavaScript 生成的内容，适用于 SPA 等动态页面。

        Args:
            url: 目标 URL
            timeout: 超时时间（秒）

        Returns:
            获取结果
        """
        try:
            from playwright.async_api import TimeoutError as PlaywrightTimeout
            from playwright.async_api import async_playwright
        except ImportError:
            return FetchResult(
                url=url,
                success=False,
                error="Playwright 未安装，请运行: pip install playwright && playwright install chromium",
                method_used=FetchMethod.PLAYWRIGHT,
            )

        try:
            async with async_playwright() as p:
                # 启动浏览器
                browser = await p.chromium.launch(
                    headless=self.config.playwright_headless,
                )

                # 创建上下文和页面
                context = await browser.new_context(
                    user_agent=self.config.user_agent,
                    java_script_enabled=self.config.playwright_js_enabled,
                )
                page = await context.new_page()

                # 设置页面超时
                page.set_default_timeout(timeout * 1000)  # 转换为毫秒

                try:
                    # 导航到页面
                    await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)

                    # 等待指定选择器（如果配置了）
                    if self.config.playwright_wait_for_selector:
                        await page.wait_for_selector(
                            self.config.playwright_wait_for_selector,
                            timeout=timeout * 1000,
                        )

                    # 滚动到页面底部（加载懒加载内容）
                    if self.config.playwright_scroll_to_bottom:
                        await self._scroll_to_bottom(page)

                    # 页面加载后额外等待
                    if self.config.playwright_wait_after_load > 0:
                        await asyncio.sleep(self.config.playwright_wait_after_load)

                    # 获取页面内容
                    if self.config.content_format == ContentFormat.HTML:
                        content = await page.content()
                    else:
                        # 获取纯文本或转换为文本
                        content = await page.evaluate("() => document.body.innerText")

                    # 获取页面标题
                    title = await page.title()

                    return FetchResult(
                        url=url,
                        success=True,
                        content=content,
                        method_used=FetchMethod.PLAYWRIGHT,
                        metadata={"title": title},
                    )

                finally:
                    await context.close()
                    await browser.close()

        except PlaywrightTimeout:
            return FetchResult(
                url=url,
                success=False,
                error=f"Playwright 页面加载超时 ({timeout}s)",
                method_used=FetchMethod.PLAYWRIGHT,
            )
        except Exception as e:
            return FetchResult(
                url=url,
                success=False,
                error=f"Playwright 获取异常: {e}",
                method_used=FetchMethod.PLAYWRIGHT,
            )

    async def _scroll_to_bottom(self, page) -> None:
        """滚动页面到底部以加载懒加载内容

        Args:
            page: Playwright Page 对象
        """
        try:
            # 获取页面高度并逐步滚动
            await page.evaluate("""
                async () => {
                    await new Promise((resolve) => {
                        let totalHeight = 0;
                        const distance = 300;
                        const timer = setInterval(() => {
                            const scrollHeight = document.body.scrollHeight;
                            window.scrollBy(0, distance);
                            totalHeight += distance;

                            if (totalHeight >= scrollHeight) {
                                clearInterval(timer);
                                resolve();
                            }
                        }, 100);

                        // 最多滚动 10 秒
                        setTimeout(() => {
                            clearInterval(timer);
                            resolve();
                        }, 10000);
                    });
                }
            """)

            # 等待可能的懒加载完成
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.warning(f"滚动页面时出错: {e}")

    async def _fetch_via_curl(self, url: str, timeout: int) -> FetchResult:
        """通过 curl 获取网页"""
        try:
            cmd = [
                "curl",
                "-s",  # 静默模式
                "-L",  # 跟随重定向
                "-m",
                str(timeout),  # 超时
                "-A",
                self.config.user_agent,  # 用户代理
                "-w",
                "\n%{http_code}",  # 输出状态码
                url,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout + 5,  # 额外缓冲
            )

            output = stdout.decode("utf-8", errors="replace")

            # 提取状态码（最后一行）
            lines = output.rsplit("\n", 1)
            if len(lines) == 2:
                content = lines[0]
                try:
                    status_code = int(lines[1].strip())
                except ValueError:
                    status_code = None
                    content = output
            else:
                content = output
                status_code = None

            success = process.returncode == 0 and (status_code is None or 200 <= status_code < 400)

            # 如果需要纯文本格式，转换 HTML
            if success and self.config.content_format == ContentFormat.TEXT:
                content = self._html_to_text(content)

            return FetchResult(
                url=url,
                success=success,
                content=content,
                status_code=status_code,
                error=None if success else f"HTTP {status_code}" if status_code else "curl 执行失败",
                method_used=FetchMethod.CURL,
            )

        except asyncio.TimeoutError:
            return FetchResult(
                url=url,
                success=False,
                error=f"curl 超时 ({timeout}s)",
                method_used=FetchMethod.CURL,
            )
        except Exception as e:
            return FetchResult(
                url=url,
                success=False,
                error=f"curl 异常: {e}",
                method_used=FetchMethod.CURL,
            )

    async def _fetch_via_lynx(self, url: str, timeout: int) -> FetchResult:
        """通过 lynx 获取网页（自动转为纯文本）"""
        try:
            cmd = [
                "lynx",
                "-dump",  # 输出纯文本
                "-nolist",  # 不输出链接列表
                "-connect_timeout",
                str(timeout),
                url,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout + 5,
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if process.returncode == 0:
                return FetchResult(
                    url=url,
                    success=True,
                    content=output,
                    method_used=FetchMethod.LYNX,
                )
            else:
                return FetchResult(
                    url=url,
                    success=False,
                    error=error_output or "lynx 执行失败",
                    method_used=FetchMethod.LYNX,
                )

        except asyncio.TimeoutError:
            return FetchResult(
                url=url,
                success=False,
                error=f"lynx 超时 ({timeout}s)",
                method_used=FetchMethod.LYNX,
            )
        except Exception as e:
            return FetchResult(
                url=url,
                success=False,
                error=f"lynx 异常: {e}",
                method_used=FetchMethod.LYNX,
            )

    def _html_to_text(self, html: str) -> str:
        """简单的 HTML 转纯文本

        注意: 这是一个简化实现，复杂场景建议使用 lynx 或专门的库
        """
        import re

        # 移除 script 和 style 标签
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # 移除 HTML 标签
        text = re.sub(r"<[^>]+>", " ", text)

        # 解码常见 HTML 实体
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")

        # 清理空白字符
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def get_available_methods(self) -> list[FetchMethod]:
        """获取可用的获取方式"""
        return self._available_methods.copy()

    def check_url_valid(self, url: str) -> bool:
        """检查 URL 是否有效"""
        import re

        pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        return bool(re.match(pattern, url, re.IGNORECASE))


async def fetch_url(
    url: str,
    method: FetchMethod = FetchMethod.AUTO,
    timeout: int = 30,
) -> FetchResult:
    """便捷函数：获取单个 URL

    Args:
        url: 目标 URL
        method: 获取方式
        timeout: 超时时间

    Returns:
        获取结果

    示例:
    ```python
    result = await fetch_url("https://example.com")
    if result.success:
        print(result.content)
    ```
    """
    config = FetchConfig(method=method, timeout=timeout)
    fetcher = WebFetcher(config)
    return await fetcher.fetch(url)


async def fetch_urls(
    urls: list[str],
    max_concurrent: int = 5,
    timeout: int = 30,
) -> list[FetchResult]:
    """便捷函数：并发获取多个 URL

    Args:
        urls: URL 列表
        max_concurrent: 并发限制
        timeout: 每个请求的超时时间

    Returns:
        获取结果列表

    示例:
    ```python
    urls = ["https://example.com", "https://google.com"]
    results = await fetch_urls(urls)
    for result in results:
        print(f"{result.url}: {'成功' if result.success else '失败'}")
    ```
    """
    config = FetchConfig(max_concurrent=max_concurrent, timeout=timeout)
    fetcher = WebFetcher(config)
    return await fetcher.fetch_many(urls)
