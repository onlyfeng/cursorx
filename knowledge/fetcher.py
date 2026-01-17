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
"""
import asyncio
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from loguru import logger

# 全局缓存：避免每个 WebFetcher 实例都做一遍外部依赖探测（会启动子进程，较慢）
_AVAILABLE_METHODS_CACHE: Optional[list["FetchMethod"]] = None
_AVAILABLE_METHODS_LOCK = asyncio.Lock()


class FetchMethod(str, Enum):
    """获取方式"""
    MCP = "mcp"              # MCP fetch（通过 agent CLI）
    CURL = "curl"            # curl 命令
    LYNX = "lynx"            # lynx 命令（纯文本）
    PLAYWRIGHT = "playwright"  # Playwright 浏览器自动化（支持 JS 渲染）
    AUTO = "auto"            # 自动选择


class ContentFormat(str, Enum):
    """内容格式"""
    TEXT = "text"        # 纯文本
    HTML = "html"        # 原始 HTML
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
    playwright_headless: bool = True          # 无头模式
    playwright_wait_for_selector: Optional[str] = None  # 等待指定选择器出现
    playwright_scroll_to_bottom: bool = False  # 滚动到页面底部（加载懒加载内容）
    playwright_wait_after_load: float = 0.0    # 页面加载后额外等待时间（秒）
    playwright_js_enabled: bool = True         # 是否启用 JavaScript


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
    
    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)


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
                self.config.agent_path, "mcp", "list",
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
                "python", "-c", 
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
        
        method = method or self.config.method
        timeout = timeout or self.config.timeout
        
        # 带重试的获取
        return await self._fetch_with_retry(url, method, timeout)
    
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
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
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
            "/#/",              # Hash 路由
            "/app/",            # 应用路径
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
                "-p", prompt,
                "--output-format", "text",
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
            from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
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
                "-s",                           # 静默模式
                "-L",                           # 跟随重定向
                "-m", str(timeout),             # 超时
                "-A", self.config.user_agent,   # 用户代理
                "-w", "\n%{http_code}",         # 输出状态码
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
                "-dump",                        # 输出纯文本
                "-nolist",                      # 不输出链接列表
                "-connect_timeout", str(timeout),
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
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 解码常见 HTML 实体
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        # 清理空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def get_available_methods(self) -> list[FetchMethod]:
        """获取可用的获取方式"""
        return self._available_methods.copy()
    
    def check_url_valid(self, url: str) -> bool:
        """检查 URL 是否有效"""
        import re
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
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
