"""
Cursor Cloud Agent 网络配置模块

提供出口 IP 范围管理、验证和防火墙规则导出功能。
"""

from __future__ import annotations

import ipaddress
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

from core.config import CloudAgentConfig, get_config


class FirewallFormat(Enum):
    """防火墙规则导出格式"""

    IPTABLES = "iptables"
    NGINX = "nginx"
    APACHE = "apache"
    UFW = "ufw"
    CLOUDFLARE = "cloudflare"
    JSON = "json"


@dataclass
class EgressIPConfig:
    """
    出口 IP 配置类

    存储和管理 Cursor Cloud Agent 的出口 IP 范围列表。
    """

    ip_ranges: list[str] = field(default_factory=list)
    last_updated: float = 0.0
    source: str = ""
    version: str = ""

    # 缓存配置
    cache_file: Path | None = None
    cache_ttl: int = 3600  # 默认 1 小时

    def __post_init__(self):
        """初始化后处理：解析 IP 网络"""
        self._networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self._parse_networks()

    def _parse_networks(self) -> None:
        """解析 IP 范围为网络对象"""
        self._networks = []
        for ip_range in self.ip_ranges:
            try:
                network = ipaddress.ip_network(ip_range, strict=False)
                self._networks.append(network)
            except ValueError as e:
                # 跳过无效的 IP 范围
                print(f"警告: 无效的 IP 范围 '{ip_range}': {e}")

    @property
    def networks(self) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
        """获取已解析的网络对象列表"""
        return self._networks

    def is_allowed_ip(self, ip: str) -> bool:
        """
        验证 IP 是否在允许范围内

        Args:
            ip: 要验证的 IP 地址

        Returns:
            bool: IP 是否在允许的范围内
        """
        try:
            ip_addr = ipaddress.ip_address(ip)
            return any(ip_addr in network for network in self._networks)
        except ValueError:
            return False

    def is_cache_valid(self) -> bool:
        """检查缓存是否仍然有效"""
        if not self.last_updated:
            return False
        return (time.time() - self.last_updated) < self.cache_ttl

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "ip_ranges": self.ip_ranges,
            "last_updated": self.last_updated,
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict, cache_file: Path | None = None, cache_ttl: int = 3600) -> EgressIPConfig:
        """从字典创建实例"""
        return cls(
            ip_ranges=data.get("ip_ranges", []),
            last_updated=data.get("last_updated", 0.0),
            source=data.get("source", ""),
            version=data.get("version", ""),
            cache_file=cache_file,
            cache_ttl=cache_ttl,
        )

    def save_cache(self) -> bool:
        """保存到本地缓存文件"""
        if not self.cache_file:
            return False
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except OSError as e:
            print(f"警告: 无法保存缓存文件: {e}")
            return False

    @classmethod
    def load_cache(cls, cache_file: Path, cache_ttl: int = 3600) -> EgressIPConfig | None:
        """从本地缓存文件加载"""
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            config = cls.from_dict(data, cache_file=cache_file, cache_ttl=cache_ttl)
            if config.is_cache_valid():
                return config
            return None
        except (OSError, json.JSONDecodeError) as e:
            print(f"警告: 无法加载缓存文件: {e}")
            return None

    def export_firewall_rules(self, format: FirewallFormat, **kwargs) -> str:
        """
        导出为防火墙规则格式

        Args:
            format: 防火墙格式类型
            **kwargs: 额外参数
                - chain: iptables 链名（默认 INPUT）
                - port: 目标端口（可选）
                - comment: 规则注释

        Returns:
            str: 格式化的防火墙规则
        """
        if format == FirewallFormat.IPTABLES:
            return self._export_iptables(**kwargs)
        elif format == FirewallFormat.NGINX:
            return self._export_nginx(**kwargs)
        elif format == FirewallFormat.APACHE:
            return self._export_apache(**kwargs)
        elif format == FirewallFormat.UFW:
            return self._export_ufw(**kwargs)
        elif format == FirewallFormat.CLOUDFLARE:
            return self._export_cloudflare(**kwargs)
        elif format == FirewallFormat.JSON:
            return json.dumps(self.ip_ranges, indent=2)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def _export_iptables(
        self, chain: str = "INPUT", port: int | None = None, comment: str = "Cursor Cloud Agent"
    ) -> str:
        """导出为 iptables 格式"""
        lines = [f"# {comment}", f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

        port_spec = f"-p tcp --dport {port} " if port else ""

        for ip_range in self.ip_ranges:
            lines.append(f'iptables -A {chain} -s {ip_range} {port_spec}-j ACCEPT -m comment --comment "{comment}"')

        return "\n".join(lines)

    def _export_nginx(self, **kwargs) -> str:
        """导出为 nginx allow 格式"""
        lines = [
            "# Cursor Cloud Agent Egress IPs",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        for ip_range in self.ip_ranges:
            lines.append(f"allow {ip_range};")

        lines.append("")
        lines.append("# deny all;  # 取消注释以拒绝其他 IP")

        return "\n".join(lines)

    def _export_apache(self, **kwargs) -> str:
        """导出为 Apache .htaccess 格式"""
        lines = [
            "# Cursor Cloud Agent Egress IPs",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "<RequireAny>",
        ]

        for ip_range in self.ip_ranges:
            lines.append(f"    Require ip {ip_range}")

        lines.append("</RequireAny>")

        return "\n".join(lines)

    def _export_ufw(self, port: int | None = None, comment: str = "Cursor Cloud Agent", **kwargs) -> str:
        """导出为 UFW 格式"""
        lines = [
            f"# {comment}",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        port_spec = f" to any port {port}" if port else ""

        for ip_range in self.ip_ranges:
            lines.append(f"ufw allow from {ip_range}{port_spec} comment '{comment}'")

        return "\n".join(lines)

    def _export_cloudflare(self, **kwargs) -> str:
        """导出为 Cloudflare IP Access Rules 格式（JSON）"""
        rules = []
        for ip_range in self.ip_ranges:
            rules.append(
                {
                    "mode": "whitelist",
                    "configuration": {
                        "target": "ip_range",
                        "value": ip_range,
                    },
                    "notes": "Cursor Cloud Agent",
                }
            )

        return json.dumps(rules, indent=2)


class EgressIPManager:
    """
    出口 IP 管理器

    负责获取、缓存和验证 Cursor Cloud Agent 的出口 IP 范围。

    配置来源说明:
        TTL 配置统一从 core.config.get_config().cloud_agent.egress_ip_cache_ttl 获取，
        避免模块内自行读取 YAML 配置文件，确保配置来源一致性。

    初始化方式:
        1. 默认: 自动从 core.config 获取 TTL
           manager = EgressIPManager()

        2. 传入配置对象: 使用指定的 CloudAgentConfig
           manager = EgressIPManager(cloud_config=config.cloud_agent)

        3. 传入数值: 显式指定 TTL（用于测试）
           manager = EgressIPManager(cache_ttl=7200)
    """

    # Cursor API 端点（假设的 API URL）
    CURSOR_API_URL = "https://api.cursor.com/v1/egress-ips"
    CURSOR_FALLBACK_URL = "https://cursor.com/.well-known/egress-ips.json"

    # 已知的 Cursor Cloud Agent 出口 IP 范围（备用）
    KNOWN_IP_RANGES = [
        # Cursor Cloud 基础设施（示例，实际需从 API 获取）
        "35.192.0.0/12",  # Google Cloud
        "34.64.0.0/10",  # Google Cloud
        "104.196.0.0/14",  # Google Cloud
        "35.186.0.0/16",  # Google Cloud
    ]

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl: int | None = None,
        cloud_config: CloudAgentConfig | None = None,
    ):
        """
        初始化管理器

        Args:
            cache_dir: 缓存目录路径
            cache_ttl: 缓存有效期（秒），显式指定时覆盖配置值（主要用于测试）
            cloud_config: CloudAgentConfig 配置对象，如不提供则从 get_config() 获取

        配置优先级:
            1. cache_ttl 参数（最高，显式指定）
            2. cloud_config.egress_ip_cache_ttl
            3. get_config().cloud_agent.egress_ip_cache_ttl（默认）
        """
        # 获取 TTL 配置（统一来源）
        if cache_ttl is not None:
            # 显式指定 TTL（主要用于测试场景）
            self.cache_ttl = cache_ttl
        elif cloud_config is not None:
            # 使用传入的配置对象
            self.cache_ttl = cloud_config.egress_ip_cache_ttl
        else:
            # 从 core.config 单例获取（默认行为）
            self.cache_ttl = get_config().cloud_agent.egress_ip_cache_ttl

        # 设置缓存目录
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = Path.home() / ".cursor" / "cache"

        self.cache_file = self.cache_dir / "egress_ips.json"

        self._config: EgressIPConfig | None = None

    async def fetch_egress_ip_ranges(self, force_refresh: bool = False) -> EgressIPConfig:
        """
        从 Cursor API 获取最新的出口 IP 范围

        Args:
            force_refresh: 是否强制刷新（忽略缓存）

        Returns:
            EgressIPConfig: IP 配置对象
        """
        # 检查缓存
        if not force_refresh:
            cached = self._load_from_cache()
            if cached:
                self._config = cached
                return cached

        # 尝试从 API 获取
        config = await self._fetch_from_api()

        if config:
            self._config = config
            config.save_cache()
            return config

        # 使用备用 IP 范围
        return self._get_fallback_config()

    async def _fetch_from_api(self) -> EgressIPConfig | None:
        """从 Cursor API 获取 IP 范围"""
        urls = [self.CURSOR_API_URL, self.CURSOR_FALLBACK_URL]

        for url in urls:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        return EgressIPConfig(
                            ip_ranges=data.get("ip_ranges", data.get("ranges", [])),
                            last_updated=time.time(),
                            source=url,
                            version=data.get("version", ""),
                            cache_file=self.cache_file,
                            cache_ttl=self.cache_ttl,
                        )
            except Exception as e:
                print(f"警告: 无法从 {url} 获取 IP 范围: {e}")
                continue

        return None

    def _load_from_cache(self) -> EgressIPConfig | None:
        """从本地缓存加载"""
        return EgressIPConfig.load_cache(self.cache_file, self.cache_ttl)

    def _get_fallback_config(self) -> EgressIPConfig:
        """获取备用配置"""
        return EgressIPConfig(
            ip_ranges=self.KNOWN_IP_RANGES,
            last_updated=time.time(),
            source="fallback",
            version="builtin",
            cache_file=self.cache_file,
            cache_ttl=self.cache_ttl,
        )

    def is_allowed_ip(self, ip: str) -> bool:
        """
        验证 IP 是否在允许范围内

        Args:
            ip: 要验证的 IP 地址

        Returns:
            bool: IP 是否允许
        """
        if not self._config:
            # 如果没有配置，使用备用配置
            self._config = self._get_fallback_config()

        return self._config.is_allowed_ip(ip)

    def get_config(self) -> EgressIPConfig | None:
        """获取当前配置"""
        return self._config

    def export_rules(self, format: FirewallFormat, **kwargs) -> str:
        """
        导出防火墙规则

        Args:
            format: 防火墙格式
            **kwargs: 额外参数

        Returns:
            str: 格式化的规则
        """
        if not self._config:
            self._config = self._get_fallback_config()

        return self._config.export_firewall_rules(format, **kwargs)


# 全局实例
_manager: EgressIPManager | None = None


def get_manager(
    cache_dir: Path | None = None,
    cache_ttl: int | None = None,
    cloud_config: CloudAgentConfig | None = None,
) -> EgressIPManager:
    """
    获取全局 IP 管理器实例

    Args:
        cache_dir: 缓存目录
        cache_ttl: 缓存有效期（秒），显式指定时覆盖配置值
        cloud_config: CloudAgentConfig 配置对象

    Returns:
        EgressIPManager: 管理器实例

    Note:
        TTL 配置默认从 core.config.get_config().cloud_agent.egress_ip_cache_ttl 获取，
        确保与项目配置系统保持一致。
    """
    global _manager
    if _manager is None:
        _manager = EgressIPManager(
            cache_dir=cache_dir,
            cache_ttl=cache_ttl,
            cloud_config=cloud_config,
        )
    return _manager


async def fetch_egress_ip_ranges(force_refresh: bool = False) -> EgressIPConfig:
    """
    获取 Cursor Cloud Agent 出口 IP 范围

    Args:
        force_refresh: 是否强制刷新

    Returns:
        EgressIPConfig: IP 配置
    """
    manager = get_manager()
    return await manager.fetch_egress_ip_ranges(force_refresh)


def is_allowed_ip(ip: str) -> bool:
    """
    验证 IP 是否在 Cursor Cloud Agent 允许范围内

    Args:
        ip: 要验证的 IP 地址

    Returns:
        bool: 是否允许
    """
    manager = get_manager()
    return manager.is_allowed_ip(ip)


def export_firewall_rules(format: FirewallFormat, **kwargs) -> str:
    """
    导出防火墙规则

    Args:
        format: 防火墙格式
        **kwargs: 额外参数

    Returns:
        str: 格式化的规则
    """
    manager = get_manager()
    return manager.export_rules(format, **kwargs)


# 便捷常量导出
IPTABLES = FirewallFormat.IPTABLES
NGINX = FirewallFormat.NGINX
APACHE = FirewallFormat.APACHE
UFW = FirewallFormat.UFW
CLOUDFLARE = FirewallFormat.CLOUDFLARE
JSON = FirewallFormat.JSON
