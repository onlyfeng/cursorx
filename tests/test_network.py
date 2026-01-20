"""Network 模块测试

测试内容:
1. EgressIPConfig - IP 配置和验证
2. EgressIPManager - 网络请求、超时处理、重试逻辑
3. 防火墙规则导出
4. 缓存机制
5. 使用 respx 进行 HTTP mock 测试
"""
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import pytest
import respx

from cursor.network import (
    APACHE,
    CLOUDFLARE,
    IPTABLES,
    JSON,
    NGINX,
    UFW,
    EgressIPConfig,
    EgressIPManager,
    FirewallFormat,
    export_firewall_rules,
    fetch_egress_ip_ranges,
    get_manager,
    is_allowed_ip,
)


# ==================== Fixtures ====================


@pytest.fixture
def sample_ip_ranges():
    """示例 IP 范围列表"""
    return [
        "35.192.0.0/12",
        "34.64.0.0/10",
        "104.196.0.0/14",
        "2001:db8::/32",  # IPv6
    ]


@pytest.fixture
def egress_config(sample_ip_ranges):
    """创建 EgressIPConfig 实例"""
    return EgressIPConfig(
        ip_ranges=sample_ip_ranges,
        last_updated=time.time(),
        source="test",
        version="1.0.0",
    )


@pytest.fixture
def temp_cache_dir():
    """临时缓存目录"""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def egress_manager(temp_cache_dir):
    """创建 EgressIPManager 实例"""
    return EgressIPManager(
        cache_dir=temp_cache_dir,
        cache_ttl=3600,
    )


@pytest.fixture
def api_response_data(sample_ip_ranges):
    """模拟 API 响应数据"""
    return {
        "ip_ranges": sample_ip_ranges,
        "version": "2.0.0",
    }


# ==================== FirewallFormat 测试 ====================


class TestFirewallFormat:
    """FirewallFormat 枚举测试"""

    def test_format_values(self):
        """格式值"""
        assert FirewallFormat.IPTABLES.value == "iptables"
        assert FirewallFormat.NGINX.value == "nginx"
        assert FirewallFormat.APACHE.value == "apache"
        assert FirewallFormat.UFW.value == "ufw"
        assert FirewallFormat.CLOUDFLARE.value == "cloudflare"
        assert FirewallFormat.JSON.value == "json"

    def test_convenience_constants(self):
        """便捷常量"""
        assert IPTABLES == FirewallFormat.IPTABLES
        assert NGINX == FirewallFormat.NGINX
        assert APACHE == FirewallFormat.APACHE
        assert UFW == FirewallFormat.UFW
        assert CLOUDFLARE == FirewallFormat.CLOUDFLARE
        assert JSON == FirewallFormat.JSON


# ==================== EgressIPConfig 测试 ====================


class TestEgressIPConfig:
    """EgressIPConfig 测试"""

    def test_init_default(self):
        """默认初始化"""
        config = EgressIPConfig()
        assert config.ip_ranges == []
        assert config.last_updated == 0.0
        assert config.source == ""
        assert config.version == ""
        assert len(config.networks) == 0

    def test_init_with_ranges(self, sample_ip_ranges):
        """使用 IP 范围初始化"""
        config = EgressIPConfig(ip_ranges=sample_ip_ranges)
        assert len(config.ip_ranges) == 4
        assert len(config.networks) == 4

    def test_parse_networks_ipv4(self):
        """解析 IPv4 网络"""
        config = EgressIPConfig(ip_ranges=["192.168.1.0/24", "10.0.0.0/8"])
        assert len(config.networks) == 2

    def test_parse_networks_ipv6(self):
        """解析 IPv6 网络"""
        config = EgressIPConfig(ip_ranges=["2001:db8::/32"])
        assert len(config.networks) == 1

    def test_parse_networks_invalid(self, capsys):
        """跳过无效 IP 范围"""
        config = EgressIPConfig(ip_ranges=["invalid-ip", "192.168.1.0/24"])
        assert len(config.networks) == 1
        captured = capsys.readouterr()
        assert "无效的 IP 范围" in captured.out

    def test_is_allowed_ip_ipv4_in_range(self, egress_config):
        """IPv4 在范围内"""
        # 35.192.0.0/12 包含 35.192.0.1 到 35.207.255.255
        assert egress_config.is_allowed_ip("35.192.0.1") is True
        assert egress_config.is_allowed_ip("35.200.100.50") is True

    def test_is_allowed_ip_ipv4_out_of_range(self, egress_config):
        """IPv4 不在范围内"""
        assert egress_config.is_allowed_ip("8.8.8.8") is False
        assert egress_config.is_allowed_ip("192.168.1.1") is False

    def test_is_allowed_ip_ipv6_in_range(self, egress_config):
        """IPv6 在范围内"""
        # 2001:db8::/32 包含 2001:db8:开头的地址
        assert egress_config.is_allowed_ip("2001:db8::1") is True
        assert egress_config.is_allowed_ip("2001:db8:abcd::1234") is True

    def test_is_allowed_ip_ipv6_out_of_range(self, egress_config):
        """IPv6 不在范围内"""
        assert egress_config.is_allowed_ip("2001:db9::1") is False

    def test_is_allowed_ip_invalid(self, egress_config):
        """无效 IP 返回 False"""
        assert egress_config.is_allowed_ip("not-an-ip") is False
        assert egress_config.is_allowed_ip("") is False

    def test_is_cache_valid_no_timestamp(self):
        """无时间戳时缓存无效"""
        config = EgressIPConfig()
        assert config.is_cache_valid() is False

    def test_is_cache_valid_fresh(self):
        """新缓存有效"""
        config = EgressIPConfig(
            ip_ranges=["192.168.1.0/24"],
            last_updated=time.time(),
            cache_ttl=3600,
        )
        assert config.is_cache_valid() is True

    def test_is_cache_valid_expired(self):
        """过期缓存无效"""
        config = EgressIPConfig(
            ip_ranges=["192.168.1.0/24"],
            last_updated=time.time() - 7200,  # 2 小时前
            cache_ttl=3600,  # 1 小时有效
        )
        assert config.is_cache_valid() is False

    def test_to_dict(self, egress_config):
        """转换为字典"""
        data = egress_config.to_dict()
        assert "ip_ranges" in data
        assert "last_updated" in data
        assert "source" in data
        assert "version" in data
        assert data["source"] == "test"
        assert data["version"] == "1.0.0"

    def test_from_dict(self, sample_ip_ranges):
        """从字典创建"""
        data = {
            "ip_ranges": sample_ip_ranges,
            "last_updated": time.time(),
            "source": "api",
            "version": "2.0.0",
        }
        config = EgressIPConfig.from_dict(data)
        assert len(config.ip_ranges) == 4
        assert config.source == "api"
        assert config.version == "2.0.0"

    def test_from_dict_with_cache_config(self, temp_cache_dir):
        """从字典创建时带缓存配置"""
        cache_file = temp_cache_dir / "test_cache.json"
        data = {"ip_ranges": ["192.168.1.0/24"]}
        config = EgressIPConfig.from_dict(data, cache_file=cache_file, cache_ttl=7200)
        assert config.cache_file == cache_file
        assert config.cache_ttl == 7200


# ==================== EgressIPConfig 缓存测试 ====================


class TestEgressIPConfigCache:
    """EgressIPConfig 缓存功能测试"""

    def test_save_cache(self, temp_cache_dir, sample_ip_ranges):
        """保存缓存文件"""
        cache_file = temp_cache_dir / "cache" / "egress_ips.json"
        config = EgressIPConfig(
            ip_ranges=sample_ip_ranges,
            last_updated=time.time(),
            source="test",
            cache_file=cache_file,
        )
        result = config.save_cache()
        assert result is True
        assert cache_file.exists()

        # 验证文件内容
        with open(cache_file) as f:
            saved_data = json.load(f)
        assert saved_data["ip_ranges"] == sample_ip_ranges
        assert saved_data["source"] == "test"

    def test_save_cache_no_cache_file(self, sample_ip_ranges):
        """无缓存文件时保存失败"""
        config = EgressIPConfig(ip_ranges=sample_ip_ranges, cache_file=None)
        result = config.save_cache()
        assert result is False

    def test_load_cache_success(self, temp_cache_dir, sample_ip_ranges):
        """加载缓存成功"""
        cache_file = temp_cache_dir / "egress_ips.json"
        cache_data = {
            "ip_ranges": sample_ip_ranges,
            "last_updated": time.time(),
            "source": "cached",
            "version": "1.0.0",
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        config = EgressIPConfig.load_cache(cache_file, cache_ttl=3600)
        assert config is not None
        assert len(config.ip_ranges) == 4
        assert config.source == "cached"

    def test_load_cache_not_exists(self, temp_cache_dir):
        """缓存文件不存在"""
        cache_file = temp_cache_dir / "nonexistent.json"
        config = EgressIPConfig.load_cache(cache_file)
        assert config is None

    def test_load_cache_expired(self, temp_cache_dir, sample_ip_ranges):
        """加载过期缓存返回 None"""
        cache_file = temp_cache_dir / "egress_ips.json"
        cache_data = {
            "ip_ranges": sample_ip_ranges,
            "last_updated": time.time() - 7200,  # 2 小时前
            "source": "cached",
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        config = EgressIPConfig.load_cache(cache_file, cache_ttl=3600)
        assert config is None

    def test_load_cache_invalid_json(self, temp_cache_dir, capsys):
        """加载无效 JSON"""
        cache_file = temp_cache_dir / "invalid.json"
        with open(cache_file, "w") as f:
            f.write("not valid json {")

        config = EgressIPConfig.load_cache(cache_file)
        assert config is None
        captured = capsys.readouterr()
        assert "无法加载缓存文件" in captured.out


# ==================== EgressIPConfig 防火墙规则导出测试 ====================


class TestFirewallExport:
    """防火墙规则导出测试"""

    def test_export_iptables(self, egress_config):
        """导出 iptables 规则"""
        rules = egress_config.export_firewall_rules(FirewallFormat.IPTABLES)
        assert "iptables -A INPUT" in rules
        assert "35.192.0.0/12" in rules
        assert "-j ACCEPT" in rules
        assert "Cursor Cloud Agent" in rules

    def test_export_iptables_with_port(self, egress_config):
        """导出 iptables 规则带端口"""
        rules = egress_config.export_firewall_rules(
            FirewallFormat.IPTABLES, port=443
        )
        assert "--dport 443" in rules

    def test_export_iptables_custom_chain(self, egress_config):
        """导出 iptables 规则自定义链"""
        rules = egress_config.export_firewall_rules(
            FirewallFormat.IPTABLES, chain="FORWARD"
        )
        assert "iptables -A FORWARD" in rules

    def test_export_nginx(self, egress_config):
        """导出 nginx 规则"""
        rules = egress_config.export_firewall_rules(FirewallFormat.NGINX)
        assert "allow 35.192.0.0/12;" in rules
        assert "Cursor Cloud Agent Egress IPs" in rules
        assert "deny all;" in rules

    def test_export_apache(self, egress_config):
        """导出 Apache 规则"""
        rules = egress_config.export_firewall_rules(FirewallFormat.APACHE)
        assert "<RequireAny>" in rules
        assert "Require ip 35.192.0.0/12" in rules
        assert "</RequireAny>" in rules

    def test_export_ufw(self, egress_config):
        """导出 UFW 规则"""
        rules = egress_config.export_firewall_rules(FirewallFormat.UFW)
        assert "ufw allow from 35.192.0.0/12" in rules
        assert "Cursor Cloud Agent" in rules

    def test_export_ufw_with_port(self, egress_config):
        """导出 UFW 规则带端口"""
        rules = egress_config.export_firewall_rules(
            FirewallFormat.UFW, port=8080
        )
        assert "to any port 8080" in rules

    def test_export_cloudflare(self, egress_config):
        """导出 Cloudflare 规则"""
        rules = egress_config.export_firewall_rules(FirewallFormat.CLOUDFLARE)
        data = json.loads(rules)
        assert isinstance(data, list)
        assert len(data) == 4
        assert data[0]["mode"] == "whitelist"
        assert data[0]["configuration"]["target"] == "ip_range"

    def test_export_json(self, egress_config):
        """导出 JSON 格式"""
        rules = egress_config.export_firewall_rules(FirewallFormat.JSON)
        data = json.loads(rules)
        assert isinstance(data, list)
        assert "35.192.0.0/12" in data

    def test_export_invalid_format(self, egress_config):
        """无效格式抛出异常"""
        with pytest.raises(ValueError, match="不支持的格式"):
            egress_config.export_firewall_rules("invalid")


# ==================== EgressIPManager 测试 ====================


class TestEgressIPManager:
    """EgressIPManager 测试"""

    def test_init_default(self, temp_cache_dir):
        """默认初始化"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)
        assert manager.cache_ttl == 3600
        assert manager.cache_file == temp_cache_dir / "egress_ips.json"

    def test_init_with_cache_ttl(self, temp_cache_dir):
        """自定义缓存 TTL"""
        manager = EgressIPManager(cache_dir=temp_cache_dir, cache_ttl=7200)
        assert manager.cache_ttl == 7200

    def test_get_fallback_config(self, egress_manager):
        """获取备用配置"""
        config = egress_manager._get_fallback_config()
        assert len(config.ip_ranges) > 0
        assert config.source == "fallback"
        assert config.version == "builtin"

    def test_is_allowed_ip_with_fallback(self, egress_manager):
        """使用备用配置验证 IP"""
        # 35.192.0.0/12 是备用范围之一
        assert egress_manager.is_allowed_ip("35.192.0.1") is True
        assert egress_manager.is_allowed_ip("8.8.8.8") is False


# ==================== EgressIPManager 网络请求测试 (使用 respx) ====================


class TestEgressIPManagerNetwork:
    """EgressIPManager 网络请求测试 (使用 respx mock)"""

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_from_api_success(self, temp_cache_dir, api_response_data):
        """成功从 API 获取 IP 范围"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        # Mock API 响应
        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(200, json=api_response_data)
        )

        config = await manager._fetch_from_api()
        assert config is not None
        assert len(config.ip_ranges) == 4
        assert config.source == EgressIPManager.CURSOR_API_URL
        assert config.version == "2.0.0"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_from_api_fallback_url(self, temp_cache_dir, api_response_data):
        """主 URL 失败时使用备用 URL"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        # 主 URL 返回 500
        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(500)
        )
        # 备用 URL 返回成功
        respx.get(EgressIPManager.CURSOR_FALLBACK_URL).mock(
            return_value=httpx.Response(200, json=api_response_data)
        )

        config = await manager._fetch_from_api()
        assert config is not None
        assert config.source == EgressIPManager.CURSOR_FALLBACK_URL

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_from_api_all_fail(self, temp_cache_dir):
        """所有 URL 都失败（HTTP 错误码）"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(500)
        )
        respx.get(EgressIPManager.CURSOR_FALLBACK_URL).mock(
            return_value=httpx.Response(404)
        )

        config = await manager._fetch_from_api()
        # 所有 URL 返回错误状态码时，应返回 None
        assert config is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_from_api_timeout(self, temp_cache_dir, capsys):
        """请求超时处理"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        # 模拟超时
        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        respx.get(EgressIPManager.CURSOR_FALLBACK_URL).mock(
            side_effect=httpx.TimeoutException("timeout")
        )

        config = await manager._fetch_from_api()
        assert config is None
        captured = capsys.readouterr()
        assert "无法从" in captured.out

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_from_api_network_error(self, temp_cache_dir, capsys):
        """网络错误处理"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        respx.get(EgressIPManager.CURSOR_FALLBACK_URL).mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        config = await manager._fetch_from_api()
        assert config is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_egress_ip_ranges_from_api(self, temp_cache_dir, api_response_data):
        """完整获取流程 - 从 API"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(200, json=api_response_data)
        )

        config = await manager.fetch_egress_ip_ranges()
        assert config is not None
        assert len(config.ip_ranges) == 4
        assert manager._config == config

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_egress_ip_ranges_from_cache(self, temp_cache_dir, sample_ip_ranges):
        """完整获取流程 - 从缓存"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        # 预先创建缓存
        cache_data = {
            "ip_ranges": sample_ip_ranges,
            "last_updated": time.time(),
            "source": "cached",
            "version": "1.0.0",
        }
        manager.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(manager.cache_file, "w") as f:
            json.dump(cache_data, f)

        # 不应该发起网络请求
        config = await manager.fetch_egress_ip_ranges()
        assert config is not None
        assert config.source == "cached"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_egress_ip_ranges_force_refresh(
        self, temp_cache_dir, sample_ip_ranges, api_response_data
    ):
        """强制刷新绕过缓存"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        # 预先创建缓存
        cache_data = {
            "ip_ranges": sample_ip_ranges,
            "last_updated": time.time(),
            "source": "cached",
        }
        manager.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(manager.cache_file, "w") as f:
            json.dump(cache_data, f)

        # Mock API
        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(200, json=api_response_data)
        )

        config = await manager.fetch_egress_ip_ranges(force_refresh=True)
        assert config is not None
        assert config.source == EgressIPManager.CURSOR_API_URL

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_egress_ip_ranges_fallback(self, temp_cache_dir):
        """所有方式失败时使用备用配置"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(500)
        )
        respx.get(EgressIPManager.CURSOR_FALLBACK_URL).mock(
            return_value=httpx.Response(500)
        )

        config = await manager.fetch_egress_ip_ranges()
        assert config is not None
        assert config.source == "fallback"


# ==================== EgressIPManager 重试逻辑测试 ====================


class TestEgressIPManagerRetry:
    """EgressIPManager 重试逻辑测试"""

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_first_url_failure(self, temp_cache_dir, api_response_data):
        """第一个 URL 失败时重试第二个"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        # 第一个 URL 返回错误
        route1 = respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(503)
        )
        # 第二个 URL 成功
        route2 = respx.get(EgressIPManager.CURSOR_FALLBACK_URL).mock(
            return_value=httpx.Response(200, json=api_response_data)
        )

        config = await manager._fetch_from_api()
        assert config is not None
        assert route1.called
        assert route2.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_connection_error(self, temp_cache_dir, api_response_data):
        """连接错误时重试备用 URL"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            side_effect=httpx.ConnectError("connection failed")
        )
        respx.get(EgressIPManager.CURSOR_FALLBACK_URL).mock(
            return_value=httpx.Response(200, json=api_response_data)
        )

        config = await manager._fetch_from_api()
        assert config is not None
        assert config.source == EgressIPManager.CURSOR_FALLBACK_URL

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_read_timeout(self, temp_cache_dir, api_response_data):
        """读取超时时重试备用 URL"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            side_effect=httpx.ReadTimeout("read timeout")
        )
        respx.get(EgressIPManager.CURSOR_FALLBACK_URL).mock(
            return_value=httpx.Response(200, json=api_response_data)
        )

        config = await manager._fetch_from_api()
        assert config is not None


# ==================== EgressIPManager 配置来源测试 ====================


class TestEgressIPManagerConfig:
    """EgressIPManager 配置来源测试

    验证 TTL 配置统一从 core.config 获取，确保来源一致性。
    """

    def test_ttl_from_core_config(self, temp_cache_dir, monkeypatch):
        """TTL 默认从 core.config.get_config() 获取"""
        from core.config import ConfigManager, CloudAgentConfig

        # 重置 ConfigManager 单例
        ConfigManager.reset_instance()

        # 创建自定义配置
        custom_ttl = 5400  # 1.5 小时
        mock_cloud_config = CloudAgentConfig(egress_ip_cache_ttl=custom_ttl)

        # Mock get_config 返回自定义 TTL
        def mock_get_config():
            class MockConfig:
                cloud_agent = mock_cloud_config
            return MockConfig()

        monkeypatch.setattr("cursor.network.get_config", mock_get_config)

        # 不传入 cache_ttl，应从 core.config 获取
        manager = EgressIPManager(cache_dir=temp_cache_dir)
        assert manager.cache_ttl == custom_ttl

        # 清理
        ConfigManager.reset_instance()

    def test_ttl_from_cloud_config_object(self, temp_cache_dir):
        """TTL 从传入的 CloudAgentConfig 对象获取"""
        from core.config import CloudAgentConfig

        custom_ttl = 7200
        cloud_config = CloudAgentConfig(egress_ip_cache_ttl=custom_ttl)

        manager = EgressIPManager(
            cache_dir=temp_cache_dir,
            cloud_config=cloud_config
        )
        assert manager.cache_ttl == custom_ttl

    def test_ttl_explicit_override(self, temp_cache_dir):
        """显式传入 cache_ttl 参数覆盖配置值"""
        from core.config import CloudAgentConfig

        # 即使传入 cloud_config，显式 cache_ttl 优先
        cloud_config = CloudAgentConfig(egress_ip_cache_ttl=3600)
        explicit_ttl = 9000

        manager = EgressIPManager(
            cache_dir=temp_cache_dir,
            cache_ttl=explicit_ttl,
            cloud_config=cloud_config
        )
        assert manager.cache_ttl == explicit_ttl

    def test_ttl_priority_order(self, temp_cache_dir, monkeypatch):
        """验证 TTL 配置优先级: 显式参数 > cloud_config > get_config()"""
        from core.config import ConfigManager, CloudAgentConfig

        ConfigManager.reset_instance()

        # 设置 mock get_config 返回 1000
        def mock_get_config():
            class MockConfig:
                cloud_agent = CloudAgentConfig(egress_ip_cache_ttl=1000)
            return MockConfig()
        monkeypatch.setattr("cursor.network.get_config", mock_get_config)

        # 场景 1: 仅使用 get_config() 默认值
        manager1 = EgressIPManager(cache_dir=temp_cache_dir)
        assert manager1.cache_ttl == 1000

        # 场景 2: 传入 cloud_config 覆盖
        cloud_config = CloudAgentConfig(egress_ip_cache_ttl=2000)
        manager2 = EgressIPManager(cache_dir=temp_cache_dir, cloud_config=cloud_config)
        assert manager2.cache_ttl == 2000

        # 场景 3: 显式 cache_ttl 参数最高优先
        manager3 = EgressIPManager(
            cache_dir=temp_cache_dir,
            cache_ttl=3000,
            cloud_config=cloud_config
        )
        assert manager3.cache_ttl == 3000

        ConfigManager.reset_instance()

    def test_ttl_consistency_with_config_yaml(self, temp_cache_dir, monkeypatch):
        """确保 TTL 来源与 config.yaml 中 cloud_agent.egress_ip_cache_ttl 一致"""
        from core.config import ConfigManager, get_config as real_get_config

        # 重置并获取真实配置
        ConfigManager.reset_instance()

        # 获取配置文件中的 TTL 值
        try:
            config = real_get_config()
            config_ttl = config.cloud_agent.egress_ip_cache_ttl
        except Exception:
            # 如果无法加载配置，使用默认值
            config_ttl = 3600

        # 不 mock，使用真实的 get_config
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        # manager 的 TTL 应与 config 一致
        assert manager.cache_ttl == config_ttl

        ConfigManager.reset_instance()


# ==================== 模块级便捷函数测试 ====================


class TestModuleFunctions:
    """模块级便捷函数测试"""

    def test_get_manager_singleton(self):
        """get_manager 返回单例"""
        # 重置全局管理器
        import cursor.network as network_module
        network_module._manager = None

        manager1 = get_manager()
        manager2 = get_manager()
        assert manager1 is manager2

        # 清理
        network_module._manager = None

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_egress_ip_ranges_function(self):
        """fetch_egress_ip_ranges 便捷函数"""
        import cursor.network as network_module
        network_module._manager = None

        api_response = {"ip_ranges": ["10.0.0.0/8"], "version": "1.0.0"}
        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(200, json=api_response)
        )

        config = await fetch_egress_ip_ranges()
        assert config is not None

        # 清理
        network_module._manager = None

    def test_is_allowed_ip_function(self):
        """is_allowed_ip 便捷函数"""
        import cursor.network as network_module
        network_module._manager = None

        # 使用备用配置
        result = is_allowed_ip("35.192.0.1")
        assert result is True

        result = is_allowed_ip("8.8.8.8")
        assert result is False

        # 清理
        network_module._manager = None

    def test_export_firewall_rules_function(self):
        """export_firewall_rules 便捷函数"""
        import cursor.network as network_module
        network_module._manager = None

        rules = export_firewall_rules(NGINX)
        assert "allow" in rules
        assert "Cursor Cloud Agent" in rules

        # 清理
        network_module._manager = None


# ==================== 边界条件测试 ====================


class TestEdgeCases:
    """边界条件测试"""

    def test_empty_ip_ranges(self):
        """空 IP 范围列表"""
        config = EgressIPConfig(ip_ranges=[])
        assert config.is_allowed_ip("192.168.1.1") is False
        rules = config.export_firewall_rules(FirewallFormat.NGINX)
        assert "allow" not in rules.split("\n")[3:]  # 跳过注释行

    def test_single_ip_range(self):
        """单个 IP 范围"""
        config = EgressIPConfig(ip_ranges=["192.168.1.0/24"])
        assert config.is_allowed_ip("192.168.1.100") is True
        assert config.is_allowed_ip("192.168.2.1") is False

    def test_overlapping_ip_ranges(self):
        """重叠 IP 范围"""
        config = EgressIPConfig(ip_ranges=["10.0.0.0/8", "10.10.0.0/16"])
        assert config.is_allowed_ip("10.10.10.10") is True
        assert config.is_allowed_ip("10.20.20.20") is True

    def test_mixed_ipv4_ipv6(self):
        """混合 IPv4 和 IPv6"""
        config = EgressIPConfig(
            ip_ranges=["192.168.1.0/24", "2001:db8::/32"]
        )
        assert config.is_allowed_ip("192.168.1.50") is True
        assert config.is_allowed_ip("2001:db8::1") is True
        assert config.is_allowed_ip("10.0.0.1") is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_returns_ranges_key(self, temp_cache_dir):
        """API 返回 ranges 而不是 ip_ranges"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        # 某些 API 可能使用 ranges 键
        api_response = {"ranges": ["192.168.0.0/16"], "version": "1.0.0"}
        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(200, json=api_response)
        )

        config = await manager._fetch_from_api()
        assert config is not None
        assert "192.168.0.0/16" in config.ip_ranges

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_returns_empty_response(self, temp_cache_dir):
        """API 返回空响应"""
        manager = EgressIPManager(cache_dir=temp_cache_dir)

        api_response = {"ip_ranges": [], "version": "1.0.0"}
        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(200, json=api_response)
        )

        config = await manager._fetch_from_api()
        assert config is not None
        assert len(config.ip_ranges) == 0


# ==================== 并发测试 ====================


class TestConcurrency:
    """并发访问测试"""

    @pytest.mark.asyncio
    @respx.mock
    async def test_concurrent_fetch(self, temp_cache_dir, api_response_data):
        """并发获取 IP 范围"""
        import asyncio

        manager = EgressIPManager(cache_dir=temp_cache_dir)

        respx.get(EgressIPManager.CURSOR_API_URL).mock(
            return_value=httpx.Response(200, json=api_response_data)
        )

        # 同时发起多个请求
        tasks = [manager.fetch_egress_ip_ranges() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # 所有结果应该一致
        for config in results:
            assert config is not None
            assert len(config.ip_ranges) == 4
