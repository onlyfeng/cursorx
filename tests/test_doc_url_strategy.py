"""测试 knowledge/doc_url_strategy.py 文档 URL 策略模块

测试覆盖：
1. 顺序=优先级拼接后稳定去重
2. 数量不超过 max_urls
3. 不足时按规则补足到 fallback_core_docs_count
4. 重复/大小写/fragment 不影响确定性
5. 无关键词时 llms 优先级不因关键词提升（但仍参与选择）
6. llms 命中但超限截断
7. changelog_links 与 related_doc_urls 互相重复的去重
8. normalize_url 严格规范化（重复斜杠、大小写、fragment 等）
9. is_allowed_doc_url 对规范化后 URL 的一致判断
"""

import pytest

from knowledge.doc_url_strategy import (
    DocURLStrategyConfig,
    deduplicate_urls,
    filter_urls_by_keywords,
    is_allowed_doc_url,
    normalize_url,
    parse_llms_txt_urls,
    select_urls_to_fetch,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def default_config() -> DocURLStrategyConfig:
    """默认配置"""
    return DocURLStrategyConfig()


@pytest.fixture
def small_max_config() -> DocURLStrategyConfig:
    """小 max_urls 配置用于测试截断"""
    return DocURLStrategyConfig(
        max_urls=3,
        fallback_core_docs_count=2,
        allowed_domains=[],  # 允许所有域名，避免测试被默认域名过滤影响
    )


@pytest.fixture
def no_filter_config() -> DocURLStrategyConfig:
    """无域名过滤配置"""
    return DocURLStrategyConfig(
        allowed_domains=[],
        exclude_patterns=[],  # 禁用所有排除规则
        max_urls=20,
        fallback_core_docs_count=5,
    )


# ============================================================
# Test: normalize_url
# ============================================================


class TestNormalizeUrl:
    """测试 URL 规范化"""

    def test_remove_fragment(self) -> None:
        """测试移除 URL fragment（锚点）"""
        url = "https://example.com/page#section"
        result = normalize_url(url)
        assert result == "https://example.com/page"

    def test_remove_fragment_with_query(self) -> None:
        """测试移除 fragment 但保留 query"""
        url = "https://example.com/page?key=value#section"
        result = normalize_url(url)
        assert result == "https://example.com/page?key=value"

    def test_remove_trailing_slash(self) -> None:
        """测试移除末尾斜杠"""
        url = "https://example.com/docs/"
        result = normalize_url(url)
        assert result == "https://example.com/docs"

    def test_keep_root_slash(self) -> None:
        """测试保留根路径斜杠"""
        url = "https://example.com/"
        result = normalize_url(url)
        assert result == "https://example.com/"

    def test_lowercase_domain(self) -> None:
        """测试域名小写化"""
        url = "https://EXAMPLE.COM/Docs"
        result = normalize_url(url)
        assert result == "https://example.com/Docs"

    def test_lowercase_scheme(self) -> None:
        """测试 scheme 小写化"""
        url = "HTTPS://example.com/page"
        result = normalize_url(url)
        assert result == "https://example.com/page"

    def test_preserve_path_case(self) -> None:
        """测试保留 path 大小写"""
        url = "https://example.com/Docs/Guide/README"
        result = normalize_url(url)
        assert result == "https://example.com/Docs/Guide/README"

    def test_preserve_query_case(self) -> None:
        """测试保留 query 大小写"""
        url = "https://example.com/search?Query=Test&Filter=Active"
        result = normalize_url(url)
        assert result == "https://example.com/search?Query=Test&Filter=Active"

    def test_resolve_relative_path(self) -> None:
        """测试相对路径解析"""
        url = "/docs/guide"
        base_url = "https://example.com"
        result = normalize_url(url, base_url)
        assert result == "https://example.com/docs/guide"

    def test_resolve_relative_path_with_dot(self) -> None:
        """测试 ./ 相对路径解析"""
        url = "./guide"
        base_url = "https://example.com/docs/"
        result = normalize_url(url, base_url)
        assert result == "https://example.com/docs/guide"

    def test_resolve_parent_path(self) -> None:
        """测试 ../ 路径解析"""
        url = "https://example.com/docs/../api/"
        result = normalize_url(url)
        assert result == "https://example.com/api"

    def test_resolve_multiple_parent_paths(self) -> None:
        """测试多个 ../ 路径解析"""
        url = "https://example.com/a/b/c/../../d"
        result = normalize_url(url)
        assert result == "https://example.com/a/d"

    def test_remove_duplicate_slashes(self) -> None:
        """测试移除重复斜杠"""
        url = "https://example.com//docs//page"
        result = normalize_url(url)
        assert result == "https://example.com/docs/page"

    def test_remove_multiple_duplicate_slashes(self) -> None:
        """测试移除多个连续重复斜杠"""
        url = "https://example.com///docs////page///"
        result = normalize_url(url)
        assert result == "https://example.com/docs/page"

    def test_mixed_case_and_fragment(self) -> None:
        """测试混合大小写和 fragment"""
        url = "HTTPS://EXAMPLE.COM/Docs/Guide#Section"
        result = normalize_url(url)
        assert result == "https://example.com/Docs/Guide"

    def test_mixed_case_duplicate_slashes_fragment(self) -> None:
        """测试混合场景：大小写 + 重复斜杠 + fragment"""
        url = "HTTPS://EXAMPLE.COM//Docs//Guide//#section"
        result = normalize_url(url)
        assert result == "https://example.com/Docs/Guide"

    def test_deterministic_output(self) -> None:
        """测试输出确定性（多次调用结果一致）"""
        url = "https://Example.COM/docs#section/"
        results = [normalize_url(url) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_empty_url(self) -> None:
        """测试空 URL"""
        assert normalize_url("") == ""
        assert normalize_url("   ") == ""

    def test_non_http_scheme(self) -> None:
        """测试非 HTTP(S) scheme 保持原样"""
        url = "ftp://example.com/file"
        result = normalize_url(url)
        assert result == "ftp://example.com/file"

    def test_default_https_scheme(self) -> None:
        """测试无 scheme 时默认使用 https"""
        # 相对路径需要 base_url 才能变成绝对 URL
        url = "//example.com/page"
        base_url = "https://other.com"
        result = normalize_url(url, base_url)
        assert result.startswith("https://")

    def test_idempotent(self) -> None:
        """测试规范化是幂等的（多次规范化结果不变）"""
        original = "HTTPS://EXAMPLE.COM//Docs//Guide//#section"
        first = normalize_url(original)
        second = normalize_url(first)
        third = normalize_url(second)
        assert first == second == third


# ============================================================
# Test: is_allowed_doc_url
# ============================================================


class TestIsAllowedDocUrl:
    """测试 URL 允许判断（基于规范化后的 URL）"""

    def test_allowed_domain_exact_match(self) -> None:
        """测试精确域名匹配"""
        config = DocURLStrategyConfig(allowed_domains=["docs.python.org"])
        assert is_allowed_doc_url("https://docs.python.org/3/library/", config)

    def test_allowed_domain_subdomain(self) -> None:
        """测试子域名匹配"""
        config = DocURLStrategyConfig(allowed_domains=["python.org"])
        assert is_allowed_doc_url("https://docs.python.org/guide", config)

    def test_not_allowed_domain(self) -> None:
        """测试不在允许列表的域名"""
        config = DocURLStrategyConfig(allowed_domains=["python.org"])
        assert not is_allowed_doc_url("https://other.com/docs", config)

    def test_case_insensitive_domain(self) -> None:
        """测试域名大小写不敏感"""
        config = DocURLStrategyConfig(allowed_domains=["docs.python.org"])
        assert is_allowed_doc_url("https://DOCS.PYTHON.ORG/guide", config)
        assert is_allowed_doc_url("https://Docs.Python.Org/guide", config)

    def test_normalized_url_with_fragment(self) -> None:
        """测试带 fragment 的 URL 规范化后检查"""
        config = DocURLStrategyConfig(
            allowed_domains=["example.com"],
            exclude_patterns=[],  # 禁用排除规则
        )
        # 带 fragment 的 URL 应该被规范化后检查
        assert is_allowed_doc_url("https://example.com/page#section", config)

    def test_normalized_url_with_duplicate_slashes(self) -> None:
        """测试带重复斜杠的 URL 规范化后检查"""
        config = DocURLStrategyConfig(
            allowed_domains=["example.com"],
            exclude_patterns=[],
        )
        assert is_allowed_doc_url("https://example.com//docs//page", config)

    def test_relative_url_with_base(self) -> None:
        """测试相对 URL 带 base_url 的检查"""
        config = DocURLStrategyConfig(
            allowed_domains=["example.com"],
            exclude_patterns=[],
        )
        assert is_allowed_doc_url("/docs/guide", config, base_url="https://example.com")

    def test_relative_url_wrong_domain(self) -> None:
        """测试相对 URL 转换到非允许域名"""
        config = DocURLStrategyConfig(
            allowed_domains=["example.com"],
            exclude_patterns=[],
        )
        assert not is_allowed_doc_url("/docs/guide", config, base_url="https://other.com")

    def test_exclude_pattern_file_extension(self) -> None:
        """测试排除模式：文件扩展名"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[r".*\.(png|jpg|gif)$"],
        )
        assert not is_allowed_doc_url("https://example.com/image.png", config)
        assert is_allowed_doc_url("https://example.com/page.html", config)

    def test_empty_url(self) -> None:
        """测试空 URL"""
        config = DocURLStrategyConfig()
        assert not is_allowed_doc_url("", config)

    def test_no_domain_filter(self) -> None:
        """测试无域名过滤（允许所有）"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
        )
        assert is_allowed_doc_url("https://any-domain.com/page", config)

    def test_normalize_disabled(self) -> None:
        """测试禁用规范化时的行为"""
        config = DocURLStrategyConfig(
            allowed_domains=["example.com"],
            exclude_patterns=[],
            normalize=False,
        )
        # 大写域名在不规范化时应该无法匹配（因为比较是小写的）
        # 但实际上 is_allowed_doc_url 内部仍会对 netloc 进行 lower() 处理
        assert is_allowed_doc_url("https://EXAMPLE.COM/page", config)


# ============================================================
# Test: deduplicate_urls
# ============================================================


class TestDeduplicateUrls:
    """测试 URL 去重"""

    def test_remove_duplicates(self) -> None:
        """测试基本去重"""
        urls = [
            "https://a.com/page",
            "https://b.com/page",
            "https://a.com/page",  # 重复
        ]
        result = deduplicate_urls(urls)
        assert result == ["https://a.com/page", "https://b.com/page"]

    def test_normalize_before_dedup(self) -> None:
        """测试规范化后去重"""
        urls = [
            "https://a.com/page/",
            "https://a.com/page",  # 规范化后相同
            "https://a.com/page#section",  # 规范化后相同
        ]
        result = deduplicate_urls(urls, normalize_before_dedup=True)
        assert len(result) == 1
        assert result[0] == "https://a.com/page"

    def test_preserve_order(self) -> None:
        """测试保持原始顺序"""
        urls = [
            "https://c.com/page",
            "https://a.com/page",
            "https://b.com/page",
        ]
        result = deduplicate_urls(urls)
        assert result == [
            "https://c.com/page",
            "https://a.com/page",
            "https://b.com/page",
        ]

    def test_case_insensitive_domain_dedup(self) -> None:
        """测试域名大小写不敏感去重"""
        urls = [
            "https://EXAMPLE.com/page",
            "https://example.COM/page",  # 应被去重
        ]
        result = deduplicate_urls(urls, normalize_before_dedup=True)
        assert len(result) == 1

    def test_duplicate_slashes_dedup(self) -> None:
        """测试重复斜杠去重"""
        urls = [
            "https://example.com//docs//page",
            "https://example.com/docs/page",  # 规范化后相同
        ]
        result = deduplicate_urls(urls, normalize_before_dedup=True)
        assert len(result) == 1
        assert result[0] == "https://example.com/docs/page"

    def test_mixed_normalize_dedup(self) -> None:
        """测试混合规范化去重（大小写 + 重复斜杠 + fragment）"""
        urls = [
            "https://EXAMPLE.COM//docs//page#section1",
            "https://example.com/docs/page#section2",
            "https://Example.Com/docs/page/",  # 末尾斜杠
        ]
        result = deduplicate_urls(urls, normalize_before_dedup=True)
        assert len(result) == 1
        assert result[0] == "https://example.com/docs/page"

    def test_empty_list(self) -> None:
        """测试空列表"""
        result = deduplicate_urls([])
        assert result == []


# ============================================================
# Test: parse_llms_txt_urls
# ============================================================


class TestParseLlmsTxtUrls:
    """测试 llms.txt 解析"""

    def test_parse_plain_urls(self) -> None:
        """测试解析纯 URL"""
        content = """# My Docs
https://example.com/guide
https://example.com/api
"""
        result = parse_llms_txt_urls(content)
        assert len(result) == 2
        assert "https://example.com/guide" in result
        assert "https://example.com/api" in result

    def test_parse_markdown_links(self) -> None:
        """测试解析 Markdown 链接"""
        content = """
[Guide](https://example.com/guide)
[API Reference](https://example.com/api)
"""
        result = parse_llms_txt_urls(content)
        assert len(result) == 2

    def test_parse_relative_urls(self) -> None:
        """测试解析相对 URL"""
        # 注意: ./guide 相对于 https://example.com/docs/ 会解析为 https://example.com/docs/guide
        # 但相对于 https://example.com/docs（无末尾斜杠）会解析为 https://example.com/guide
        content = "[Guide](./guide)\n[API](/api)"
        base_url = "https://example.com/docs/"  # 添加末尾斜杠
        result = parse_llms_txt_urls(content, base_url)
        assert "https://example.com/docs/guide" in result
        assert "https://example.com/api" in result

    def test_skip_anchor_only_links(self) -> None:
        """测试跳过纯锚点链接"""
        content = "[Skip](#section)\n[Keep](https://example.com/page)"
        result = parse_llms_txt_urls(content)
        assert len(result) == 1
        assert "https://example.com/page" in result

    def test_empty_content(self) -> None:
        """测试空内容"""
        result = parse_llms_txt_urls("")
        assert result == []

    def test_none_content(self) -> None:
        """测试 None 内容"""
        result = parse_llms_txt_urls(None)  # type: ignore
        assert result == []

    # ============================================================
    # 新增测试：注释/空行/非 URL 行
    # ============================================================

    def test_skip_comment_lines(self) -> None:
        """测试注释行中的 URL 也会被提取

        parse_llms_txt_urls 基于正则匹配 URL 模式，不过滤注释行。
        注释行中的 URL 也会被提取出来。
        """
        content = """# This is a comment
# Another comment line
https://example.com/valid-url
# Comment with URL mention: https://example.com/ignored
"""
        result = parse_llms_txt_urls(content)
        # 正则匹配会提取所有 http(s):// 开头的 URL，包括注释行中的
        assert len(result) == 2
        assert "https://example.com/valid-url" in result
        assert "https://example.com/ignored" in result

    def test_skip_empty_lines(self) -> None:
        """测试跳过空行"""
        content = """

https://example.com/url1

   
https://example.com/url2

"""
        result = parse_llms_txt_urls(content)
        assert len(result) == 2
        assert "https://example.com/url1" in result
        assert "https://example.com/url2" in result

    def test_skip_non_url_lines(self) -> None:
        """测试跳过非 URL 行（普通文本）"""
        content = """Welcome to our documentation
This is just a description
https://example.com/docs
More plain text here
Last valid URL: https://example.com/end
"""
        result = parse_llms_txt_urls(content)
        # 应只提取有效 URL
        assert len(result) == 2
        assert "https://example.com/docs" in result
        assert "https://example.com/end" in result

    def test_mixed_content_types(self) -> None:
        """测试混合内容：注释、空行、非 URL 行、有效 URL"""
        content = """# Header comment

Welcome to docs

https://example.com/guide
[API](https://example.com/api)

# Another section
Some description text

https://example.com/reference
"""
        result = parse_llms_txt_urls(content)
        assert len(result) == 3
        assert "https://example.com/guide" in result
        assert "https://example.com/api" in result
        assert "https://example.com/reference" in result

    # ============================================================
    # 新增测试：混合域名
    # ============================================================

    def test_parse_mixed_domains(self) -> None:
        """测试解析多域名 URL"""
        content = """
https://docs.python.org/3/library/
https://nodejs.org/en/docs/
https://example.com/guide
[React](https://reactjs.org/docs)
"""
        result = parse_llms_txt_urls(content)
        assert len(result) == 4
        assert "https://docs.python.org/3/library/" in result
        assert "https://nodejs.org/en/docs/" in result
        assert "https://example.com/guide" in result
        assert "https://reactjs.org/docs" in result

    # ============================================================
    # 新增测试：带 query/fragment 的 URL
    # ============================================================

    def test_parse_url_with_query(self) -> None:
        """测试解析带 query string 的 URL"""
        content = """
https://example.com/search?q=test&page=1
https://example.com/api?version=v2
"""
        result = parse_llms_txt_urls(content)
        assert len(result) == 2
        assert "https://example.com/search?q=test&page=1" in result
        assert "https://example.com/api?version=v2" in result

    def test_parse_url_with_fragment(self) -> None:
        """测试解析带 fragment 的 URL"""
        content = """
https://example.com/docs#introduction
https://example.com/guide#getting-started
"""
        result = parse_llms_txt_urls(content)
        # fragment 可能在 URL 匹配时被截断，取决于正则模式
        # 检查至少匹配到基础 URL
        assert len(result) >= 2
        # 验证基础 URL 存在
        base_urls = [u.split("#")[0] for u in result]
        assert "https://example.com/docs" in base_urls
        assert "https://example.com/guide" in base_urls

    def test_parse_url_with_query_and_fragment(self) -> None:
        """测试解析同时带 query 和 fragment 的 URL"""
        content = """
https://example.com/page?tab=overview#section
[Link](https://example.com/api?v=2#usage)
"""
        result = parse_llms_txt_urls(content)
        assert len(result) >= 2

    # ============================================================
    # 新增测试：重复 URL
    # ============================================================

    def test_parse_duplicate_urls(self) -> None:
        """测试解析重复 URL（不去重，保持原始顺序）

        parse_llms_txt_urls 本身不负责去重，只负责提取。
        去重由 select_urls_to_fetch 处理。
        """
        content = """
https://example.com/guide
https://example.com/api
https://example.com/guide
[Same Guide](https://example.com/guide)
"""
        result = parse_llms_txt_urls(content)
        # 应提取所有 URL（包括重复的）
        assert result.count("https://example.com/guide") >= 2
        assert "https://example.com/api" in result

    def test_parse_duplicate_urls_different_formats(self) -> None:
        """测试解析不同格式的重复 URL"""
        content = """
https://example.com/docs
[Docs](https://example.com/docs)
https://example.com/docs/
"""
        result = parse_llms_txt_urls(content)
        # 应提取所有，末尾斜杠的视为不同 URL（规范化后才相同）
        assert len(result) >= 2


# ============================================================
# Test: filter_urls_by_keywords
# ============================================================


class TestFilterUrlsByKeywords:
    """测试 filter_urls_by_keywords 函数"""

    def test_filter_with_matching_keywords(self) -> None:
        """测试过滤匹配关键词的 URL

        需设置 min_match_score > 0 才能过滤不匹配的 URL。
        默认 min_match_score=0.0 会返回所有 URL（因为 0 >= 0）。
        """
        urls = [
            "https://example.com/python-guide",
            "https://example.com/java-docs",
            "https://example.com/python-api",
        ]
        # min_match_score=0.5 表示需要匹配至少 50% 的关键词
        # 对于单个关键词，0.5 表示需要匹配（score=1.0 > 0.5）
        result = filter_urls_by_keywords(urls, ["python"], min_match_score=0.5)
        assert len(result) == 2
        assert "https://example.com/python-guide" in result
        assert "https://example.com/python-api" in result
        assert "https://example.com/java-docs" not in result

    def test_filter_with_no_matching_keywords(self) -> None:
        """测试没有匹配关键词时返回空"""
        urls = [
            "https://example.com/python-guide",
            "https://example.com/java-docs",
        ]
        result = filter_urls_by_keywords(urls, ["nonexistent"], min_match_score=0.5)
        # min_match_score=0.5 时，需要至少匹配 50% 的关键词
        assert len(result) == 0

    def test_filter_with_empty_keywords(self) -> None:
        """测试空关键词列表返回全部"""
        urls = [
            "https://example.com/python",
            "https://example.com/java",
        ]
        result = filter_urls_by_keywords(urls, [])
        assert result == urls

    def test_filter_case_insensitive(self) -> None:
        """测试大小写不敏感匹配

        需设置 min_match_score > 0 才能过滤不匹配的 URL。
        """
        urls = [
            "https://example.com/Python-Guide",
            "https://example.com/JAVA-DOCS",
        ]
        result = filter_urls_by_keywords(urls, ["python"], min_match_score=0.5)
        assert len(result) == 1
        assert "https://example.com/Python-Guide" in result

    def test_filter_preserves_order(self) -> None:
        """测试保持原始顺序"""
        urls = [
            "https://example.com/c-python",
            "https://example.com/a-python",
            "https://example.com/b-python",
        ]
        result = filter_urls_by_keywords(urls, ["python"])
        assert result == urls  # 顺序应保持不变

    def test_filter_with_min_score_threshold(self) -> None:
        """测试最小匹配得分阈值"""
        urls = [
            "https://example.com/python-asyncio",  # 匹配 2/2 = 1.0
            "https://example.com/python-only",  # 匹配 1/2 = 0.5
            "https://example.com/asyncio-only",  # 匹配 1/2 = 0.5
            "https://example.com/other",  # 匹配 0/2 = 0.0
        ]
        # min_match_score=0.6 应该只保留 python-asyncio
        result = filter_urls_by_keywords(urls, ["python", "asyncio"], min_match_score=0.6)
        assert len(result) == 1
        assert "https://example.com/python-asyncio" in result

    def test_filter_with_zero_threshold(self) -> None:
        """测试阈值为 0 时返回所有 URL"""
        urls = [
            "https://example.com/python",
            "https://example.com/java",
            "https://example.com/other",
        ]
        result = filter_urls_by_keywords(urls, ["python"], min_match_score=0.0)
        # 阈值为 0，所有 URL 都应该返回（包括不匹配的）
        assert len(result) == 3

    def test_filter_multiple_keywords_partial_match(self) -> None:
        """测试多个关键词的部分匹配"""
        urls = [
            "https://example.com/python-django-tutorial",  # 匹配 2/3
            "https://example.com/python-flask",  # 匹配 1/3
            "https://example.com/java-spring",  # 匹配 0/3
        ]
        # 阈值 0.3，允许匹配 1 个关键词
        result = filter_urls_by_keywords(urls, ["python", "django", "react"], min_match_score=0.3)
        assert len(result) == 2
        assert "https://example.com/python-django-tutorial" in result
        assert "https://example.com/python-flask" in result


# ============================================================
# Test: 域名过滤与允许列表
# ============================================================


class TestAllowedDomainsFiltering:
    """测试域名过滤：仅保留允许域名的 URL"""

    def test_filter_by_allowed_domains(self) -> None:
        """测试仅保留允许域名的 URL"""
        config = DocURLStrategyConfig(
            allowed_domains=["docs.python.org", "example.com"],
            exclude_patterns=[],
            max_urls=20,
        )
        result = select_urls_to_fetch(
            changelog_links=[
                "https://docs.python.org/guide",
                "https://other-domain.com/page",  # 应被过滤
            ],
            related_doc_urls=[
                "https://example.com/api",
                "https://blocked.com/docs",  # 应被过滤
            ],
            llms_txt_content="https://docs.python.org/ref\nhttps://unknown.org/page",
            core_docs=["https://example.com/core"],
            keywords=[],
            config=config,
        )
        # 只应包含允许域名的 URL
        for url in result:
            from urllib.parse import urlparse

            domain = urlparse(url).netloc
            assert (
                domain in ["docs.python.org", "example.com"]
                or domain.endswith(".docs.python.org")
                or domain.endswith(".example.com")
            ), f"域名 {domain} 不在允许列表中"

    def test_filter_mixed_domains_from_llms_txt(self) -> None:
        """测试过滤 llms.txt 中的混合域名"""
        config = DocURLStrategyConfig(
            allowed_domains=["cursor.com"],
            exclude_patterns=[],
            max_urls=10,
        )
        llms_content = """
https://cursor.com/docs
https://github.com/cursor/repo
https://cursor.com/api
https://stackoverflow.com/questions/cursor
https://api.cursor.com/reference
"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=llms_content,
            core_docs=[],
            keywords=[],
            config=config,
        )
        # 只应包含 cursor.com 及其子域名
        assert len(result) == 3
        assert "https://cursor.com/docs" in result
        assert "https://cursor.com/api" in result
        assert "https://api.cursor.com/reference" in result
        # 不应包含其他域名
        assert not any("github.com" in u for u in result)
        assert not any("stackoverflow.com" in u for u in result)

    def test_subdomain_matching(self) -> None:
        """测试子域名匹配"""
        config = DocURLStrategyConfig(
            allowed_domains=["python.org"],  # 允许所有 python.org 子域名
            exclude_patterns=[],
            max_urls=10,
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://docs.python.org/3/library/",
                "https://wiki.python.org/moin/",
                "https://pypi.python.org/simple/",
                "https://python.org/about/",
                "https://other.org/python",  # 不应匹配
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=config,
        )
        assert len(result) == 4
        assert not any("other.org" in u for u in result)

    def test_mixed_cursor_and_external_domains_filtered(self) -> None:
        """测试 cursor.com 与外域（如 github.com）混合链接的过滤

        场景：传入包含 cursor.com 和多个外域链接，
        配置 allowed_domains=["cursor.com"]，断言结果不含外域。
        """
        config = DocURLStrategyConfig(
            allowed_domains=["cursor.com"],  # 仅允许 cursor.com
            exclude_patterns=[],  # 禁用默认排除规则以便测试
            max_urls=20,
        )
        result = select_urls_to_fetch(
            changelog_links=[
                "https://cursor.com/changelog/2026",
                "https://github.com/cursor/releases",  # 外域
            ],
            related_doc_urls=[
                "https://cursor.com/docs/guide",
                "https://stackoverflow.com/questions/cursor",  # 外域
                "https://api.cursor.com/reference",  # 子域名，应保留
            ],
            llms_txt_content=(
                "https://cursor.com/docs/overview\n"
                "https://npmjs.com/package/cursor\n"  # 外域
                "https://docs.cursor.com/getting-started\n"  # 子域名
            ),
            core_docs=[
                "https://cursor.com/docs/core",
                "https://external-site.org/docs",  # 外域
            ],
            keywords=[],
            config=config,
        )

        # 验证只包含 cursor.com 及其子域名的 URL
        for url in result:
            from urllib.parse import urlparse

            domain = urlparse(url).netloc.lower()
            assert domain == "cursor.com" or domain.endswith(".cursor.com"), f"外域 URL 不应出现: {url}"

        # 验证外域 URL 不在结果中
        assert not any("github.com" in u for u in result)
        assert not any("stackoverflow.com" in u for u in result)
        assert not any("npmjs.com" in u for u in result)
        assert not any("external-site.org" in u for u in result)

        # 验证 cursor.com 相关 URL 存在
        assert any("cursor.com/changelog" in u for u in result)
        assert any("cursor.com/docs/guide" in u for u in result)
        assert any("api.cursor.com" in u for u in result)
        assert any("docs.cursor.com" in u for u in result)


# ============================================================
# Test: 去重与截断策略
# ============================================================


class TestDedupAndTruncationStrategy:
    """测试去重与截断策略"""

    def test_dedup_url_with_query_difference(self) -> None:
        """测试带不同 query 的 URL 不会被去重（query 被保留）"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=10,
        )
        result = select_urls_to_fetch(
            changelog_links=[
                "https://example.com/page?version=1",
                "https://example.com/page?version=2",
            ],
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=config,
        )
        # 不同 query 的 URL 应该都被保留
        assert len(result) == 2

    def test_dedup_url_fragment_removed(self) -> None:
        """测试 fragment 被移除后的去重"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=10,
        )
        result = select_urls_to_fetch(
            changelog_links=[
                "https://example.com/page#section1",
                "https://example.com/page#section2",
                "https://example.com/page",
            ],
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=config,
        )
        # fragment 移除后，三个 URL 规范化为同一个，应只保留一个
        assert len(result) == 1
        assert "#" not in result[0]

    def test_truncation_preserves_priority_order(self) -> None:
        """测试截断时保持优先级顺序"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=3,
        )
        result = select_urls_to_fetch(
            changelog_links=["https://example.com/changelog"],
            related_doc_urls=["https://example.com/related"],
            llms_txt_content="https://example.com/llms",
            core_docs=["https://example.com/core1", "https://example.com/core2"],
            keywords=[],
            config=config,
        )
        # 截断到 3 个，应保留优先级最高的
        assert len(result) == 3
        assert result[0] == "https://example.com/changelog"  # changelog 最高优先级
        assert result[1] == "https://example.com/llms"  # llms 次之

    def test_dedup_across_all_sources(self) -> None:
        """测试跨所有来源的去重"""
        shared_url = "https://example.com/shared"
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=20,
        )
        result = select_urls_to_fetch(
            changelog_links=[shared_url],
            related_doc_urls=[shared_url],
            llms_txt_content=shared_url,
            core_docs=[shared_url],
            keywords=[],
            config=config,
        )
        # 四个来源都有相同 URL，应只保留一个
        assert result.count(shared_url) == 1

    def test_order_stability_with_same_priority(self) -> None:
        """测试相同优先级时的顺序稳定性（按 URL 字母序）"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=10,
        )
        # 多次运行验证稳定性
        for _ in range(5):
            result = select_urls_to_fetch(
                changelog_links=[
                    "https://z-example.com/page",
                    "https://a-example.com/page",
                    "https://m-example.com/page",
                ],
                related_doc_urls=[],
                llms_txt_content=None,
                core_docs=[],
                keywords=[],
                config=config,
            )
            # 相同来源（changelog）优先级相同，应按 URL 字母序排序
            assert result[0] == "https://a-example.com/page"
            assert result[1] == "https://m-example.com/page"
            assert result[2] == "https://z-example.com/page"


# ============================================================
# Test: 关键词匹配 - path_words 交集
# ============================================================


class TestKeywordPathMatching:
    """测试关键词与 URL path 的匹配"""

    def test_keyword_match_in_path(self) -> None:
        """测试关键词在 URL 路径中命中"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=10,
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://example.com/python-guide",  # 匹配 python
                "https://example.com/java-docs",  # 不匹配
                "https://example.com/python-tutorial",  # 匹配 python
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=["python"],
            config=config,
        )
        # 匹配 python 的应该排在前面
        python_indices = [i for i, u in enumerate(result) if "python" in u]
        java_idx = next(i for i, u in enumerate(result) if "java" in u)
        assert all(pi < java_idx for pi in python_indices)

    def test_keyword_no_match(self) -> None:
        """测试关键词不命中时的行为"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=10,
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://example.com/docs/api",
                "https://example.com/docs/guide",
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=["nonexistent-keyword"],  # 不存在于任何 URL 中
            config=config,
        )
        # 没有关键词匹配，按默认优先级排序
        assert len(result) == 2
        # 相同来源优先级，按字母序
        assert result[0] == "https://example.com/docs/api"
        assert result[1] == "https://example.com/docs/guide"

    def test_keyword_multiple_matches(self) -> None:
        """测试多个关键词匹配"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=10,
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://example.com/python-guide",  # 匹配 1 个关键词
                "https://example.com/python-asyncio-tutorial",  # 匹配 2 个关键词
                "https://example.com/java-docs",  # 不匹配
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=["python", "asyncio"],
            config=config,
        )
        # 匹配更多关键词的应该排在前面
        asyncio_idx = next(i for i, u in enumerate(result) if "asyncio" in u)
        python_only_idx = next(i for i, u in enumerate(result) if "python" in u and "asyncio" not in u)
        assert asyncio_idx < python_only_idx

    def test_keyword_case_insensitive_path(self) -> None:
        """测试关键词与路径的大小写不敏感匹配"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=10,
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://example.com/Python-Guide",  # 大写 Python
                "https://example.com/other-docs",
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=["python"],  # 小写 python
            config=config,
        )
        # 大小写不敏感，应匹配
        python_idx = next(i for i, u in enumerate(result) if "Python" in u)
        other_idx = next(i for i, u in enumerate(result) if "other" in u)
        assert python_idx < other_idx

    def test_keyword_partial_match(self) -> None:
        """测试关键词部分匹配（子字符串）"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=10,
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://example.com/pythonic-style",  # 包含 python 子串
                "https://example.com/java-style",
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=["python"],
            config=config,
        )
        # python 应该作为子字符串匹配到 pythonic
        pythonic_idx = next(i for i, u in enumerate(result) if "pythonic" in u)
        java_idx = next(i for i, u in enumerate(result) if "java" in u)
        assert pythonic_idx < java_idx


# ============================================================
# Test: 关键词匹配 - limit 截断
# ============================================================


class TestKeywordLimitTruncation:
    """测试关键词匹配时的 limit 截断"""

    def test_keyword_boost_with_limit_truncation(self) -> None:
        """测试关键词提升优先级后的截断

        高优先级（关键词匹配）的 URL 应该在截断后保留。
        """
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=3,
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://example.com/no-match-1",
                "https://example.com/no-match-2",
                "https://example.com/python-match",  # 匹配关键词
                "https://example.com/no-match-3",
                "https://example.com/python-another",  # 匹配关键词
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=["python"],
            config=config,
        )
        # 截断到 3 个，关键词匹配的应该优先保留
        assert len(result) == 3
        python_count = sum(1 for u in result if "python" in u)
        assert python_count == 2, "两个匹配关键词的 URL 应该都被保留"

    def test_limit_truncation_order_stable(self) -> None:
        """测试截断后的顺序稳定性"""
        config = DocURLStrategyConfig(
            allowed_domains=[],
            exclude_patterns=[],
            max_urls=5,
        )
        urls = [f"https://example.com/page-{i}" for i in range(10)]
        # 多次运行验证稳定性
        results = []
        for _ in range(5):
            result = select_urls_to_fetch(
                changelog_links=urls[:3],
                related_doc_urls=urls[3:6],
                llms_txt_content="\n".join(urls[6:]),
                core_docs=[],
                keywords=[],
                config=config,
            )
            results.append(result)
        # 所有结果应该相同
        assert all(r == results[0] for r in results)


# ============================================================
# Test: select_urls_to_fetch - 优先级与确定性
# ============================================================


class TestSelectUrlsToFetchPriority:
    """测试 URL 选择优先级"""

    def test_priority_order_changelog_first(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试 changelog 链接优先级最高"""
        result = select_urls_to_fetch(
            changelog_links=["https://example.com/changelog"],
            related_doc_urls=["https://example.com/related"],
            llms_txt_content="https://example.com/llms",
            core_docs=["https://example.com/core"],
            keywords=[],
            config=no_filter_config,
        )
        # changelog 应该在最前面
        assert result[0] == "https://example.com/changelog"

    def test_priority_order_llms_before_related(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试 llms.txt 链接优先级高于 related_doc_urls"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=["https://example.com/related"],
            llms_txt_content="https://example.com/llms",
            core_docs=["https://example.com/core"],
            keywords=[],
            config=no_filter_config,
        )
        # llms 应该在 related 之前
        llms_idx = result.index("https://example.com/llms")
        related_idx = result.index("https://example.com/related")
        assert llms_idx < related_idx

    def test_priority_order_related_before_core(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试 related_doc_urls 优先级高于 core_docs"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=["https://example.com/related"],
            llms_txt_content=None,
            core_docs=["https://example.com/core"],
            keywords=[],
            config=no_filter_config,
        )
        related_idx = result.index("https://example.com/related")
        core_idx = result.index("https://example.com/core")
        assert related_idx < core_idx

    def test_deterministic_output(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试输出确定性（多次调用结果一致）"""
        results = [
            select_urls_to_fetch(
                changelog_links=["https://a.com/cl1", "https://a.com/cl2"],
                related_doc_urls=["https://a.com/r1", "https://a.com/r2"],
                llms_txt_content="https://a.com/l1\nhttps://a.com/l2",
                core_docs=["https://a.com/c1", "https://a.com/c2"],
                keywords=["test"],
                config=no_filter_config,
            )
            for _ in range(10)
        ]
        assert all(r == results[0] for r in results), "结果应确定性一致"

    def test_same_priority_sorted_by_url(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试相同优先级按 URL 字母序排序"""
        result = select_urls_to_fetch(
            changelog_links=["https://z.com/page", "https://a.com/page"],
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=no_filter_config,
        )
        # 相同优先级应按 URL 字母序
        assert result.index("https://a.com/page") < result.index("https://z.com/page")


# ============================================================
# Test: select_urls_to_fetch - 去重
# ============================================================


class TestSelectUrlsToFetchDedup:
    """测试 URL 选择去重"""

    def test_dedup_changelog_and_related(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试 changelog_links 与 related_doc_urls 重复 URL 去重"""
        duplicate_url = "https://example.com/same-page"
        result = select_urls_to_fetch(
            changelog_links=[duplicate_url],
            related_doc_urls=[duplicate_url],  # 重复
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=no_filter_config,
        )
        # 重复的 URL 应只出现一次
        assert result.count(duplicate_url) == 1
        # 应保留 changelog 优先级版本（第一个位置）
        assert result[0] == duplicate_url

    def test_dedup_with_fragment_difference(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试 fragment 不同但 URL 相同的去重"""
        result = select_urls_to_fetch(
            changelog_links=["https://example.com/page#section1"],
            related_doc_urls=["https://example.com/page#section2"],  # 规范化后相同
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=no_filter_config,
        )
        # 规范化后去重，应只有一个
        assert len(result) == 1
        assert "#" not in result[0]  # fragment 被移除

    def test_dedup_with_case_difference(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试域名大小写不同的去重"""
        result = select_urls_to_fetch(
            changelog_links=["https://EXAMPLE.com/page"],
            related_doc_urls=["https://example.COM/page"],  # 规范化后相同
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=no_filter_config,
        )
        assert len(result) == 1

    def test_dedup_with_trailing_slash(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试末尾斜杠不同的去重"""
        result = select_urls_to_fetch(
            changelog_links=["https://example.com/page/"],
            related_doc_urls=["https://example.com/page"],  # 规范化后相同
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=no_filter_config,
        )
        assert len(result) == 1

    def test_dedup_keeps_highest_priority(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试去重时保留最高优先级版本"""
        url = "https://example.com/page"
        # changelog 有更高的优先级
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[url],  # 较低优先级先加入
            llms_txt_content=url,  # 较高优先级后加入
            core_docs=[],
            keywords=[],
            config=no_filter_config,
        )
        # 应该在靠前的位置（因为 llms 优先级更高）
        assert result[0] == url


# ============================================================
# Test: select_urls_to_fetch - 截断与兜底
# ============================================================


class TestSelectUrlsToFetchTruncation:
    """测试 URL 选择截断与兜底"""

    def test_truncate_to_max_urls(self, small_max_config: DocURLStrategyConfig) -> None:
        """测试截断到 max_urls"""
        result = select_urls_to_fetch(
            changelog_links=[f"https://a.com/cl{i}" for i in range(5)],
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=small_max_config,  # max_urls=3
        )
        assert len(result) == 3

    def test_llms_matched_but_truncated(self) -> None:
        """测试 llms 命中但超限截断

        场景：llms.txt 有很多 URL，但因为 max_urls 限制被截断
        """
        config = DocURLStrategyConfig(
            max_urls=2,
            allowed_domains=[],
            exclude_patterns=[],
        )
        result = select_urls_to_fetch(
            changelog_links=["https://a.com/high-priority"],
            related_doc_urls=[],
            llms_txt_content=("https://a.com/llms1\nhttps://a.com/llms2\nhttps://a.com/llms3\n"),
            core_docs=[],
            keywords=[],
            config=config,
        )
        # max_urls=2，changelog 优先，只能放入 1 个 llms
        assert len(result) == 2
        assert result[0] == "https://a.com/high-priority"
        # llms 被截断
        assert sum(1 for u in result if "llms" in u) == 1

    def test_fallback_to_core_docs_when_insufficient(self) -> None:
        """测试不足时从 core_docs 补充"""
        config = DocURLStrategyConfig(
            max_urls=10,
            fallback_core_docs_count=5,
            allowed_domains=[],
            exclude_patterns=[],
        )
        result = select_urls_to_fetch(
            changelog_links=["https://a.com/cl1"],  # 只有 1 个
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[f"https://a.com/core{i}" for i in range(10)],
            keywords=[],
            config=config,
        )
        # 总数应至少达到 fallback_core_docs_count（5）
        assert len(result) >= 5

    def test_fallback_does_not_exceed_max(self) -> None:
        """测试兜底补充不超过 max_urls"""
        config = DocURLStrategyConfig(
            max_urls=3,
            fallback_core_docs_count=10,  # 比 max 大
            allowed_domains=[],
            exclude_patterns=[],
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[f"https://a.com/core{i}" for i in range(20)],
            keywords=[],
            config=config,
        )
        assert len(result) <= 3

    def test_fallback_min_required(self) -> None:
        """测试 fallback 补充到最小要求数量"""
        config = DocURLStrategyConfig(
            max_urls=10,
            fallback_core_docs_count=3,
            allowed_domains=[],
            exclude_patterns=[],
        )
        result = select_urls_to_fetch(
            changelog_links=[],  # 没有高优先级来源
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[f"https://a.com/core{i}" for i in range(5)],
            keywords=[],
            config=config,
        )
        # 应从 core_docs 补充到 fallback_core_docs_count
        assert len(result) >= 3


# ============================================================
# Test: select_urls_to_fetch - 关键词影响
# ============================================================


class TestSelectUrlsToFetchKeywords:
    """测试关键词对 URL 选择的影响"""

    def test_keyword_boost_priority(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试关键词匹配提升优先级"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://example.com/api",
                "https://example.com/guide-python-tutorial",  # 匹配关键词
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=["python"],
            config=no_filter_config,
        )
        # 匹配 python 的应该排在前面
        python_idx = next(i for i, u in enumerate(result) if "python" in u)
        api_idx = next(i for i, u in enumerate(result) if "api" in u)
        assert python_idx < api_idx

    def test_no_keywords_llms_still_participates(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试无关键词时 llms 仍参与选择（但不因关键词提升）

        关键词只影响优先级加成，不影响是否参与选择。
        """
        result = select_urls_to_fetch(
            changelog_links=["https://a.com/cl"],
            related_doc_urls=["https://a.com/related"],
            llms_txt_content="https://a.com/llms",
            core_docs=["https://a.com/core"],
            keywords=[],  # 无关键词
            config=no_filter_config,
        )
        # llms 应该参与选择
        assert "https://a.com/llms" in result

    def test_empty_keywords_no_boost(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试空关键词列表不产生优先级加成"""
        # 所有来源相同优先级时，应按字母序
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://a.com/zebra",
                "https://a.com/alpha",
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=no_filter_config,
        )
        # 相同来源优先级，按字母序
        assert result.index("https://a.com/alpha") < result.index("https://a.com/zebra")

    def test_keyword_case_insensitive(self, no_filter_config: DocURLStrategyConfig) -> None:
        """测试关键词匹配大小写不敏感"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[
                "https://a.com/PYTHON-Guide",
                "https://a.com/other",
            ],
            llms_txt_content=None,
            core_docs=[],
            keywords=["python"],  # 小写
            config=no_filter_config,
        )
        # 大写 PYTHON 也应匹配
        python_idx = next(i for i, u in enumerate(result) if "PYTHON" in u)
        other_idx = next(i for i, u in enumerate(result) if "other" in u)
        assert python_idx < other_idx


# ============================================================
# Test: 集成测试 - 模拟 UpdateAnalysis 场景
# ============================================================


class TestSelectUrlsIntegration:
    """集成测试：模拟 UpdateAnalysis 与 select_urls_to_fetch 配合"""

    def test_update_analysis_scenario(self) -> None:
        """测试模拟 UpdateAnalysis 场景

        构造：
        - changelog_links: 从 UpdateAnalysis.changelog_links
        - related_doc_urls: 从 UpdateAnalysis.related_doc_urls
        - keywords: 从 UpdateAnalysis.entries[*].keywords

        注意：关键词匹配会提升 URL 优先级（默认权重来自 DocURLStrategyConfig 的
        keyword_boost_weight 属性，默认值 1.2 与 core.config.DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT
        保持同步）。如果 llms.txt 中的 URL 匹配多个关键词，可能超过 changelog 优先级。
        """
        # 模拟 UpdateAnalysis 数据
        changelog_links = [
            "https://cursor.com/changelog-2026",
            "https://cursor.com/docs/new-feature",
        ]
        related_doc_urls = [
            "https://cursor.com/docs/cli/reference",
            "https://cursor.com/docs/new-feature",  # 与 changelog 重复
        ]
        # 不使用关键词，避免关键词加成影响优先级测试
        keywords: list[str] = []

        llms_txt_content = """
# Cursor CLI Docs
[Overview](https://cursor.com/docs/overview)
[Guide](https://cursor.com/docs/guide)
[Reference](https://cursor.com/docs/ref)
"""
        core_docs = [
            "https://cursor.com/docs/getting-started",
            "https://cursor.com/docs/installation",
            "https://cursor.com/docs/configuration",
        ]

        config = DocURLStrategyConfig(
            max_urls=5,
            fallback_core_docs_count=2,
            allowed_domains=[],
            exclude_patterns=[],
        )

        result = select_urls_to_fetch(
            changelog_links=changelog_links,
            related_doc_urls=related_doc_urls,
            llms_txt_content=llms_txt_content,
            core_docs=core_docs,
            keywords=keywords,
            config=config,
        )

        # 验证基本约束
        assert len(result) <= 5, "不应超过 max_urls"
        assert len(result) == len(set(result)), "不应有重复"

        # 验证 changelog 链接在结果中（优先级最高）
        assert "https://cursor.com/changelog-2026" in result, "changelog 链接应在结果中"
        # 无关键词时，changelog 优先级 3.0 > llms_txt 2.5，应在第一位
        assert result[0] == "https://cursor.com/changelog-2026", "changelog 应在第一位"

        # 验证去重（new-feature 只出现一次）
        new_feature_count = sum(1 for u in result if "new-feature" in u)
        assert new_feature_count <= 1, "重复的 new-feature URL 应被去重"

        # 验证确定性
        result2 = select_urls_to_fetch(
            changelog_links=changelog_links,
            related_doc_urls=related_doc_urls,
            llms_txt_content=llms_txt_content,
            core_docs=core_docs,
            keywords=keywords,
            config=config,
        )
        assert result == result2, "结果应确定性一致"

    def test_large_scale_deterministic(self) -> None:
        """测试大规模数据的确定性"""
        changelog_links = [f"https://a.com/cl{i}" for i in range(20)]
        related_doc_urls = [f"https://a.com/rel{i}" for i in range(20)]
        llms_txt_content = "\n".join(f"https://a.com/llms{i}" for i in range(20))
        core_docs = [f"https://a.com/core{i}" for i in range(20)]
        keywords = ["test", "feature", "api"]

        config = DocURLStrategyConfig(
            max_urls=10,
            fallback_core_docs_count=3,
            allowed_domains=[],
            exclude_patterns=[],
        )

        results = [
            select_urls_to_fetch(
                changelog_links=changelog_links,
                related_doc_urls=related_doc_urls,
                llms_txt_content=llms_txt_content,
                core_docs=core_docs,
                keywords=keywords,
                config=config,
            )
            for _ in range(5)
        ]

        # 所有结果应相同
        assert all(r == results[0] for r in results), "大规模数据应保持确定性"
        assert len(results[0]) == 10, "应截断到 max_urls"

    def test_mixed_duplicates_across_sources(self) -> None:
        """测试跨来源的混合重复"""
        # 使用 aaa- 前缀确保 shared_url 按字母序排在最前面
        shared_url = "https://cursor.com/aaa-shared"

        config = DocURLStrategyConfig(
            max_urls=10,
            allowed_domains=[],
            exclude_patterns=[],
        )

        result = select_urls_to_fetch(
            changelog_links=[shared_url, "https://cursor.com/cl1"],
            related_doc_urls=[shared_url, "https://cursor.com/rel1"],  # 重复
            llms_txt_content=f"{shared_url}\nhttps://cursor.com/llms1",  # 重复
            core_docs=[shared_url, "https://cursor.com/core1"],  # 重复
            keywords=[],
            config=config,
        )

        # shared_url 只应出现一次（去重）
        assert result.count(shared_url) == 1
        # 应在第一位（changelog 最高优先级 + 按字母序排在前面）
        assert result[0] == shared_url

    def test_fallback_with_partial_sources(self) -> None:
        """测试部分来源为空时的兜底"""
        config = DocURLStrategyConfig(
            max_urls=10,
            fallback_core_docs_count=5,
            allowed_domains=[],
            exclude_patterns=[],
        )

        result = select_urls_to_fetch(
            changelog_links=[],  # 空
            related_doc_urls=[],  # 空
            llms_txt_content=None,  # 空
            core_docs=[f"https://a.com/core{i}" for i in range(10)],
            keywords=[],
            config=config,
        )

        # 应从 core_docs 补充到 fallback_core_docs_count
        assert len(result) >= 5

    def test_all_sources_empty(self) -> None:
        """测试所有来源都为空"""
        config = DocURLStrategyConfig(
            max_urls=10,
            fallback_core_docs_count=5,
            allowed_domains=[],
            exclude_patterns=[],
        )

        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[],
            keywords=[],
            config=config,
        )

        assert result == []


# ============================================================
# Test: 边界情况测试
# ============================================================


class TestBoundaryConditions:
    """测试边界情况：max_urls/fallback_count 极值"""

    def test_max_urls_zero(self) -> None:
        """测试 max_urls=0 边界"""
        config = DocURLStrategyConfig(
            max_urls=0,
            fallback_core_docs_count=0,
            allowed_domains=[],
            exclude_patterns=[],
        )
        result = select_urls_to_fetch(
            changelog_links=["https://a.com/cl1"],
            related_doc_urls=["https://a.com/rel1"],
            llms_txt_content="https://a.com/llms1",
            core_docs=["https://a.com/core1"],
            keywords=[],
            config=config,
        )
        # max_urls=0 应该返回空列表
        assert result == []

    def test_max_urls_one(self) -> None:
        """测试 max_urls=1 边界"""
        config = DocURLStrategyConfig(
            max_urls=1,
            fallback_core_docs_count=0,
            allowed_domains=[],
            exclude_patterns=[],
        )
        result = select_urls_to_fetch(
            changelog_links=["https://a.com/cl1"],
            related_doc_urls=["https://a.com/rel1"],
            llms_txt_content="https://a.com/llms1",
            core_docs=["https://a.com/core1"],
            keywords=[],
            config=config,
        )
        # max_urls=1 应该只保留优先级最高的 1 个
        assert len(result) == 1
        assert result[0] == "https://a.com/cl1"  # changelog 优先级最高

    def test_fallback_count_zero(self) -> None:
        """测试 fallback_core_docs_count=0 边界"""
        config = DocURLStrategyConfig(
            max_urls=10,
            fallback_core_docs_count=0,
            allowed_domains=[],
            exclude_patterns=[],
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=["https://a.com/core1", "https://a.com/core2"],
            keywords=[],
            config=config,
        )
        # fallback_count=0 时，不应强制从 core_docs 补充（但 core_docs 仍会参与选择）
        # 实际上 core_docs 会作为候选参与选择，只是不会强制补充
        # 由于只有 core_docs 有内容，结果应该包含它们
        assert len(result) == 2

    def test_fallback_count_exceeds_core_docs(self) -> None:
        """测试 fallback_core_docs_count > len(core_docs) 边界"""
        config = DocURLStrategyConfig(
            max_urls=10,
            fallback_core_docs_count=100,  # 远超 core_docs 数量
            allowed_domains=[],
            exclude_patterns=[],
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=["https://a.com/core1", "https://a.com/core2"],
            keywords=[],
            config=config,
        )
        # 应该返回所有可用的 core_docs（不能超过实际数量）
        assert len(result) == 2

    def test_max_urls_less_than_fallback(self) -> None:
        """测试 max_urls < fallback_core_docs_count 边界"""
        config = DocURLStrategyConfig(
            max_urls=2,
            fallback_core_docs_count=10,  # 比 max_urls 大
            allowed_domains=[],
            exclude_patterns=[],
        )
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=None,
            core_docs=[f"https://a.com/core{i}" for i in range(20)],
            keywords=[],
            config=config,
        )
        # max_urls 应该是硬限制
        assert len(result) <= 2

    def test_deduplicate_stability(self) -> None:
        """测试 deduplicate_urls 的稳定性（多次调用结果一致）"""
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page1",  # 重复
            "https://example.com/page3",
            "https://Example.COM/page2",  # 规范化后重复
        ]
        results = [deduplicate_urls(urls, normalize_before_dedup=True) for _ in range(10)]
        # 所有结果应该完全相同
        assert all(r == results[0] for r in results)
        # 结果应该按首次出现顺序保留
        assert len(results[0]) == 3

    def test_normalize_url_special_characters(self) -> None:
        """测试 normalize_url 处理特殊字符"""
        # 带查询参数的 URL
        url_with_query = "https://example.com/search?q=test&page=1"
        result = normalize_url(url_with_query)
        assert "?q=test&page=1" in result

        # 带空格编码的 URL
        url_with_encoded = "https://example.com/path%20with%20space"
        result = normalize_url(url_with_encoded)
        assert "%20" in result or " " not in result

    def test_normalize_url_unicode(self) -> None:
        """测试 normalize_url 处理 Unicode 字符"""
        url_unicode = "https://example.com/路径/文档"
        result = normalize_url(url_unicode)
        # 应该保留 Unicode 字符或正确编码
        assert "example.com" in result

    def test_is_allowed_empty_domains_allows_all(self) -> None:
        """测试空 allowed_domains 允许所有域名"""
        config = DocURLStrategyConfig(
            allowed_domains=[],  # 空列表
            exclude_patterns=[],
        )
        assert is_allowed_doc_url("https://any-domain.com/page", config)
        assert is_allowed_doc_url("https://another.org/docs", config)

    def test_select_urls_with_only_llms_content(self) -> None:
        """测试仅有 llms.txt 内容时的 URL 选择"""
        config = DocURLStrategyConfig(
            max_urls=5,
            fallback_core_docs_count=0,
            allowed_domains=[],
            exclude_patterns=[],
        )
        llms_content = """
https://example.com/doc1
https://example.com/doc2
https://example.com/doc3
"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=llms_content,
            core_docs=[],
            keywords=[],
            config=config,
        )
        assert len(result) == 3
        assert "https://example.com/doc1" in result

    def test_select_urls_deterministic_with_same_input(self) -> None:
        """测试相同输入的确定性输出"""
        config = DocURLStrategyConfig(
            max_urls=5,
            allowed_domains=[],
            exclude_patterns=[],
        )
        results = [
            select_urls_to_fetch(
                changelog_links=["https://a.com/cl1", "https://a.com/cl2"],
                related_doc_urls=["https://a.com/rel1"],
                llms_txt_content="https://a.com/llms1\nhttps://a.com/llms2",
                core_docs=["https://a.com/core1"],
                keywords=["test"],
                config=config,
            )
            for _ in range(20)
        ]
        # 所有结果应该完全相同
        assert all(r == results[0] for r in results)

    def test_normalize_url_root_path_preserved(self) -> None:
        """测试根路径斜杠保留"""
        url = "https://example.com/"
        result = normalize_url(url)
        # 根路径的斜杠应该保留
        assert result == "https://example.com/"

    def test_deduplicate_with_mixed_case_and_fragments(self) -> None:
        """测试混合大小写和 fragment 的去重"""
        urls = [
            "https://EXAMPLE.com/Page#section1",
            "https://example.COM/page#section2",
            "https://Example.Com/PAGE/",  # 末尾斜杠
        ]
        result = deduplicate_urls(urls, normalize_before_dedup=True)
        # 规范化后：domain 小写，fragment 移除，末尾斜杠移除
        # 注意：normalize_url 保留 path 大小写
        # /Page, /page, /PAGE 是三个不同的路径（path 区分大小写）
        assert len(result) == 3
        # 验证 fragment 和末尾斜杠被移除
        assert all("#" not in u for u in result)
        assert not result[2].endswith("/")

    def test_deduplicate_identical_urls_with_fragments(self) -> None:
        """测试完全相同的 URL（仅 fragment 不同）的去重"""
        urls = [
            "https://example.com/page#section1",
            "https://example.com/page#section2",
            "https://example.com/page",
        ]
        result = deduplicate_urls(urls, normalize_before_dedup=True)
        # fragment 移除后，三个 URL 规范化为同一个
        assert len(result) == 1
        assert result[0] == "https://example.com/page"


# ============================================================
# Test: 外域 URL 过滤（llms.txt 场景）
# ============================================================


class TestExternalDomainFiltering:
    """测试外域 URL 过滤：llms.txt 含外域 URL 时不进入 fetch 列表"""

    def test_llms_txt_external_urls_filtered_by_prefix(self) -> None:
        """测试 llms.txt 中的外域 URL 被 allowed_url_prefixes 过滤

        场景：llms.txt 包含 github.com、stackoverflow.com 等外域链接，
        配置 allowed_url_prefixes 后，这些外域 URL 不会出现在最终列表中。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/cn/docs",
                "https://cursor.com/docs",
                "https://cursor.com/cn/changelog",
                "https://cursor.com/changelog",
            ],
            max_urls=20,
        )

        llms_content = """
# Cursor CLI Documentation
https://cursor.com/docs/overview
https://cursor.com/cn/docs/guide
https://github.com/cursor/cursor-cli
https://stackoverflow.com/questions/cursor
https://cursor.com/docs/api
https://npmjs.com/package/@cursor/cli
"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=llms_content,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 应只包含 cursor.com/docs 或 cursor.com/cn/docs 前缀的 URL
        assert len(result) == 3
        assert "https://cursor.com/docs/overview" in result
        assert "https://cursor.com/cn/docs/guide" in result
        assert "https://cursor.com/docs/api" in result

        # 外域 URL 不应出现
        assert not any("github.com" in u for u in result)
        assert not any("stackoverflow.com" in u for u in result)
        assert not any("npmjs.com" in u for u in result)

    def test_llms_txt_external_urls_filtered_by_domain(self) -> None:
        """测试 llms.txt 中的外域 URL 被 allowed_domains 过滤

        场景：仅使用 allowed_domains（不使用 allowed_url_prefixes），
        外域 URL 仍会被正确过滤。
        """
        config = DocURLStrategyConfig(
            allowed_domains=["cursor.com"],  # 仅允许 cursor.com 域名
            allowed_url_prefixes=[],  # 不使用前缀匹配
            max_urls=20,
            exclude_patterns=[],  # 禁用排除规则以便测试
        )

        llms_content = """
https://cursor.com/docs/overview
https://api.cursor.com/reference
https://github.com/cursor/repo
https://docs.cursor.com/guide
"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=llms_content,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 应包含 cursor.com 及其子域名
        assert "https://cursor.com/docs/overview" in result
        assert "https://api.cursor.com/reference" in result
        assert "https://docs.cursor.com/guide" in result

        # 外域 URL 不应出现
        assert not any("github.com" in u for u in result)

    def test_llms_txt_mixed_sources_external_filtered(self) -> None:
        """测试混合来源中的外域 URL 被统一过滤

        场景：changelog_links、related_doc_urls、llms_txt、core_docs
        都可能包含外域链接，它们都应被过滤。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/docs",
                "https://cursor.com/cn/docs",
            ],
            max_urls=10,
        )

        result = select_urls_to_fetch(
            changelog_links=[
                "https://cursor.com/docs/changelog-feature",
                "https://github.com/cursor/releases",  # 外域
            ],
            related_doc_urls=[
                "https://cursor.com/cn/docs/related",
                "https://external.com/docs",  # 外域
            ],
            llms_txt_content=(
                "https://cursor.com/docs/llms-guide\nhttps://stackoverflow.com/cursor\n"  # 外域
            ),
            core_docs=[
                "https://cursor.com/docs/core",
                "https://other-site.org/docs",  # 外域
            ],
            keywords=[],
            config=config,
        )

        # 应只包含允许前缀的 URL
        assert all(
            u.startswith("https://cursor.com/docs") or u.startswith("https://cursor.com/cn/docs") for u in result
        ), f"所有 URL 应匹配允许前缀，实际: {result}"

        # 外域 URL 不应出现
        assert not any("github.com" in u for u in result)
        assert not any("external.com" in u for u in result)
        assert not any("stackoverflow.com" in u for u in result)
        assert not any("other-site.org" in u for u in result)

    def test_prefix_takes_priority_over_domain(self) -> None:
        """测试 allowed_url_prefixes 优先级高于 allowed_domains

        当 allowed_url_prefixes 不为空时，使用前缀匹配而非域名匹配。
        即使域名在 allowed_domains 中，如果不匹配前缀也会被过滤。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],  # 只允许 /docs 路径
            allowed_domains=["cursor.com"],  # 域名白名单（会被忽略）
            max_urls=10,
            exclude_patterns=[],
        )

        llms_content = """
https://cursor.com/docs/guide
https://cursor.com/pricing
https://cursor.com/blog/post
"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=llms_content,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 只有 /docs 前缀的 URL 应被保留
        assert len(result) == 1
        assert result[0] == "https://cursor.com/docs/guide"

        # /pricing 和 /blog 虽然是 cursor.com 但不匹配前缀
        assert not any("pricing" in u for u in result)
        assert not any("blog" in u for u in result)

    def test_empty_allowlist_allows_all(self) -> None:
        """测试空白名单允许所有 URL

        当 allowed_url_prefixes 和 allowed_domains 都为空时，
        所有 URL 都被允许（不进行域名过滤）。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[],
            allowed_domains=[],
            max_urls=10,
            exclude_patterns=[],  # 同时禁用排除规则
        )

        llms_content = """
https://cursor.com/docs/guide
https://github.com/cursor/repo
https://example.com/page
"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=llms_content,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 所有 URL 都应被保留
        assert len(result) == 3
        assert "https://cursor.com/docs/guide" in result
        assert "https://github.com/cursor/repo" in result
        assert "https://example.com/page" in result


# ============================================================
# Test: allowed_url_prefixes 专项测试
# ============================================================


class TestAllowedUrlPrefixes:
    """测试 allowed_url_prefixes 前缀匹配功能

    验证 allowed_url_prefixes 的优先级规则、前缀匹配逻辑、
    与 allowed_domains 的关系等核心行为。
    """

    def test_prefix_matching_exact(self) -> None:
        """测试精确前缀匹配"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://docs.example.com/api"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 精确匹配前缀
        assert is_allowed_doc_url("https://docs.example.com/api", config)
        assert is_allowed_doc_url("https://docs.example.com/api/v1", config)
        assert is_allowed_doc_url("https://docs.example.com/api/v2/endpoint", config)

        # 不匹配（前缀不同）
        assert not is_allowed_doc_url("https://docs.example.com/guide", config)
        assert not is_allowed_doc_url("https://docs.example.com/ap", config)

    def test_prefix_matching_multiple_prefixes(self) -> None:
        """测试多个前缀的匹配"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/cn/docs",
                "https://cursor.com/docs",
                "https://cursor.com/changelog",
            ],
            max_urls=10,
            exclude_patterns=[],
        )

        # 匹配任一前缀
        assert is_allowed_doc_url("https://cursor.com/cn/docs/cli", config)
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)
        assert is_allowed_doc_url("https://cursor.com/changelog/2026", config)

        # 不匹配任何前缀
        assert not is_allowed_doc_url("https://cursor.com/pricing", config)
        assert not is_allowed_doc_url("https://cursor.com/blog/post", config)

    def test_prefix_case_insensitive_domain(self) -> None:
        """测试前缀匹配时域名大小写不敏感

        URL 规范化会将域名转为小写，前缀也会被规范化。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 域名大小写应该不影响匹配（规范化后都是小写）
        assert is_allowed_doc_url("https://CURSOR.COM/docs/guide", config)
        assert is_allowed_doc_url("https://Cursor.Com/docs/api", config)

    def test_prefix_path_case_sensitive(self) -> None:
        """测试前缀匹配时路径大小写敏感

        URL 规范化保留路径大小写，路径匹配是大小写敏感的。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/Docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 路径大小写应该完全匹配
        assert is_allowed_doc_url("https://cursor.com/Docs/guide", config)
        # 路径大小写不同应该不匹配
        assert not is_allowed_doc_url("https://cursor.com/docs/guide", config)
        assert not is_allowed_doc_url("https://cursor.com/DOCS/guide", config)

    def test_prefix_normalization_trailing_slash(self) -> None:
        """测试前缀规范化：末尾斜杠处理"""
        # 配置前缀带末尾斜杠
        config_with_slash = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs/"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 配置前缀不带末尾斜杠
        config_without_slash = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 前缀规范化后应该都移除末尾斜杠
        # 两种配置应该产生相同的匹配行为
        test_url = "https://cursor.com/docs/guide"
        assert is_allowed_doc_url(test_url, config_with_slash)
        assert is_allowed_doc_url(test_url, config_without_slash)

    def test_prefix_normalization_duplicate_slashes(self) -> None:
        """测试前缀规范化：重复斜杠处理"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com//docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # URL 中的重复斜杠应该被规范化
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)
        assert is_allowed_doc_url("https://cursor.com//docs//guide", config)

    def test_prefix_priority_over_domain(self) -> None:
        """测试 allowed_url_prefixes 优先级高于 allowed_domains

        当 allowed_url_prefixes 不为空时，应该使用前缀匹配，
        而不是回退到 allowed_domains 的域名匹配。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],  # 只允许 /docs 路径
            allowed_domains=["cursor.com"],  # 域名白名单会被忽略
            max_urls=10,
            exclude_patterns=[],
        )

        # 匹配前缀的 URL
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)

        # 域名匹配但前缀不匹配的 URL 应该被拒绝
        assert not is_allowed_doc_url("https://cursor.com/pricing", config)
        assert not is_allowed_doc_url("https://cursor.com/blog", config)
        assert not is_allowed_doc_url("https://cursor.com/", config)

    def test_domain_fallback_when_prefix_empty(self) -> None:
        """测试 allowed_url_prefixes 为空时回退到 allowed_domains"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[],  # 为空
            allowed_domains=["cursor.com"],  # 域名白名单生效
            max_urls=10,
            exclude_patterns=[],
        )

        # 域名匹配应该生效
        assert is_allowed_doc_url("https://cursor.com/any/path", config)
        assert is_allowed_doc_url("https://cursor.com/docs", config)
        assert is_allowed_doc_url("https://cursor.com/pricing", config)

        # 其他域名应该被拒绝
        assert not is_allowed_doc_url("https://github.com/cursor", config)

    def test_prefix_with_select_urls_integration(self) -> None:
        """测试 allowed_url_prefixes 与 select_urls_to_fetch 的集成"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/cn/docs",
                "https://cursor.com/docs",
            ],
            max_urls=10,
            exclude_patterns=[],
        )

        llms_content = """
https://cursor.com/docs/overview
https://cursor.com/cn/docs/cli
https://cursor.com/pricing
https://github.com/cursor/repo
https://cursor.com/docs/api
"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=llms_content,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 只有匹配前缀的 URL 应该被选中
        assert len(result) == 3
        assert "https://cursor.com/docs/overview" in result
        assert "https://cursor.com/cn/docs/cli" in result
        assert "https://cursor.com/docs/api" in result

        # 不匹配前缀的 URL 不应出现
        assert not any("pricing" in u for u in result)
        assert not any("github.com" in u for u in result)

    def test_prefix_empty_url_rejected(self) -> None:
        """测试空 URL 被正确拒绝"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
        )

        assert not is_allowed_doc_url("", config)
        assert not is_allowed_doc_url("   ", config)

    def test_prefix_exclude_patterns_still_apply(self) -> None:
        """测试 allowed_url_prefixes 模式下 exclude_patterns 仍然生效"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            exclude_patterns=[r".*\.png$", r".*\.css$"],
            max_urls=10,
        )

        # 匹配前缀但匹配排除规则的 URL
        assert not is_allowed_doc_url("https://cursor.com/docs/image.png", config)
        assert not is_allowed_doc_url("https://cursor.com/docs/style.css", config)

        # 匹配前缀且不匹配排除规则的 URL
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)
        assert is_allowed_doc_url("https://cursor.com/docs/api.html", config)

    def test_prefix_with_subdomain(self) -> None:
        """测试前缀匹配与子域名"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/docs",
                "https://api.cursor.com/docs",
            ],
            max_urls=10,
            exclude_patterns=[],
        )

        # 主域名前缀匹配
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)

        # 子域名前缀匹配
        assert is_allowed_doc_url("https://api.cursor.com/docs/v1", config)

        # 不在允许前缀中的子域名
        assert not is_allowed_doc_url("https://blog.cursor.com/docs/post", config)

    def test_prefix_deterministic_matching(self) -> None:
        """测试前缀匹配的确定性（多次调用结果一致）"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/cn/docs",
                "https://cursor.com/docs",
            ],
            max_urls=10,
            exclude_patterns=[],
        )

        test_url = "https://cursor.com/docs/cli/overview"

        # 多次调用应该返回相同结果
        results = [is_allowed_doc_url(test_url, config) for _ in range(10)]
        assert all(r is True for r in results)

        # 对于不匹配的 URL 也应该返回一致结果
        reject_url = "https://github.com/cursor/repo"
        reject_results = [is_allowed_doc_url(reject_url, config) for _ in range(10)]
        assert all(r is False for r in reject_results)

    # ============================================================
    # 精确匹配测试（URL 与前缀规范化后完全相同）
    # ============================================================

    def test_prefix_exact_match_identical(self) -> None:
        """测试精确匹配：URL 与前缀完全相同"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # URL 与前缀完全相同
        assert is_allowed_doc_url("https://cursor.com/docs", config)

    def test_prefix_exact_match_with_trailing_slash_normalization(self) -> None:
        """测试精确匹配：末尾斜杠规范化后相同"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # URL 带末尾斜杠，规范化后与前缀相同
        assert is_allowed_doc_url("https://cursor.com/docs/", config)

    def test_prefix_exact_match_case_normalization(self) -> None:
        """测试精确匹配：域名大小写规范化后相同"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 域名大小写不同，规范化后相同
        assert is_allowed_doc_url("https://CURSOR.COM/docs", config)
        assert is_allowed_doc_url("https://Cursor.Com/docs", config)

    def test_prefix_exact_match_prefix_with_trailing_slash(self) -> None:
        """测试精确匹配：前缀配置带末尾斜杠"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs/"],  # 前缀带末尾斜杠
            max_urls=10,
            exclude_patterns=[],
        )

        # 前缀规范化后移除末尾斜杠，URL 应该匹配
        assert is_allowed_doc_url("https://cursor.com/docs", config)
        assert is_allowed_doc_url("https://cursor.com/docs/", config)

    # ============================================================
    # 子路径匹配测试（URL 以前缀开头但更长）
    # ============================================================

    def test_prefix_subpath_single_level(self) -> None:
        """测试子路径匹配：单级子路径"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 单级子路径
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)
        assert is_allowed_doc_url("https://cursor.com/docs/api", config)

    def test_prefix_subpath_multi_level(self) -> None:
        """测试子路径匹配：多级子路径"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 多级子路径
        assert is_allowed_doc_url("https://cursor.com/docs/cli/reference/params", config)
        assert is_allowed_doc_url("https://cursor.com/docs/a/b/c/d/e", config)

    def test_prefix_subpath_with_query_params(self) -> None:
        """测试子路径匹配：带查询参数"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 子路径带查询参数
        assert is_allowed_doc_url("https://cursor.com/docs/search?q=test", config)
        assert is_allowed_doc_url("https://cursor.com/docs/api?version=v2&lang=zh", config)

    def test_prefix_subpath_boundary_not_matching_partial(self) -> None:
        """测试子路径匹配边界：部分前缀不匹配"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # "docsify" 以 "docs" 开头，但不是子路径（应该用 /docs/ 或 /docs 作为边界）
        # 注意：当前实现使用 startswith，所以 /docsify 会匹配 /docs
        # 这是一个已知的边界行为
        # 如果需要严格边界匹配，应该在前缀末尾加 / 或在实现中检查边界
        result = is_allowed_doc_url("https://cursor.com/docsify", config)
        # 当前实现：/docsify startswith /docs = True
        assert result is True  # 记录当前行为

    def test_prefix_subpath_strict_boundary_with_slash(self) -> None:
        """测试子路径匹配：使用末尾斜杠前缀实现严格边界"""
        # 如果需要严格边界匹配，可以在前缀末尾加 /
        # 但由于规范化会移除末尾斜杠，这种方式不可行
        # 这是当前实现的设计决策
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs/"],  # 末尾斜杠会被规范化移除
            max_urls=10,
            exclude_patterns=[],
        )

        # 规范化后前缀变成 "https://cursor.com/docs"
        # /docsify 仍然会匹配
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)

    # ============================================================
    # exclude_patterns 与前缀匹配的交互测试
    # ============================================================

    def test_exclude_patterns_takes_priority_over_prefix(self) -> None:
        """测试 exclude_patterns 优先于前缀匹配

        即使 URL 匹配 allowed_url_prefixes，如果也匹配 exclude_patterns，
        应该返回 False。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            exclude_patterns=[
                r".*\.(png|jpg|gif|css|js)$",  # 静态资源
                r".*/internal/.*",  # 内部路径
            ],
            max_urls=10,
        )

        # 匹配前缀但也匹配排除规则
        assert not is_allowed_doc_url("https://cursor.com/docs/image.png", config)
        assert not is_allowed_doc_url("https://cursor.com/docs/style.css", config)
        assert not is_allowed_doc_url("https://cursor.com/docs/internal/secret", config)

        # 匹配前缀且不匹配排除规则
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)
        assert is_allowed_doc_url("https://cursor.com/docs/api.html", config)

    def test_exclude_patterns_with_exact_prefix_match(self) -> None:
        """测试 exclude_patterns 对精确匹配前缀的影响"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            exclude_patterns=[r".*docs$"],  # 排除以 docs 结尾的 URL
            max_urls=10,
        )

        # 精确匹配前缀，但也匹配排除规则
        assert not is_allowed_doc_url("https://cursor.com/docs", config)

        # 子路径不匹配排除规则
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)

    def test_exclude_patterns_regex_case_insensitive(self) -> None:
        """测试 exclude_patterns 正则大小写不敏感"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            exclude_patterns=[r".*\.PNG$"],  # 大写 PNG
            max_urls=10,
        )

        # 小写也应该被排除（正则使用 re.IGNORECASE）
        assert not is_allowed_doc_url("https://cursor.com/docs/image.png", config)
        assert not is_allowed_doc_url("https://cursor.com/docs/image.PNG", config)

    def test_exclude_patterns_empty_allows_all_matching_prefix(self) -> None:
        """测试 exclude_patterns 为空时所有匹配前缀的 URL 都通过"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            exclude_patterns=[],  # 空排除规则
            max_urls=10,
        )

        # 所有匹配前缀的 URL 都应该通过
        assert is_allowed_doc_url("https://cursor.com/docs/image.png", config)
        assert is_allowed_doc_url("https://cursor.com/docs/style.css", config)
        assert is_allowed_doc_url("https://cursor.com/docs/internal/secret", config)

    def test_exclude_patterns_with_multiple_prefixes(self) -> None:
        """测试 exclude_patterns 对多个前缀的统一应用"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/cn/docs",
                "https://cursor.com/docs",
            ],
            exclude_patterns=[r".*\.(png|css)$"],
            max_urls=10,
        )

        # 两个前缀都应该受 exclude_patterns 影响
        assert not is_allowed_doc_url("https://cursor.com/docs/image.png", config)
        assert not is_allowed_doc_url("https://cursor.com/cn/docs/image.png", config)

        # 不匹配排除规则的 URL 通过
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)
        assert is_allowed_doc_url("https://cursor.com/cn/docs/guide", config)

    def test_exclude_patterns_order_check_before_prefix(self) -> None:
        """测试 exclude_patterns 在前缀匹配之前检查

        验证 URL 先被 exclude_patterns 检查，再进行前缀匹配。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            exclude_patterns=[r".*\.js$"],
            max_urls=10,
        )

        # 即使不匹配前缀的 URL，如果匹配排除规则也返回 False
        # （实际上不匹配前缀也会返回 False，但排除规则先生效）
        assert not is_allowed_doc_url("https://cursor.com/other/script.js", config)

        # 匹配前缀但也匹配排除规则
        assert not is_allowed_doc_url("https://cursor.com/docs/app.js", config)

    # ============================================================
    # 相对路径前缀 + base_url 测试
    # ============================================================

    def test_relative_prefix_with_base_url(self) -> None:
        """测试 allowed_url_prefixes 使用相对路径 + base_url

        允许配置相对路径前缀，通过 base_url 参数补全为绝对路径。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["/docs", "/cn/docs", "/changelog"],
            max_urls=10,
            exclude_patterns=[],
        )

        base_url = "https://cursor.com"

        # 匹配相对前缀（通过 base_url 补全）
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config, base_url)
        assert is_allowed_doc_url("https://cursor.com/cn/docs/cli", config, base_url)
        assert is_allowed_doc_url("https://cursor.com/changelog/2026", config, base_url)

        # 不匹配的路径
        assert not is_allowed_doc_url("https://cursor.com/pricing", config, base_url)
        assert not is_allowed_doc_url("https://cursor.com/blog", config, base_url)

    def test_relative_prefix_mixed_with_absolute(self) -> None:
        """测试相对路径和绝对路径前缀混合使用"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "/docs",  # 相对路径
                "https://cursor.com/cn/docs",  # 绝对路径
                "/changelog",  # 相对路径
            ],
            max_urls=10,
            exclude_patterns=[],
        )

        base_url = "https://cursor.com"

        # 相对路径前缀匹配
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config, base_url)
        assert is_allowed_doc_url("https://cursor.com/changelog/2026", config, base_url)

        # 绝对路径前缀匹配
        assert is_allowed_doc_url("https://cursor.com/cn/docs/cli", config, base_url)

        # 不匹配
        assert not is_allowed_doc_url("https://cursor.com/pricing", config, base_url)

    def test_relative_prefix_without_base_url_stays_relative(self) -> None:
        """测试相对路径前缀在无 base_url 时保持相对格式

        当不提供 base_url 时，相对路径前缀规范化后仍为 https:///docs 格式，
        无法匹配任何正常的绝对 URL。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 不提供 base_url 时，相对前缀无法匹配绝对 URL
        assert not is_allowed_doc_url("https://cursor.com/docs/guide", config)

    def test_relative_url_candidate_with_base_url(self) -> None:
        """测试候选 URL 为相对路径时通过 base_url 补全"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        base_url = "https://cursor.com"

        # 相对路径候选 URL，通过 base_url 补全后匹配
        assert is_allowed_doc_url("/docs/guide", config, base_url)
        assert is_allowed_doc_url("/docs/api/v1", config, base_url)

        # 相对路径不匹配前缀
        assert not is_allowed_doc_url("/pricing", config, base_url)

    def test_relative_prefix_and_relative_url_both_with_base_url(self) -> None:
        """测试前缀和 URL 都是相对路径时，通过同一 base_url 补全

        确保确定性输出不变。
        """
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["/docs", "/cn/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        base_url = "https://cursor.com"

        # 相对路径 URL + 相对路径前缀，都通过 base_url 补全
        assert is_allowed_doc_url("/docs/guide", config, base_url)
        assert is_allowed_doc_url("/cn/docs/cli", config, base_url)

        # 确定性验证：多次调用结果一致
        results = [is_allowed_doc_url("/docs/guide", config, base_url) for _ in range(5)]
        assert all(r is True for r in results), "结果应确定性一致"

    def test_select_urls_with_relative_prefix_and_base_url(self) -> None:
        """测试 select_urls_to_fetch 使用相对路径前缀 + base_url"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["/docs", "/changelog"],
            max_urls=10,
            exclude_patterns=[],
        )

        base_url = "https://cursor.com"

        llms_content = """
https://cursor.com/docs/overview
https://cursor.com/docs/cli
https://cursor.com/changelog/2026
https://cursor.com/pricing
https://github.com/cursor/repo
"""
        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=llms_content,
            core_docs=[],
            keywords=[],
            config=config,
            base_url=base_url,
        )

        # 只有匹配相对前缀的 URL 被选中
        assert len(result) == 3
        assert "https://cursor.com/docs/overview" in result
        assert "https://cursor.com/docs/cli" in result
        assert "https://cursor.com/changelog/2026" in result

        # 不匹配的被过滤
        assert not any("pricing" in u for u in result)
        assert not any("github.com" in u for u in result)

    def test_select_urls_with_relative_candidate_urls(self) -> None:
        """测试 select_urls_to_fetch 候选 URL 为相对路径"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            max_urls=10,
            exclude_patterns=[],
        )

        base_url = "https://cursor.com"

        # 候选 URL 包含相对路径
        llms_content = """
/docs/overview
/docs/cli
/changelog/2026
"""
        result = select_urls_to_fetch(
            changelog_links=["/docs/latest"],
            related_doc_urls=["/docs/guide", "/pricing"],
            llms_txt_content=llms_content,
            core_docs=["/docs/getting-started"],
            keywords=[],
            config=config,
            base_url=base_url,
        )

        # 验证相对路径被正确补全并匹配
        # 只有 /docs/* 路径匹配前缀
        for url in result:
            assert url.startswith("https://cursor.com/docs"), f"URL 应匹配前缀: {url}"

        # 确保 /changelog 和 /pricing 被过滤
        assert not any("changelog" in u for u in result)
        assert not any("pricing" in u for u in result)

    def test_relative_prefix_deterministic_output(self) -> None:
        """测试相对路径前缀场景下输出的确定性"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["/docs", "/cn/docs"],
            max_urls=5,
            exclude_patterns=[],
        )

        base_url = "https://cursor.com"

        llms_content = """
https://cursor.com/docs/a
https://cursor.com/docs/b
https://cursor.com/cn/docs/c
"""
        # 多次调用
        results = [
            select_urls_to_fetch(
                changelog_links=[],
                related_doc_urls=[],
                llms_txt_content=llms_content,
                core_docs=[],
                keywords=[],
                config=config,
                base_url=base_url,
            )
            for _ in range(5)
        ]

        # 所有结果应相同（确定性）
        assert all(r == results[0] for r in results), "输出应确定性一致"


# ============================================================
# Test: allowed_url_prefixes 与 allowed_domains 语义区分
# ============================================================


class TestPrefixVsDomainSemantics:
    """测试 allowed_url_prefixes 与 allowed_domains 的语义区分

    - allowed_domains: 通用场景的域名过滤（如 python.org 允许所有子域名和路径）
    - allowed_url_prefixes: Cursor 文档更新流程的精确前缀过滤
    """

    def test_domain_filter_allows_any_path(self) -> None:
        """测试 allowed_domains 允许域名下的任意路径（通用场景）"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[],  # 不使用前缀
            allowed_domains=["python.org"],
            max_urls=10,
            exclude_patterns=[],
        )

        # 任意路径都应该被允许
        assert is_allowed_doc_url("https://docs.python.org/3/library/", config)
        assert is_allowed_doc_url("https://python.org/about/", config)
        assert is_allowed_doc_url("https://wiki.python.org/moin/", config)

    def test_prefix_filter_restricts_to_specific_paths(self) -> None:
        """测试 allowed_url_prefixes 限制到特定路径（Cursor 文档场景）"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/cn/docs",
                "https://cursor.com/docs",
                "https://cursor.com/cn/changelog",
                "https://cursor.com/changelog",
            ],
            max_urls=10,
            exclude_patterns=[],
        )

        # 匹配前缀的路径
        assert is_allowed_doc_url("https://cursor.com/docs/cli", config)
        assert is_allowed_doc_url("https://cursor.com/changelog/2026", config)

        # 不匹配前缀的路径（即使是 cursor.com 域名）
        assert not is_allowed_doc_url("https://cursor.com/pricing", config)
        assert not is_allowed_doc_url("https://cursor.com/about", config)
        assert not is_allowed_doc_url("https://cursor.com/blog/post", config)

    def test_mixed_scenario_prefix_takes_priority(self) -> None:
        """测试混合场景下 allowed_url_prefixes 优先（即使两者都配置）"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://cursor.com/docs"],
            allowed_domains=["cursor.com", "github.com"],  # 会被忽略
            max_urls=10,
            exclude_patterns=[],
        )

        # 前缀匹配生效
        assert is_allowed_doc_url("https://cursor.com/docs/guide", config)

        # 域名白名单不生效（因为 allowed_url_prefixes 有值）
        assert not is_allowed_doc_url("https://cursor.com/pricing", config)
        assert not is_allowed_doc_url("https://github.com/cursor/repo", config)

    def test_cursor_doc_flow_uses_prefix(self) -> None:
        """测试 Cursor 文档更新流程应使用 allowed_url_prefixes"""
        # 模拟 Cursor 文档更新配置
        cursor_doc_prefixes = [
            "https://cursor.com/cn/docs",
            "https://cursor.com/docs",
            "https://cursor.com/cn/changelog",
            "https://cursor.com/changelog",
        ]

        config = DocURLStrategyConfig(
            allowed_url_prefixes=cursor_doc_prefixes,
            max_urls=20,
        )

        # 有效的 Cursor 文档 URL
        valid_urls = [
            "https://cursor.com/cn/docs/cli/overview",
            "https://cursor.com/docs/cli/reference",
            "https://cursor.com/cn/changelog/2026-01",
            "https://cursor.com/changelog/latest",
        ]
        for url in valid_urls:
            assert is_allowed_doc_url(url, config), f"应该允许: {url}"

        # 非 Cursor 文档 URL（即使是 cursor.com 域名）
        invalid_urls = [
            "https://cursor.com/pricing",
            "https://cursor.com/blog/post",
            "https://cursor.com/download",
            "https://github.com/cursor/repo",
        ]
        for url in invalid_urls:
            assert not is_allowed_doc_url(url, config), f"应该拒绝: {url}"

    def test_general_use_case_uses_domain(self) -> None:
        """测试通用场景应使用 allowed_domains"""
        # 模拟通用文档抓取配置
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[],  # 不限制前缀
            allowed_domains=["python.org", "nodejs.org", "reactjs.org"],
            max_urls=20,
            exclude_patterns=[],
        )

        # 任意路径都应该被允许
        valid_urls = [
            "https://docs.python.org/3/library/asyncio.html",
            "https://python.org/dev/peps/",
            "https://nodejs.org/en/docs/",
            "https://reactjs.org/docs/getting-started.html",
        ]
        for url in valid_urls:
            assert is_allowed_doc_url(url, config), f"应该允许: {url}"

        # 其他域名被拒绝
        invalid_urls = [
            "https://github.com/python/cpython",
            "https://stackoverflow.com/questions/python",
        ]
        for url in invalid_urls:
            assert not is_allowed_doc_url(url, config), f"应该拒绝: {url}"


# ============================================================
# 样例数据：llms.txt 格式（供测试复用）
# ============================================================

SAMPLE_LLMS_TXT_STANDARD = """# Documentation Index

## Getting Started
[Quick Start](https://docs.example.com/quickstart)
[Installation Guide](https://docs.example.com/install)

## API Reference
https://docs.example.com/api/v1
https://docs.example.com/api/v2

## Tutorials
- [Basic Tutorial](https://docs.example.com/tutorials/basic)
- [Advanced Guide](https://docs.example.com/tutorials/advanced)
"""

SAMPLE_LLMS_TXT_WITH_EXTERNAL = """# Mixed Content

## Internal Docs
https://docs.example.com/guide
https://docs.example.com/reference

## External Resources (typically filtered)
https://github.com/example/project
https://stackoverflow.com/questions/example
https://npmjs.com/package/example-lib

## More Internal
[API](https://api.example.com/docs)
"""

SAMPLE_LLMS_TXT_MINIMAL = """https://example.com/doc1
https://example.com/doc2
https://example.com/doc3
"""


# ============================================================
# Test: llms.txt 样例数据复用测试
# ============================================================


class TestLlmsTxtSampleDataReuse:
    """测试复用 llms.txt 样例数据，确保不依赖真实网络"""

    def test_parse_standard_llms_txt(self) -> None:
        """测试解析标准格式 llms.txt"""
        urls = parse_llms_txt_urls(SAMPLE_LLMS_TXT_STANDARD)

        # 应解析出多个 URL
        assert len(urls) >= 5, f"应至少解析出 5 个 URL，实际 {len(urls)}"

        # 验证包含预期的 URL
        assert any("quickstart" in u for u in urls)
        assert any("api/v1" in u for u in urls)

    def test_parse_llms_txt_with_external_urls(self) -> None:
        """测试解析包含外域链接的 llms.txt"""
        urls = parse_llms_txt_urls(SAMPLE_LLMS_TXT_WITH_EXTERNAL)

        # 应解析出所有 URL（包括外域）
        assert len(urls) >= 5

        # 验证包含内外域 URL
        assert any("docs.example.com" in u for u in urls)
        assert any("github.com" in u for u in urls)

    def test_parse_minimal_llms_txt(self) -> None:
        """测试解析最小格式 llms.txt"""
        urls = parse_llms_txt_urls(SAMPLE_LLMS_TXT_MINIMAL)

        assert len(urls) == 3
        assert "https://example.com/doc1" in urls
        assert "https://example.com/doc2" in urls
        assert "https://example.com/doc3" in urls

    def test_filter_external_urls_from_sample(self) -> None:
        """测试从样例数据过滤外域 URL"""
        config = DocURLStrategyConfig(
            allowed_domains=["docs.example.com", "api.example.com"],
            exclude_patterns=[],
            max_urls=20,
        )

        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=SAMPLE_LLMS_TXT_WITH_EXTERNAL,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 验证外域被过滤
        assert not any("github.com" in u for u in result)
        assert not any("stackoverflow.com" in u for u in result)
        assert not any("npmjs.com" in u for u in result)

        # 验证内域保留
        assert any("docs.example.com" in u for u in result)

    def test_sample_llms_txt_with_prefix_filter(self) -> None:
        """测试使用前缀过滤样例 llms.txt"""
        config = DocURLStrategyConfig(
            allowed_url_prefixes=["https://docs.example.com/api"],
            max_urls=10,
        )

        result = select_urls_to_fetch(
            changelog_links=[],
            related_doc_urls=[],
            llms_txt_content=SAMPLE_LLMS_TXT_STANDARD,
            core_docs=[],
            keywords=[],
            config=config,
        )

        # 只保留匹配前缀的 URL
        for url in result:
            assert url.startswith("https://docs.example.com/api"), f"URL 应匹配前缀: {url}"

    def test_sample_data_deterministic_parsing(self) -> None:
        """测试样例数据解析结果的确定性"""
        # 多次解析应得到相同结果
        results = [parse_llms_txt_urls(SAMPLE_LLMS_TXT_STANDARD) for _ in range(5)]

        # 所有结果应相同
        assert all(r == results[0] for r in results), "解析结果应确定性一致"

    def test_sample_llms_txt_markdown_links_extraction(self) -> None:
        """测试从 Markdown 链接格式提取 URL"""
        content = """
[Link 1](https://example.com/page1)
[Link 2](https://example.com/page2)
Plain URL: https://example.com/page3
"""
        urls = parse_llms_txt_urls(content)

        # 应提取所有 URL
        assert len(urls) >= 3
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls
        assert "https://example.com/page3" in urls


# ============================================================
# Test: HTML 样例数据测试
# ============================================================


SAMPLE_HTML_DOC_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Documentation</title>
</head>
<body>
    <nav>Navigation Menu</nav>
    <main>
        <h1>Getting Started</h1>
        <p>Welcome to the documentation.</p>
        <h2>Installation</h2>
        <p>Run the following command:</p>
        <code>pip install example</code>
        <h2>Usage</h2>
        <p>Basic usage example:</p>
        <a href="https://docs.example.com/api">API Reference</a>
        <a href="https://github.com/example/repo">Source Code</a>
    </main>
    <footer>Footer Content</footer>
</body>
</html>
"""


class TestHtmlSampleDataReuse:
    """测试复用 HTML 样例数据"""

    def test_extract_links_from_sample_html(self) -> None:
        """测试从样例 HTML 提取链接"""
        # 使用简单的正则提取（不依赖 BeautifulSoup 测试）
        import re

        links = re.findall(r'href="([^"]+)"', SAMPLE_HTML_DOC_PAGE)

        assert len(links) >= 2
        assert "https://docs.example.com/api" in links
        assert "https://github.com/example/repo" in links

    def test_sample_html_contains_expected_structure(self) -> None:
        """测试样例 HTML 包含预期结构"""
        assert "<title>" in SAMPLE_HTML_DOC_PAGE
        assert "<main>" in SAMPLE_HTML_DOC_PAGE
        assert "<h1>" in SAMPLE_HTML_DOC_PAGE
        assert "<h2>" in SAMPLE_HTML_DOC_PAGE

    def test_filter_links_from_sample_html(self) -> None:
        """测试从样例 HTML 过滤链接"""
        import re

        links = re.findall(r'href="(https?://[^"]+)"', SAMPLE_HTML_DOC_PAGE)

        # 使用配置过滤
        config = DocURLStrategyConfig(
            allowed_domains=["docs.example.com"],
            exclude_patterns=[],
            max_urls=10,
        )

        filtered = [url for url in links if is_allowed_doc_url(url, config)]

        # 应只保留 docs.example.com
        assert len(filtered) == 1
        assert "docs.example.com" in filtered[0]
        assert not any("github.com" in u for u in filtered)


# ============================================================
# Test: 不依赖网络的集成测试
# ============================================================


class TestOfflineIntegration:
    """不依赖网络的集成测试"""

    def test_full_url_strategy_workflow_offline(self) -> None:
        """测试完整 URL 策略工作流（离线）"""
        config = DocURLStrategyConfig(
            allowed_domains=["cursor.com"],
            max_urls=10,
            fallback_core_docs_count=3,
            exclude_patterns=[],
        )

        # 模拟各来源数据（不需要网络）
        changelog_links = [
            "https://cursor.com/changelog/2026-01",
            "https://github.com/cursor/releases",  # 外域
        ]

        related_doc_urls = [
            "https://cursor.com/docs/guide",
            "https://stackoverflow.com/cursor",  # 外域
        ]

        llms_txt_content = """
https://cursor.com/docs/api
https://cursor.com/docs/reference
https://external.com/docs  # 外域
"""

        core_docs = [
            "https://cursor.com/docs/getting-started",
            "https://other.org/docs",  # 外域
        ]

        result = select_urls_to_fetch(
            changelog_links=changelog_links,
            related_doc_urls=related_doc_urls,
            llms_txt_content=llms_txt_content,
            core_docs=core_docs,
            keywords=[],
            config=config,
        )

        # 验证结果
        assert len(result) >= 1

        # 所有结果应来自 cursor.com
        for url in result:
            assert "cursor.com" in url, f"URL 应来自 cursor.com: {url}"

        # 外域应被过滤
        assert not any("github.com" in u for u in result)
        assert not any("stackoverflow.com" in u for u in result)
        assert not any("external.com" in u for u in result)
        assert not any("other.org" in u for u in result)

    def test_url_normalization_workflow_offline(self) -> None:
        """测试 URL 规范化工作流（离线）"""
        # 测试各种需要规范化的 URL
        test_cases = [
            ("https://EXAMPLE.COM/page", "https://example.com/page"),
            ("https://example.com/page/", "https://example.com/page"),
            ("https://example.com/page#section", "https://example.com/page"),
            ("https://example.com//docs//page", "https://example.com/docs/page"),
        ]

        for original, expected in test_cases:
            result = normalize_url(original)
            assert result == expected, f"规范化失败: {original} -> {result}, 期望 {expected}"

    def test_deduplication_workflow_offline(self) -> None:
        """测试去重工作流（离线）"""
        urls = [
            "https://example.com/page",
            "https://example.com/page/",  # 末尾斜杠
            "https://EXAMPLE.COM/page",  # 大写域名
            "https://example.com/page#section",  # 带 fragment
            "https://example.com/other",  # 不同页面
        ]

        result = deduplicate_urls(urls, normalize_before_dedup=True)

        # 前 4 个应该去重为 1 个
        assert len(result) == 2
        assert "https://example.com/page" in result
        assert "https://example.com/other" in result


# ============================================================
# Test: _matches_allowlist 函数语义测试
# ============================================================


class TestMatchesAllowlistSemantics:
    """测试 _matches_allowlist 函数的语义

    allowlist 支持两种格式：
    1. 域名格式（如 "github.com"）：匹配该域名及其子域名
    2. URL 前缀格式（如 "https://github.com/cursor"）：精确前缀匹配
    """

    def test_matches_allowlist_domain_exact_match(self) -> None:
        """测试域名格式的精确匹配"""
        from knowledge.doc_url_strategy import _matches_allowlist

        url = "https://github.com/cursor/repo"
        allowlist = ["github.com"]

        assert _matches_allowlist(url, allowlist) is True

    def test_matches_allowlist_domain_subdomain_match(self) -> None:
        """测试域名格式的子域名匹配"""
        from knowledge.doc_url_strategy import _matches_allowlist

        url = "https://api.github.com/v1/repos"
        allowlist = ["github.com"]

        # 子域名 api.github.com 应匹配 github.com
        assert _matches_allowlist(url, allowlist) is True

    def test_matches_allowlist_domain_no_match(self) -> None:
        """测试域名格式的不匹配场景"""
        from knowledge.doc_url_strategy import _matches_allowlist

        url = "https://gitlab.com/cursor/repo"
        allowlist = ["github.com"]

        assert _matches_allowlist(url, allowlist) is False

    def test_matches_allowlist_url_prefix_exact_match(self) -> None:
        """测试 URL 前缀格式的精确匹配"""
        from knowledge.doc_url_strategy import _matches_allowlist

        url = "https://github.com/cursor/repo"
        allowlist = ["https://github.com/cursor"]

        assert _matches_allowlist(url, allowlist) is True

    def test_matches_allowlist_url_prefix_no_match_different_path(self) -> None:
        """测试 URL 前缀格式不匹配不同路径"""
        from knowledge.doc_url_strategy import _matches_allowlist

        url = "https://github.com/other/repo"
        allowlist = ["https://github.com/cursor"]

        # /other/repo 不匹配 /cursor 前缀
        assert _matches_allowlist(url, allowlist) is False

    def test_matches_allowlist_mixed_formats(self) -> None:
        """测试混合格式的 allowlist"""
        from knowledge.doc_url_strategy import _matches_allowlist

        allowlist = [
            "github.com",  # 域名格式
            "https://docs.example.com/api",  # URL 前缀格式
        ]

        # github.com 域名匹配
        assert _matches_allowlist("https://github.com/any/repo", allowlist) is True

        # docs.example.com/api 前缀匹配
        assert _matches_allowlist("https://docs.example.com/api/v1", allowlist) is True

        # docs.example.com/other 不匹配
        assert _matches_allowlist("https://docs.example.com/other", allowlist) is False

        # other.com 不匹配
        assert _matches_allowlist("https://other.com/page", allowlist) is False

    def test_matches_allowlist_case_insensitive(self) -> None:
        """测试大小写不敏感"""
        from knowledge.doc_url_strategy import _matches_allowlist

        url = "https://GITHUB.COM/Cursor/Repo"
        allowlist = ["github.com"]

        assert _matches_allowlist(url, allowlist) is True

    def test_matches_allowlist_empty_allowlist(self) -> None:
        """测试空 allowlist 返回 False"""
        from knowledge.doc_url_strategy import _matches_allowlist

        url = "https://github.com/cursor/repo"
        allowlist: list[str] = []

        assert _matches_allowlist(url, allowlist) is False

    def test_matches_allowlist_url_prefix_with_trailing_slash(self) -> None:
        """测试 URL 前缀带末尾斜杠的匹配"""
        from knowledge.doc_url_strategy import _matches_allowlist

        url = "https://github.com/cursor/repo"
        allowlist = ["https://github.com/cursor/"]

        # 需要验证是否匹配（前缀匹配语义）
        # 当前实现使用 startswith，所以带斜杠的前缀应该匹配
        assert _matches_allowlist(url, allowlist) is True


class TestApplyFetchPolicyExternalLinkModes:
    """测试 apply_fetch_policy 函数的三种 external_link_mode 行为"""

    def test_apply_fetch_policy_record_only_mode(self) -> None:
        """测试 record_only 模式：外链从 urls_to_fetch 移除但记录到 external_links_recorded"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
            "https://external.com/guide",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
        )

        # urls_to_fetch 应只包含内链
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" not in result.urls_to_fetch
        assert "https://external.com/guide" not in result.urls_to_fetch

        # external_links_recorded 应包含外链
        assert "https://github.com/cursor/repo" in result.external_links_recorded
        assert "https://external.com/guide" in result.external_links_recorded

    def test_apply_fetch_policy_skip_all_mode(self) -> None:
        """测试 skip_all 模式：外链既不抓取也不记录"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="skip_all",
            primary_domains=["cursor.com"],
        )

        # urls_to_fetch 应只包含内链
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" not in result.urls_to_fetch

        # skip_all 模式下 external_links_recorded 应为空
        assert len(result.external_links_recorded) == 0

        # filtered_urls 应记录被跳过的原因
        assert len(result.filtered_urls) == 1
        assert result.filtered_urls[0]["reason"] == "external_link_skip_all"

    def test_apply_fetch_policy_fetch_allowlist_mode(self) -> None:
        """测试 fetch_allowlist 模式：匹配 allowlist 的外链允许抓取"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
            "https://external.com/guide",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="fetch_allowlist",
            primary_domains=["cursor.com"],
            external_link_allowlist=["github.com"],  # 只允许 github.com
        )

        # urls_to_fetch 应包含内链和匹配 allowlist 的外链
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" in result.urls_to_fetch  # 匹配 allowlist

        # 不匹配 allowlist 的外链不应在 urls_to_fetch
        assert "https://external.com/guide" not in result.urls_to_fetch

        # external_links_recorded 应包含不匹配的外链
        assert "https://external.com/guide" in result.external_links_recorded

    def test_apply_fetch_policy_fetch_allowlist_url_prefix(self) -> None:
        """测试 fetch_allowlist 模式支持 URL 前缀格式"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://github.com/cursor/repo",
            "https://github.com/other/repo",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="fetch_allowlist",
            primary_domains=[],  # 无主域名，全部视为外链
            external_link_allowlist=["https://github.com/cursor"],  # URL 前缀
        )

        # /cursor/repo 匹配 URL 前缀
        assert "https://github.com/cursor/repo" in result.urls_to_fetch

        # /other/repo 不匹配
        assert "https://github.com/other/repo" not in result.urls_to_fetch
        assert "https://github.com/other/repo" in result.external_links_recorded


# ============================================================
# Test: 默认配置行为测试（urls_to_fetch 仅来自 cursor docs/changelog）
# ============================================================


class TestDefaultConfigBehavior:
    """测试默认配置下的 URL 策略行为

    验证默认配置：
    - urls_to_fetch 只来自 cursor docs/changelog 前缀范围
    - 外链使用 record_only 模式（仅记录不抓取）
    """

    def test_default_config_urls_to_fetch_only_cursor_prefixes(self) -> None:
        """测试默认配置下 urls_to_fetch 仅包含 cursor docs/changelog 前缀的 URL"""
        # 使用默认配置（allowed_domains=["cursor.com"]）
        config = DocURLStrategyConfig()

        # 模拟各来源数据
        changelog_links = [
            "https://cursor.com/changelog/2026-01",
            "https://cursor.com/cn/changelog/latest",
            "https://github.com/cursor/releases",  # 外域
        ]

        related_doc_urls = [
            "https://cursor.com/docs/cli/overview",
            "https://cursor.com/cn/docs/guide",
            "https://external.com/docs",  # 外域
        ]

        llms_txt_content = """
https://cursor.com/docs/api
https://cursor.com/cn/docs/reference
https://external.com/guide
"""

        core_docs = [
            "https://cursor.com/docs/getting-started",
        ]

        result = select_urls_to_fetch(
            changelog_links=changelog_links,
            related_doc_urls=related_doc_urls,
            llms_txt_content=llms_txt_content,
            core_docs=core_docs,
            keywords=[],
            config=config,
        )

        # 验证所有结果都来自 cursor.com 域名
        for url in result:
            assert "cursor.com" in url, f"URL 应来自 cursor.com: {url}"

        # 验证外域被过滤
        assert not any("github.com" in u for u in result)
        assert not any("external.com" in u for u in result)

    def test_default_config_external_links_record_only(self) -> None:
        """测试默认配置下外链使用 record_only 模式"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/cn/docs/guide",
            "https://github.com/cursor/repo",
            "https://external.com/guide",
        ]

        # 使用默认 external_link_mode（record_only）
        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",  # 默认值
            primary_domains=["cursor.com"],
        )

        # urls_to_fetch 应只包含内链
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://cursor.com/cn/docs/guide" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" not in result.urls_to_fetch
        assert "https://external.com/guide" not in result.urls_to_fetch

        # external_links_recorded 应包含所有外链
        assert "https://github.com/cursor/repo" in result.external_links_recorded
        assert "https://external.com/guide" in result.external_links_recorded

        # filtered_urls 应记录原因为 external_link_record_only
        for filtered in result.filtered_urls:
            assert filtered["reason"] == "external_link_record_only"

    def test_default_config_allowed_url_prefixes_semantic(self) -> None:
        """测试默认配置下 allowed_url_prefixes 的语义（仅 cursor docs/changelog）

        当 allowed_url_prefixes 配置为 Cursor 官方文档前缀时，
        非 docs/changelog 路径应被过滤（如 /pricing, /about）
        """
        cursor_doc_prefixes = [
            "https://cursor.com/cn/docs",
            "https://cursor.com/docs",
            "https://cursor.com/cn/changelog",
            "https://cursor.com/changelog",
        ]

        config = DocURLStrategyConfig(
            allowed_url_prefixes=cursor_doc_prefixes,
            max_urls=20,
        )

        # 测试 docs/changelog 路径应被允许
        valid_urls = [
            "https://cursor.com/docs/cli/overview",
            "https://cursor.com/cn/docs/guide",
            "https://cursor.com/changelog/2026-01",
            "https://cursor.com/cn/changelog/latest",
        ]
        for url in valid_urls:
            assert is_allowed_doc_url(url, config), f"应该允许: {url}"

        # 测试其他路径应被拒绝
        invalid_urls = [
            "https://cursor.com/pricing",
            "https://cursor.com/about",
            "https://cursor.com/blog/post",
            "https://cursor.com/download",
        ]
        for url in invalid_urls:
            assert not is_allowed_doc_url(url, config), f"应该拒绝: {url}"


# ============================================================
# Test: url_strategy 与 fetch_policy 配置交互测试
# ============================================================


class TestUrlStrategyFetchPolicyInteraction:
    """测试 url_strategy 与 fetch_policy 的配置交互

    关键点：
    - url_strategy.allowed_domains 可配置为允许更多域名
    - 但 fetch_policy.external_link_mode=record_only 时，外链仍不会被抓取
    """

    def test_url_strategy_allows_more_domains_but_fetch_policy_record_only(
        self,
    ) -> None:
        """测试 url_strategy 允许更多域名但 fetch_policy=record_only 时外链不抓取

        场景：
        - url_strategy.allowed_domains 包含 ["cursor.com", "github.com"]
        - fetch_policy.external_link_mode = "record_only"
        - 结果：github.com 的 URL 通过 url_strategy 选择，但被 fetch_policy 过滤
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        # url_strategy 允许 cursor.com 和 github.com
        config = DocURLStrategyConfig(
            allowed_domains=["cursor.com", "github.com"],
            exclude_patterns=[],
            max_urls=20,
        )

        # 模拟 URL 选择结果（url_strategy 阶段）
        selected_urls = select_urls_to_fetch(
            changelog_links=["https://cursor.com/changelog"],
            related_doc_urls=[
                "https://github.com/cursor/repo",
                "https://github.com/cursor/docs",
            ],
            llms_txt_content="",
            core_docs=["https://cursor.com/docs/guide"],
            keywords=[],
            config=config,
        )

        # url_strategy 阶段应包含 github.com（因为在 allowed_domains 中）
        assert any("github.com" in u for u in selected_urls)
        assert any("cursor.com" in u for u in selected_urls)

        # 应用 fetch_policy（record_only 模式）
        result = apply_fetch_policy(
            urls=selected_urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],  # 主域名仅 cursor.com
        )

        # fetch_policy 阶段：github.com 应被记录但不抓取
        assert not any("github.com" in u for u in result.urls_to_fetch)
        assert any("github.com" in u for u in result.external_links_recorded)

        # cursor.com 应在 urls_to_fetch 中
        assert any("cursor.com" in u for u in result.urls_to_fetch)

    def test_url_strategy_prefix_filter_with_fetch_policy(self) -> None:
        """测试 url_strategy 前缀过滤与 fetch_policy 的协同

        场景：
        - url_strategy.allowed_url_prefixes 限制为 Cursor 文档前缀
        - fetch_policy.external_link_mode = "record_only"
        - 结果：非 docs/changelog 路径在 url_strategy 阶段就被过滤
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        # url_strategy 仅允许 docs/changelog 前缀
        config = DocURLStrategyConfig(
            allowed_url_prefixes=[
                "https://cursor.com/docs",
                "https://cursor.com/changelog",
            ],
            max_urls=20,
        )

        # 混合 URL 列表
        all_urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/changelog/2026",
            "https://cursor.com/pricing",  # 应被 url_strategy 过滤
            "https://github.com/cursor/repo",  # 应被 url_strategy 过滤
        ]

        # url_strategy 阶段过滤
        selected = []
        for url in all_urls:
            if is_allowed_doc_url(url, config):
                selected.append(url)

        # 只有 docs 和 changelog 应通过
        assert "https://cursor.com/docs/cli" in selected
        assert "https://cursor.com/changelog/2026" in selected
        assert "https://cursor.com/pricing" not in selected
        assert "https://github.com/cursor/repo" not in selected

        # 应用 fetch_policy（此时应无外链）
        result = apply_fetch_policy(
            urls=selected,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
        )

        # 所有选中的 URL 都应在 urls_to_fetch 中（无外链）
        assert len(result.urls_to_fetch) == 2
        assert len(result.external_links_recorded) == 0


# ============================================================
# Test: fetch_policy 各模式行为测试
# ============================================================


class TestFetchPolicyModes:
    """测试 fetch_policy 的三种模式行为

    - skip_all: 外链既不抓取也不记录
    - record_only: 外链仅记录不抓取（默认）
    - fetch_allowlist: 仅 allowlist 内的外链可抓取
    """

    def test_skip_all_mode_external_links_not_recorded(self) -> None:
        """测试 skip_all 模式下外链不记录

        验证：
        - external_links_recorded 应为空
        - filtered_urls 应包含原因 "external_link_skip_all"
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
            "https://external.com/guide",
            "https://stackoverflow.com/questions",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="skip_all",
            primary_domains=["cursor.com"],
        )

        # urls_to_fetch 应只包含内链
        assert result.urls_to_fetch == ["https://cursor.com/docs/cli"]

        # skip_all 模式下 external_links_recorded 应为空
        assert len(result.external_links_recorded) == 0

        # filtered_urls 应包含所有外链，原因为 skip_all
        assert len(result.filtered_urls) == 3
        for filtered in result.filtered_urls:
            assert filtered["reason"] == "external_link_skip_all"
            assert filtered["url"] in [
                "https://github.com/cursor/repo",
                "https://external.com/guide",
                "https://stackoverflow.com/questions",
            ]

    def test_fetch_allowlist_mode_only_allowlist_urls_fetched(self) -> None:
        """测试 fetch_allowlist 模式下仅 allowlist 内外链可抓取

        验证：
        - 匹配 allowlist 的外链在 urls_to_fetch 中
        - 不匹配的外链在 external_links_recorded 中
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
            "https://github.com/other/repo",
            "https://external.com/guide",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="fetch_allowlist",
            primary_domains=["cursor.com"],
            external_link_allowlist=[
                "github.com/cursor",  # 允许 github.com/cursor 路径
            ],
        )

        # 内链和匹配 allowlist 的外链应在 urls_to_fetch
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" in result.urls_to_fetch

        # 不匹配 allowlist 的外链不应在 urls_to_fetch
        assert "https://github.com/other/repo" not in result.urls_to_fetch
        assert "https://external.com/guide" not in result.urls_to_fetch

        # 不匹配的外链应在 external_links_recorded
        assert "https://github.com/other/repo" in result.external_links_recorded
        assert "https://external.com/guide" in result.external_links_recorded

    def test_fetch_allowlist_domain_format(self) -> None:
        """测试 fetch_allowlist 模式支持纯域名格式"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://github.com/any/repo",
            "https://github.com/other/project",
            "https://gitlab.com/repo",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="fetch_allowlist",
            primary_domains=[],  # 无主域名
            external_link_allowlist=["github.com"],  # 域名格式
        )

        # 所有 github.com 都应匹配
        assert "https://github.com/any/repo" in result.urls_to_fetch
        assert "https://github.com/other/project" in result.urls_to_fetch

        # gitlab.com 不匹配
        assert "https://gitlab.com/repo" not in result.urls_to_fetch
        assert "https://gitlab.com/repo" in result.external_links_recorded

    def test_fetch_allowlist_url_prefix_format(self) -> None:
        """测试 fetch_allowlist 模式支持完整 URL 前缀格式"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://github.com/cursor/repo",
            "https://github.com/cursor/docs",
            "https://github.com/other/repo",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="fetch_allowlist",
            primary_domains=[],
            external_link_allowlist=["https://github.com/cursor"],  # URL 前缀
        )

        # 匹配前缀的应在 urls_to_fetch
        assert "https://github.com/cursor/repo" in result.urls_to_fetch
        assert "https://github.com/cursor/docs" in result.urls_to_fetch

        # 不匹配前缀的不应在 urls_to_fetch
        assert "https://github.com/other/repo" not in result.urls_to_fetch


# ============================================================
# Test: allowed_path_prefixes 内链路径 gate 测试
# ============================================================


class TestEnforcePathPrefixes:
    """测试 enforce_path_prefixes 内链路径前缀检查功能

    覆盖场景：
    - enforce_path_prefixes=False（Phase A 行为）
    - enforce_path_prefixes=True（Phase B 启用内链路径 gate）
    - 外链 allowlist 与内链 gate 的组合矩阵
    """

    def test_enforce_disabled_all_internal_links_pass(self) -> None:
        """测试 enforce_path_prefixes=False 时，所有内链都通过（Phase A 行为）"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/pricing",  # 不在 allowed_path_prefixes 中
            "https://cursor.com/about",  # 不在 allowed_path_prefixes 中
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["docs", "cn/docs"],
            enforce_path_prefixes=False,  # Phase A：不执行路径检查
        )

        # 所有内链都应通过（不执行路径检查）
        assert len(result.urls_to_fetch) == 3
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://cursor.com/pricing" in result.urls_to_fetch
        assert "https://cursor.com/about" in result.urls_to_fetch

        # 没有被过滤的 URL
        assert len(result.filtered_urls) == 0

    def test_enforce_enabled_only_matching_prefixes_pass(self) -> None:
        """测试 enforce_path_prefixes=True 时，仅匹配前缀的内链通过"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",  # 匹配 "docs"
            "https://cursor.com/cn/docs/guide",  # 匹配 "cn/docs"
            "https://cursor.com/pricing",  # 不匹配
            "https://cursor.com/about",  # 不匹配
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["docs", "cn/docs"],
            enforce_path_prefixes=True,  # Phase B：执行路径检查
        )

        # 仅匹配前缀的内链通过
        assert len(result.urls_to_fetch) == 2
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://cursor.com/cn/docs/guide" in result.urls_to_fetch

        # 不匹配的内链被过滤
        assert len(result.filtered_urls) == 2
        for filtered in result.filtered_urls:
            assert filtered["reason"] == "internal_link_path_not_allowed"
            assert filtered["url"] in [
                "https://cursor.com/pricing",
                "https://cursor.com/about",
            ]

    def test_enforce_enabled_empty_prefixes_all_pass(self) -> None:
        """测试 enforce_path_prefixes=True 但 allowed_path_prefixes 为空时，所有内链通过"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/pricing",
            "https://cursor.com/any/path",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=[],  # 空列表表示全部允许
            enforce_path_prefixes=True,
        )

        # 空 allowed_path_prefixes 表示全部允许
        assert len(result.urls_to_fetch) == 3
        assert len(result.filtered_urls) == 0

    def test_enforce_enabled_with_external_links(self) -> None:
        """测试 enforce_path_prefixes=True 与外链处理的组合"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",  # 内链，匹配
            "https://cursor.com/pricing",  # 内链，不匹配
            "https://github.com/cursor/repo",  # 外链
            "https://external.com/guide",  # 外链
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["docs", "changelog"],
            enforce_path_prefixes=True,
        )

        # 仅匹配前缀的内链通过
        assert result.urls_to_fetch == ["https://cursor.com/docs/cli"]

        # 外链被记录
        assert "https://github.com/cursor/repo" in result.external_links_recorded
        assert "https://external.com/guide" in result.external_links_recorded

        # filtered_urls 应包含内链路径过滤和外链记录
        reasons = {f["reason"] for f in result.filtered_urls}
        assert "internal_link_path_not_allowed" in reasons
        assert "external_link_record_only" in reasons

    def test_combination_matrix_allowlist_and_enforce(self) -> None:
        """测试外链 allowlist 与内链 gate 的组合矩阵

        矩阵：
        - 内链匹配 path_prefix: 通过
        - 内链不匹配 path_prefix: 拒绝（enforce=True）
        - 外链在 allowlist 中: 通过（fetch_allowlist 模式）
        - 外链不在 allowlist 中: 拒绝
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",  # 内链，匹配 prefix
            "https://cursor.com/pricing",  # 内链，不匹配 prefix
            "https://github.com/cursor/repo",  # 外链，在 allowlist
            "https://gitlab.com/repo",  # 外链，不在 allowlist
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="fetch_allowlist",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["docs", "cn/docs"],
            enforce_path_prefixes=True,
            external_link_allowlist=["github.com"],
        )

        # 内链匹配 prefix + 外链在 allowlist 应通过
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" in result.urls_to_fetch

        # 内链不匹配 prefix 应拒绝
        assert "https://cursor.com/pricing" not in result.urls_to_fetch

        # 外链不在 allowlist 应拒绝
        assert "https://gitlab.com/repo" not in result.urls_to_fetch

        # 验证 filtered_urls 中的原因
        filtered_reasons = {f["url"]: f["reason"] for f in result.filtered_urls}
        assert filtered_reasons.get("https://cursor.com/pricing") == "internal_link_path_not_allowed"
        assert filtered_reasons.get("https://gitlab.com/repo") == "external_link_not_in_allowlist"

    def test_path_prefix_exact_match(self) -> None:
        """测试路径前缀精确匹配"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs",  # 精确匹配 "docs"
            "https://cursor.com/docs/cli",  # 子路径匹配 "docs"
            "https://cursor.com/documentation",  # 不匹配（不是 "docs/" 的子路径）
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["docs"],
            enforce_path_prefixes=True,
        )

        # "docs" 精确匹配和 "docs/cli" 子路径匹配都应通过
        assert "https://cursor.com/docs" in result.urls_to_fetch
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch

        # "documentation" 不是 "docs" 的子路径，应被拒绝
        assert "https://cursor.com/documentation" not in result.urls_to_fetch
        assert len(result.filtered_urls) == 1
        assert result.filtered_urls[0]["reason"] == "internal_link_path_not_allowed"

    def test_path_prefix_with_leading_slash(self) -> None:
        """测试路径前缀支持前导斜杠格式"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/cn/docs/guide",
        ]

        # 带前导斜杠的前缀也应正常工作
        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["/docs", "/cn/docs"],  # 带前导斜杠
            enforce_path_prefixes=True,
        )

        assert len(result.urls_to_fetch) == 2
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://cursor.com/cn/docs/guide" in result.urls_to_fetch


# ============================================================
# Test: _matches_path_prefixes 辅助函数测试
# ============================================================


class TestMatchesPathPrefixes:
    """测试 _matches_path_prefixes 辅助函数"""

    def test_empty_prefixes_returns_true(self) -> None:
        """测试空前缀列表返回 True（全部允许）"""
        from knowledge.doc_url_strategy import _matches_path_prefixes

        assert _matches_path_prefixes("https://example.com/any/path", []) is True

    def test_exact_path_match(self) -> None:
        """测试精确路径匹配"""
        from knowledge.doc_url_strategy import _matches_path_prefixes

        assert _matches_path_prefixes("https://example.com/docs", ["docs"]) is True

    def test_subpath_match(self) -> None:
        """测试子路径匹配"""
        from knowledge.doc_url_strategy import _matches_path_prefixes

        assert _matches_path_prefixes("https://example.com/docs/guide", ["docs"]) is True
        assert _matches_path_prefixes("https://example.com/docs/api/v1", ["docs"]) is True

    def test_no_match(self) -> None:
        """测试不匹配的情况"""
        from knowledge.doc_url_strategy import _matches_path_prefixes

        assert _matches_path_prefixes("https://example.com/pricing", ["docs"]) is False
        assert _matches_path_prefixes("https://example.com/documentation", ["docs"]) is False

    def test_nested_prefix_match(self) -> None:
        """测试嵌套前缀匹配"""
        from knowledge.doc_url_strategy import _matches_path_prefixes

        assert _matches_path_prefixes("https://example.com/cn/docs/guide", ["cn/docs"]) is True
        assert _matches_path_prefixes("https://example.com/cn/other", ["cn/docs"]) is False

    def test_multiple_prefixes(self) -> None:
        """测试多个前缀匹配"""
        from knowledge.doc_url_strategy import _matches_path_prefixes

        prefixes = ["docs", "cn/docs", "changelog"]
        assert _matches_path_prefixes("https://example.com/docs", prefixes) is True
        assert _matches_path_prefixes("https://example.com/cn/docs/guide", prefixes) is True
        assert _matches_path_prefixes("https://example.com/changelog/2025", prefixes) is True
        assert _matches_path_prefixes("https://example.com/pricing", prefixes) is False


# ============================================================
# Test: 旧字段兼容性测试（包含 warning 断言）
# ============================================================


# 用于捕获 loguru 日志的辅助 fixture
@pytest.fixture
def loguru_caplog():
    """Loguru 日志捕获 fixture

    loguru 不使用标准 logging 模块，需要自定义捕获器。
    使用方法:
        def test_example(loguru_caplog):
            with loguru_caplog() as captured:
                # 调用会产生日志的函数
                some_function()
            assert "expected message" in captured.text
    """
    import io
    from contextlib import contextmanager

    from loguru import logger

    @contextmanager
    def _capture():
        output = io.StringIO()
        handler_id = logger.add(output, format="{message}", level="DEBUG")
        try:
            yield output
        finally:
            logger.remove(handler_id)
            output.seek(0)

    return _capture


class TestDeprecatedFieldCompatibility:
    """测试旧字段的兼容性及 deprecation warning

    测试场景：
    - fetch_policy.allowed_url_prefixes（旧）→ allowed_path_prefixes（新）
    - url_strategy.allowed_url_prefixes 使用路径前缀格式（旧）→ 完整 URL 格式（新）

    注意：使用自定义 loguru_caplog fixture 捕获 loguru 日志
    """

    def test_validate_fetch_policy_prefixes_deprecation_warning(self, loguru_caplog) -> None:
        """测试 validate_fetch_policy_prefixes 触发 deprecation warning"""
        from knowledge.doc_url_strategy import (
            DEPRECATED_MSG_FUNC_PREFIX,
            reset_deprecated_func_warnings,
            validate_fetch_policy_prefixes,
        )

        # 清除之前可能的警告状态（使用统一的 reset 函数）
        reset_deprecated_func_warnings()

        with loguru_caplog() as output:
            result = validate_fetch_policy_prefixes(["docs", "cn/docs"])

        # 验证返回值不变
        assert result == ["docs", "cn/docs"]

        # 验证触发 deprecation warning（使用统一的文案片段常量）
        log_text = output.getvalue()
        assert DEPRECATED_MSG_FUNC_PREFIX in log_text, f"应包含 '{DEPRECATED_MSG_FUNC_PREFIX}'，实际: {log_text}"
        assert "validate_fetch_policy_path_prefixes" in log_text

    def test_validate_url_strategy_prefixes_path_format_warning(self, loguru_caplog) -> None:
        """测试 url_strategy 使用旧版路径前缀格式时触发 warning"""
        from knowledge.doc_url_strategy import validate_url_strategy_prefixes

        with loguru_caplog() as output:
            # 使用路径前缀格式（旧版）
            result = validate_url_strategy_prefixes(["docs", "cn/docs"])

        # 验证返回值不变（向后兼容）
        assert result == ["docs", "cn/docs"]

        # 验证触发格式警告
        log_text = output.getvalue()
        assert "旧版" in log_text
        assert "完整 URL 前缀" in log_text

    def test_validate_url_strategy_prefixes_full_url_no_warning(self, loguru_caplog) -> None:
        """测试 url_strategy 使用完整 URL 前缀格式时不触发 warning"""
        from knowledge.doc_url_strategy import validate_url_strategy_prefixes

        with loguru_caplog() as output:
            # 使用完整 URL 前缀格式（新版）
            result = validate_url_strategy_prefixes(
                [
                    "https://cursor.com/docs",
                    "https://cursor.com/cn/docs",
                ]
            )

        # 验证返回值
        assert "https://cursor.com/docs" in result

        # 验证不触发格式警告（关于路径前缀的警告）
        log_text = output.getvalue()
        assert "旧版" not in log_text

    def test_validate_fetch_policy_path_prefixes_full_url_warning(self, loguru_caplog) -> None:
        """测试 fetch_policy 使用完整 URL 格式时触发 warning"""
        from knowledge.doc_url_strategy import validate_fetch_policy_path_prefixes

        with loguru_caplog() as output:
            # 使用完整 URL 格式（错误用法）
            result = validate_fetch_policy_path_prefixes(
                [
                    "https://cursor.com/docs",  # 应该是 "docs"
                ]
            )

        # 验证返回值不变
        assert result == ["https://cursor.com/docs"]

        # 验证触发格式警告
        log_text = output.getvalue()
        assert "路径前缀格式" in log_text

    def test_validate_external_link_mode_invalid_value_warning(self, loguru_caplog) -> None:
        """测试无效的 external_link_mode 值触发 warning 并回退默认值"""
        from knowledge.doc_url_strategy import validate_external_link_mode

        with loguru_caplog() as output:
            result = validate_external_link_mode("invalid_mode")

        # 验证回退到默认值
        assert result == "record_only"

        # 验证触发警告
        log_text = output.getvalue()
        assert "值无效" in log_text
        assert "record_only" in log_text

    def test_is_full_url_prefix_helper(self) -> None:
        """测试 is_full_url_prefix 辅助函数"""
        from knowledge.doc_url_strategy import is_full_url_prefix

        # 完整 URL 前缀
        assert is_full_url_prefix("https://cursor.com/docs")
        assert is_full_url_prefix("http://example.com/api")

        # 路径前缀
        assert not is_full_url_prefix("docs")
        assert not is_full_url_prefix("cn/docs")
        assert not is_full_url_prefix("/docs")

    def test_is_path_prefix_helper(self) -> None:
        """测试 is_path_prefix 辅助函数"""
        from knowledge.doc_url_strategy import is_path_prefix

        # 路径前缀
        assert is_path_prefix("docs")
        assert is_path_prefix("cn/docs")
        assert is_path_prefix("/docs")

        # 完整 URL 前缀
        assert not is_path_prefix("https://cursor.com/docs")
        assert not is_path_prefix("http://example.com/api")


# ============================================================
# Test: ExternalLinkAllowlist 结构化对象测试
# ============================================================


class TestExternalLinkAllowlistStructure:
    """测试 ExternalLinkAllowlist 结构化对象的行为"""

    def test_validate_external_link_allowlist_parsing(self) -> None:
        """测试 allowlist 解析为结构化对象"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "github.com",
                "https://docs.python.org/3",
                "api.openai.com",
                "",  # 无效项
            ]
        )

        # 验证解析结果
        assert "github.com" in result.domains
        assert "api.openai.com" in result.domains
        assert any("docs.python.org" in p for p in result.prefixes)
        assert "" in result.invalid_items

    def test_external_link_allowlist_matches_domain(self) -> None:
        """测试 ExternalLinkAllowlist.matches 域名匹配"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        allowlist = validate_external_link_allowlist(["github.com"])

        assert allowlist.matches("https://github.com/cursor/repo")
        assert allowlist.matches("https://api.github.com/v1")  # 子域名
        assert not allowlist.matches("https://gitlab.com/repo")

    def test_external_link_allowlist_matches_prefix(self) -> None:
        """测试 ExternalLinkAllowlist.matches URL 前缀匹配"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        allowlist = validate_external_link_allowlist(
            [
                "https://github.com/cursor",
            ]
        )

        assert allowlist.matches("https://github.com/cursor/repo")
        assert allowlist.matches("https://github.com/cursor/docs")
        assert not allowlist.matches("https://github.com/other/repo")

    def test_external_link_allowlist_is_empty(self) -> None:
        """测试 ExternalLinkAllowlist.is_empty"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        # 空 allowlist
        empty = validate_external_link_allowlist([])
        assert empty.is_empty()

        # 仅有无效项
        invalid_only = validate_external_link_allowlist(["", "   "])
        assert invalid_only.is_empty()

        # 有效项
        valid = validate_external_link_allowlist(["github.com"])
        assert not valid.is_empty()

    def test_external_link_allowlist_invalid_items_warning(self, loguru_caplog) -> None:
        """测试无效项触发 warning"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        with loguru_caplog() as output:
            result = validate_external_link_allowlist(["", "   ", "valid.com"])

        # 验证有效项被解析
        assert "valid.com" in result.domains

        # 验证无效项触发警告
        log_text = output.getvalue()
        assert "无效项" in log_text


# ============================================================
# Test: 综合集成测试（断言结果集合、日志字段、warning 文案）
# ============================================================


class TestIntegratedUrlPolicyBehavior:
    """综合集成测试：同时验证结果集合、结构化日志字段、关键 warning 文案

    注意：使用自定义 loguru_caplog fixture 捕获 loguru 日志
    """

    def test_full_workflow_default_config(self) -> None:
        """测试完整工作流：默认配置场景

        同时验证：
        - 结果集合正确性
        - 日志字段（filtered_urls 结构）
        - 无不必要的 warning
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        # 模拟完整工作流
        config = DocURLStrategyConfig()

        # url_strategy 阶段
        selected = select_urls_to_fetch(
            changelog_links=["https://cursor.com/changelog/2026"],
            related_doc_urls=["https://cursor.com/docs/guide"],
            llms_txt_content="https://cursor.com/docs/api",
            core_docs=[],
            keywords=[],
            config=config,
        )

        # fetch_policy 阶段
        result = apply_fetch_policy(
            urls=selected,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
        )

        # 验证结果集合
        assert len(result.urls_to_fetch) >= 1
        assert all("cursor.com" in u for u in result.urls_to_fetch)

        # 验证无外链（所有 URL 都是 cursor.com）
        assert len(result.external_links_recorded) == 0
        assert len(result.filtered_urls) == 0

    def test_full_workflow_with_external_links(self, loguru_caplog) -> None:
        """测试完整工作流：包含外链场景

        同时验证：
        - 外链被正确分类和过滤
        - filtered_urls 结构包含正确的 reason
        - 日志包含处理摘要
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
            "https://external.com/guide",
        ]

        with loguru_caplog() as output:
            result = apply_fetch_policy(
                urls=urls,
                fetch_policy_mode="record_only",
                primary_domains=["cursor.com"],
            )

        # 验证结果集合
        assert result.urls_to_fetch == ["https://cursor.com/docs/cli"]
        assert len(result.external_links_recorded) == 2
        assert len(result.filtered_urls) == 2

        # 验证 filtered_urls 结构
        for item in result.filtered_urls:
            assert "url" in item
            assert "reason" in item
            assert item["reason"] == "external_link_record_only"

        # 验证日志包含处理摘要
        log_text = output.getvalue()
        assert "apply_fetch_policy" in log_text

    def test_deprecated_field_with_behavior_consistency(self, loguru_caplog) -> None:
        """测试旧字段使用时的行为一致性

        验证：
        - 旧字段触发 deprecation warning
        - 行为与新字段一致
        """
        from knowledge.doc_url_strategy import (
            DEPRECATED_MSG_FUNC_PREFIX,
            reset_deprecated_func_warnings,
            validate_fetch_policy_path_prefixes,
            validate_fetch_policy_prefixes,
        )

        # 清除之前的警告状态（使用统一的 reset 函数）
        reset_deprecated_func_warnings()

        test_prefixes = ["docs", "cn/docs", "changelog"]

        with loguru_caplog() as output:
            # 新函数
            new_result = validate_fetch_policy_path_prefixes(test_prefixes.copy())

            # 旧函数（会触发 deprecation warning）
            old_result = validate_fetch_policy_prefixes(test_prefixes.copy())

        # 验证行为一致
        assert new_result == old_result

        # 验证旧函数触发 deprecation warning（使用统一的文案片段常量）
        log_text = output.getvalue()
        assert DEPRECATED_MSG_FUNC_PREFIX in log_text, (
            f"废弃警告应包含 '{DEPRECATED_MSG_FUNC_PREFIX}'，实际: {log_text}"
        )

    def test_config_manager_parse_fetch_policy_compat(self, loguru_caplog) -> None:
        """测试 ConfigManager 解析 fetch_policy 的兼容性

        验证 allowed_url_prefixes（旧）→ allowed_path_prefixes（新）的兼容处理
        """
        from core.config import ConfigManager

        # 模拟旧版配置使用 allowed_url_prefixes
        old_config_raw = {
            "allowed_url_prefixes": ["docs", "cn/docs"],  # 旧字段名
            "external_link_mode": "record_only",
        }

        manager = ConfigManager()

        with loguru_caplog() as output:
            result = manager._parse_fetch_policy(old_config_raw)

        # 验证解析结果正确
        assert "docs" in result.allowed_path_prefixes
        assert "cn/docs" in result.allowed_path_prefixes
        assert result.external_link_mode == "record_only"

        # 验证触发废弃警告
        log_text = output.getvalue()
        assert "废弃" in log_text

    def test_config_manager_parse_fetch_policy_new_field(self, loguru_caplog) -> None:
        """测试 ConfigManager 解析 fetch_policy 使用新字段名

        验证 allowed_path_prefixes 优先于 allowed_url_prefixes
        """
        from core.config import ConfigManager

        # 新版配置使用 allowed_path_prefixes
        new_config_raw = {
            "allowed_path_prefixes": ["docs", "cn/docs"],  # 新字段名
            "external_link_mode": "skip_all",
        }

        manager = ConfigManager()

        with loguru_caplog() as output:
            result = manager._parse_fetch_policy(new_config_raw)

        # 验证解析结果正确
        assert result.allowed_path_prefixes == ["docs", "cn/docs"]
        assert result.external_link_mode == "skip_all"

        # 验证不触发废弃警告
        log_text = output.getvalue()
        assert "废弃" not in log_text

    def test_config_manager_both_fields_priority(self, loguru_caplog) -> None:
        """测试同时配置新旧字段时新字段优先"""
        from core.config import ConfigManager

        # 同时配置新旧字段
        both_config_raw = {
            "allowed_path_prefixes": ["new_docs"],  # 新字段（应优先）
            "allowed_url_prefixes": ["old_docs"],  # 旧字段（应忽略）
            "external_link_mode": "record_only",
        }

        manager = ConfigManager()

        with loguru_caplog() as output:
            result = manager._parse_fetch_policy(both_config_raw)

        # 验证新字段优先
        assert result.allowed_path_prefixes == ["new_docs"]
        assert "old_docs" not in result.allowed_path_prefixes

        # 验证触发警告（同时配置两个字段）
        log_text = output.getvalue()
        assert "忽略" in log_text


# ============================================================
# Test: validate_external_link_allowlist
# ============================================================


class TestValidateExternalLinkAllowlist:
    """测试 validate_external_link_allowlist 函数

    测试覆盖：
    1. 域名项解析
    2. URL 前缀项解析
    3. 混合项解析
    4. 非法项处理
    5. ExternalLinkAllowlist 的 matches 方法
    """

    def test_parse_domain_items(self) -> None:
        """测试域名项解析"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "github.com",
                "docs.python.org",
                "api.openai.com",
            ]
        )

        assert "github.com" in result.domains
        assert "docs.python.org" in result.domains
        assert "api.openai.com" in result.domains
        assert len(result.domains) == 3
        assert len(result.prefixes) == 0
        assert len(result.invalid_items) == 0

    def test_parse_url_prefix_items(self) -> None:
        """测试 URL 前缀项解析"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "https://github.com/cursor",
                "https://docs.python.org/3/library",
                "http://example.com/api",
            ]
        )

        assert len(result.domains) == 0
        assert len(result.prefixes) == 3
        # 规范化后的 URL 前缀
        assert any("github.com/cursor" in p for p in result.prefixes)
        assert any("docs.python.org/3/library" in p for p in result.prefixes)
        assert any("example.com/api" in p for p in result.prefixes)
        assert len(result.invalid_items) == 0

    def test_parse_mixed_items(self) -> None:
        """测试混合项（域名 + URL 前缀）解析"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "github.com",
                "https://docs.python.org/3",
                "api.openai.com",
                "https://example.com/api/v2",
            ]
        )

        assert "github.com" in result.domains
        assert "api.openai.com" in result.domains
        assert len(result.domains) == 2
        assert len(result.prefixes) == 2
        assert len(result.invalid_items) == 0

    def test_parse_domain_with_path(self) -> None:
        """测试带路径的域名格式（如 github.com/org）"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "github.com/cursor",
                "github.com/openai/whisper",
            ]
        )

        # 带路径的域名应转为 URL 前缀
        assert len(result.domains) == 0
        assert len(result.prefixes) == 2
        assert any("github.com/cursor" in p for p in result.prefixes)
        assert any("github.com/openai/whisper" in p for p in result.prefixes)

    def test_parse_invalid_items(self) -> None:
        """测试非法项处理"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist(
            [
                "github.com",
                "",  # 空字符串
                "   ",  # 仅空白
                "https://valid.com/path",
            ]
        )

        assert "github.com" in result.domains
        assert len(result.prefixes) == 1
        assert len(result.invalid_items) == 2
        assert "" in result.invalid_items
        assert "   " in result.invalid_items

    def test_empty_allowlist(self) -> None:
        """测试空白名单"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        result = validate_external_link_allowlist([])

        assert result.is_empty()
        assert len(result.domains) == 0
        assert len(result.prefixes) == 0
        assert len(result.invalid_items) == 0

    def test_matches_domain(self) -> None:
        """测试 ExternalLinkAllowlist.matches() 域名匹配"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        allowlist = validate_external_link_allowlist(
            [
                "github.com",
                "python.org",
            ]
        )

        # 精确域名匹配
        assert allowlist.matches("https://github.com/repo")
        assert allowlist.matches("https://python.org/docs")

        # 子域名匹配
        assert allowlist.matches("https://api.github.com/v1")
        assert allowlist.matches("https://docs.python.org/3")

        # 不匹配
        assert not allowlist.matches("https://gitlab.com/repo")
        assert not allowlist.matches("https://example.com/docs")

    def test_matches_url_prefix(self) -> None:
        """测试 ExternalLinkAllowlist.matches() URL 前缀匹配"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        allowlist = validate_external_link_allowlist(
            [
                "https://github.com/cursor",
                "https://docs.python.org/3",
            ]
        )

        # 精确前缀匹配
        assert allowlist.matches("https://github.com/cursor/repo")
        assert allowlist.matches("https://docs.python.org/3/library")

        # 不匹配（不同路径前缀）
        assert not allowlist.matches("https://github.com/other/repo")
        assert not allowlist.matches("https://docs.python.org/2/library")

    def test_matches_priority_prefix_over_domain(self) -> None:
        """测试匹配优先级：URL 前缀优先于域名"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        # 同时配置域名和 URL 前缀
        allowlist = validate_external_link_allowlist(
            [
                "github.com",  # 域名
                "https://github.com/cursor",  # URL 前缀
            ]
        )

        # URL 前缀匹配
        assert allowlist.matches("https://github.com/cursor/repo")

        # 域名回退匹配
        assert allowlist.matches("https://github.com/other/repo")

    def test_matches_empty_url(self) -> None:
        """测试空 URL"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        allowlist = validate_external_link_allowlist(["github.com"])

        assert not allowlist.matches("")
        assert not allowlist.matches(None)  # type: ignore

    def test_apply_fetch_policy_with_structured_allowlist(self) -> None:
        """测试 apply_fetch_policy 使用结构化 ExternalLinkAllowlist"""
        from knowledge.doc_url_strategy import (
            apply_fetch_policy,
            validate_external_link_allowlist,
        )

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
            "https://github.com/other/repo",
            "https://external.com/guide",
        ]

        # 使用结构化 allowlist
        allowlist = validate_external_link_allowlist(
            [
                "https://github.com/cursor",  # URL 前缀
            ]
        )

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="fetch_allowlist",
            primary_domains=["cursor.com"],
            external_link_allowlist=allowlist,
        )

        # 内链 + 匹配 allowlist 的外链
        assert "https://cursor.com/docs/cli" in result.urls_to_fetch
        assert "https://github.com/cursor/repo" in result.urls_to_fetch

        # 不匹配 allowlist 的外链
        assert "https://github.com/other/repo" not in result.urls_to_fetch
        assert "https://external.com/guide" not in result.urls_to_fetch

    def test_case_insensitive_domain_matching(self) -> None:
        """测试域名匹配大小写不敏感"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        allowlist = validate_external_link_allowlist(["GITHUB.COM"])

        # 域名存储为小写
        assert "github.com" in allowlist.domains

        # 匹配不区分大小写
        assert allowlist.matches("https://GitHub.com/repo")
        assert allowlist.matches("https://GITHUB.COM/repo")

    def test_url_prefix_normalization(self) -> None:
        """测试 URL 前缀规范化"""
        from knowledge.doc_url_strategy import validate_external_link_allowlist

        # 带末尾斜杠的 URL 会被规范化
        allowlist = validate_external_link_allowlist(
            [
                "https://github.com/cursor/",
                "https://EXAMPLE.COM/Docs/",
            ]
        )

        # 规范化后应移除末尾斜杠，域名小写
        assert any("github.com/cursor" in p for p in allowlist.prefixes)
        assert any("example.com/Docs" in p for p in allowlist.prefixes)


# ============================================================
# Test: apply_fetch_policy 集成测试
# ============================================================
# 验证 apply_fetch_policy 的完整过滤流程，确保：
# 1. 返回结果正确覆盖后续抓取层输入
# 2. 日志结构包含过滤原因与外链记录
# ============================================================


class TestApplyFetchPolicyIntegration:
    """apply_fetch_policy 集成测试

    模拟 KnowledgeUpdater.update_from_analysis 的调用流程，验证：
    - 内链（允许/不允许路径）的过滤行为
    - 外链（允许/不允许域名）的过滤行为
    - 返回结果的 urls_to_fetch 可直接用于抓取层
    - filtered_urls 和 external_links_recorded 结构正确
    """

    def test_integration_mixed_urls_with_enforce_enabled(self) -> None:
        """测试启用 enforce_path_prefixes 时的混合 URL 过滤

        场景：
        - 内链允许路径: /docs/*, /cn/docs/*
        - 内链不允许路径: /pricing, /about
        - 外链允许域名: github.com/cursor
        - 外链不允许域名: external.com

        预期：
        - urls_to_fetch: 仅包含允许的内链 + 允许的外链
        - filtered_urls: 包含所有被过滤的 URL 及原因
        - external_links_recorded: 包含被记录但未抓取的外链
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        # 模拟 _build_urls_to_fetch 后的 URL 列表
        urls_from_build = [
            # 内链 - 允许的路径
            "https://cursor.com/docs/cli",
            "https://cursor.com/docs/mcp",
            "https://cursor.com/cn/docs/guide",
            # 内链 - 不允许的路径
            "https://cursor.com/pricing",
            "https://cursor.com/about",
            # 外链 - 允许的（github.com/cursor）
            "https://github.com/cursor/repo",
            # 外链 - 不允许的
            "https://github.com/other/project",
            "https://external.com/guide",
            "https://stackoverflow.com/questions/12345",
        ]

        # 调用 apply_fetch_policy（模拟 update_from_analysis 中的调用）
        result = apply_fetch_policy(
            urls=urls_from_build,
            fetch_policy_mode="fetch_allowlist",  # 允许白名单内的外链
            base_url="https://cursor.com/cn/changelog",
            primary_domains=["cursor.com"],
            allowed_domains=None,  # 不额外添加域名
            external_link_allowlist=["github.com/cursor"],  # 仅允许 cursor 组织
            allowed_path_prefixes=["docs", "cn/docs"],  # 仅允许 docs 路径
            enforce_path_prefixes=True,  # 启用内链路径检查
        )

        # 验证 1: urls_to_fetch 仅包含允许的 URL
        expected_fetch_urls = {
            "https://cursor.com/docs/cli",
            "https://cursor.com/docs/mcp",
            "https://cursor.com/cn/docs/guide",
            "https://github.com/cursor/repo",  # 匹配外链白名单
        }
        assert set(result.urls_to_fetch) == expected_fetch_urls, f"urls_to_fetch 不匹配: {result.urls_to_fetch}"

        # 验证 2: filtered_urls 包含正确的过滤原因
        filtered_reasons = {f["url"]: f["reason"] for f in result.filtered_urls}

        # 内链路径不匹配
        assert filtered_reasons.get("https://cursor.com/pricing") == "internal_link_path_not_allowed"
        assert filtered_reasons.get("https://cursor.com/about") == "internal_link_path_not_allowed"

        # 外链不在白名单
        assert filtered_reasons.get("https://github.com/other/project") == "external_link_not_in_allowlist"
        assert filtered_reasons.get("https://external.com/guide") == "external_link_not_in_allowlist"
        assert filtered_reasons.get("https://stackoverflow.com/questions/12345") == "external_link_not_in_allowlist"

        # 验证 3: external_links_recorded 包含被记录的外链
        expected_recorded = {
            "https://github.com/other/project",
            "https://external.com/guide",
            "https://stackoverflow.com/questions/12345",
        }
        assert set(result.external_links_recorded) == expected_recorded, (
            f"external_links_recorded 不匹配: {result.external_links_recorded}"
        )

    def test_integration_record_only_mode_preserves_all_internal_links(self) -> None:
        """测试 record_only 模式 + enforce_path_prefixes=False 时保留所有内链

        这是 Phase A 的默认行为，确保向后兼容。
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            # 内链 - 各种路径
            "https://cursor.com/docs/cli",
            "https://cursor.com/pricing",
            "https://cursor.com/about",
            "https://cursor.com/blog/post-1",
            # 外链
            "https://github.com/cursor/repo",
            "https://external.com/guide",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["docs", "cn/docs"],  # 配置了路径前缀
            enforce_path_prefixes=False,  # 但未启用强制检查
        )

        # 所有内链都应通过（不管路径是否匹配）
        internal_urls = [u for u in urls if "cursor.com" in u]
        for url in internal_urls:
            assert url in result.urls_to_fetch, f"内链 {url} 应在 urls_to_fetch 中"

        # 外链应被记录但不抓取
        assert "https://github.com/cursor/repo" not in result.urls_to_fetch
        assert "https://external.com/guide" not in result.urls_to_fetch
        assert "https://github.com/cursor/repo" in result.external_links_recorded
        assert "https://external.com/guide" in result.external_links_recorded

    def test_integration_skip_all_mode_no_external_links_recorded(self) -> None:
        """测试 skip_all 模式不记录外链"""
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://github.com/cursor/repo",
            "https://external.com/guide",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="skip_all",
            primary_domains=["cursor.com"],
        )

        # 仅内链通过
        assert result.urls_to_fetch == ["https://cursor.com/docs/cli"]

        # skip_all 模式下不记录外链
        assert len(result.external_links_recorded) == 0

        # 但 filtered_urls 应包含外链
        assert len(result.filtered_urls) == 2
        for f in result.filtered_urls:
            assert f["reason"] == "external_link_skip_all"

    def test_integration_result_can_be_used_for_fetch(self) -> None:
        """测试返回结果可直接用于抓取层（模拟 _fetch_related_docs 输入）

        验证 FetchPolicyResult.urls_to_fetch 的格式和内容：
        - 类型正确（list[str]）
        - URL 格式正确（完整 URL）
        - 无重复 URL
        """
        from knowledge.doc_url_strategy import FetchPolicyResult, apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/docs/cli",  # 重复 URL（测试去重不在此层）
            "https://cursor.com/cn/docs/guide",
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
        )

        # 验证返回类型
        assert isinstance(result, FetchPolicyResult)
        assert isinstance(result.urls_to_fetch, list)

        # 验证 URL 格式（完整 URL）
        for url in result.urls_to_fetch:
            assert url.startswith("https://"), f"URL 应为完整格式: {url}"

        # 注意：apply_fetch_policy 不负责去重，去重在 _build_urls_to_fetch 阶段
        # 这里保持输入顺序

    def test_integration_log_structure_for_observability(self) -> None:
        """测试日志结构可用于可观测性（模拟 UrlSelectionLog 填充）

        验证 filtered_urls 和 external_links_recorded 的结构可用于：
        - 日志记录
        - 调试诊断
        - 统计分析
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/pricing",  # 内链，路径不匹配
            "https://github.com/cursor/repo",  # 外链，白名单内
            "https://external.com/guide",  # 外链，白名单外
        ]

        result = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="fetch_allowlist",
            primary_domains=["cursor.com"],
            external_link_allowlist=["github.com/cursor"],
            allowed_path_prefixes=["docs"],
            enforce_path_prefixes=True,
        )

        # 验证 filtered_urls 结构可用于日志
        assert isinstance(result.filtered_urls, list)
        for item in result.filtered_urls:
            assert "url" in item, "filtered_urls 条目应包含 'url'"
            assert "reason" in item, "filtered_urls 条目应包含 'reason'"
            assert isinstance(item["url"], str)
            assert isinstance(item["reason"], str)
            # reason 应为预定义的值
            assert item["reason"] in {
                "internal_link_path_not_allowed",
                "external_link_record_only",
                "external_link_skip_all",
                "external_link_not_in_allowlist",
            }

        # 验证 external_links_recorded 结构
        assert isinstance(result.external_links_recorded, list)
        for url in result.external_links_recorded:
            assert isinstance(url, str)
            assert url.startswith("http")

        # 验证统计可用于分析
        internal_filtered = sum(1 for f in result.filtered_urls if f["reason"] == "internal_link_path_not_allowed")
        external_filtered = len(result.filtered_urls) - internal_filtered
        assert internal_filtered == 1  # /pricing
        assert external_filtered == 1  # external.com

    def test_integration_cli_override_enforce_path_prefixes(self) -> None:
        """测试 CLI 参数覆盖 enforce_path_prefixes

        模拟 --enforce-path-prefixes / --no-enforce-path-prefixes 的效果
        """
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/pricing",
        ]

        # 模拟 --enforce-path-prefixes
        result_enforced = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["docs"],
            enforce_path_prefixes=True,
        )
        assert "https://cursor.com/pricing" not in result_enforced.urls_to_fetch

        # 模拟 --no-enforce-path-prefixes
        result_not_enforced = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["docs"],
            enforce_path_prefixes=False,
        )
        assert "https://cursor.com/pricing" in result_not_enforced.urls_to_fetch

    def test_integration_allowed_path_prefixes_from_config(self) -> None:
        """测试 allowed_path_prefixes 配置链路（CLI > config.yaml > DEFAULT）

        验证默认值与配置覆盖的行为一致性
        """
        from core.config import DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES
        from knowledge.doc_url_strategy import apply_fetch_policy

        urls = [
            "https://cursor.com/docs/cli",
            "https://cursor.com/changelog/2025",
            "https://cursor.com/pricing",
        ]

        # 使用默认路径前缀
        result_default = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=DEFAULT_FETCH_POLICY_ALLOWED_PATH_PREFIXES,
            enforce_path_prefixes=True,
        )

        # 默认前缀包含 docs 和 changelog
        assert "https://cursor.com/docs/cli" in result_default.urls_to_fetch
        assert "https://cursor.com/changelog/2025" in result_default.urls_to_fetch
        assert "https://cursor.com/pricing" not in result_default.urls_to_fetch

        # CLI 覆盖：仅允许 docs
        result_cli_override = apply_fetch_policy(
            urls=urls,
            fetch_policy_mode="record_only",
            primary_domains=["cursor.com"],
            allowed_path_prefixes=["docs"],  # CLI 覆盖
            enforce_path_prefixes=True,
        )

        assert "https://cursor.com/docs/cli" in result_cli_override.urls_to_fetch
        assert "https://cursor.com/changelog/2025" not in result_cli_override.urls_to_fetch
        assert "https://cursor.com/pricing" not in result_cli_override.urls_to_fetch
