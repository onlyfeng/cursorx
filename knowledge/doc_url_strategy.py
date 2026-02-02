"""文档 URL 策略模块

提供 URL 选择、过滤、去重和优先级排序的策略逻辑。
用于确定从多个来源（changelog、llms.txt、core_docs 等）选择哪些 URL 进行获取。

特性：
- 确定性输出（稳定顺序）
- 清晰的优先级规则
- 可配置的截断策略

================================================================================
                              模块契约说明
================================================================================

本模块定义了文档 URL 处理的核心契约，供实现和测试参考。

================================================================================
【术语表】
================================================================================

本模块及相关配置使用以下统一术语：

┌───────────────────────────┬─────────────────────────────────────────────────┐
│ 术语                       │ 定义                                            │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ URL 规范化 (normalize)     │ 将 URL 转换为标准格式：移除锚点、末尾斜杠、      │
│                           │ scheme/host 小写化、合并重复斜杠、解析相对路径    │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ URL 过滤 (filter)         │ 根据白名单/黑名单规则筛选允许的 URL               │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ URL 去重 (deduplicate)    │ 移除重复的 URL（基于规范化后的结果）              │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ URL 选择 (select)         │ 按优先级和来源权重对 URL 排序并截取               │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ 路径前缀 (path prefix)    │ URL 路径部分的前缀，如 "docs", "cn/docs"         │
│                           │ 用于 fetch_policy.allowed_path_prefixes         │
│                           │ [DEPRECATED] allowed_url_prefixes 为废弃别名    │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ 完整 URL 前缀             │ 包含 scheme 和 host 的完整前缀                   │
│ (full URL prefix)         │ **必须包含 scheme/host**                        │
│                           │ 如 "https://cursor.com/docs"                    │
│                           │ 用于 url_strategy.allowed_url_prefixes          │
│                           │ 注意：旧版路径前缀格式（如 "docs"）已废弃       │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ 域名白名单                │ 允许的域名列表，支持子域名匹配                    │
│ (allowed_domains)         │ 如 ["cursor.com"] 可匹配 api.cursor.com         │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ fetch_policy              │ 在线抓取策略：控制哪些 URL 可以被实际抓取         │
│                           │ 配置位于 config.yaml knowledge_docs_update       │
│                           │ .docs_source.fetch_policy                       │
│                           │ **当前在 run_iterate 链路中预留但未实际执行**   │
│                           │ 见下方【fetch_policy 作用范围说明】             │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ url_strategy              │ URL 选择策略：控制 URL 的过滤、去重、优先级排序   │
│                           │ 配置位于 config.yaml knowledge_docs_update       │
│                           │ .url_strategy                                   │
├───────────────────────────┼─────────────────────────────────────────────────┤
│ allowed_doc_url_prefixes  │ 核心文档 URL 前缀过滤（完整 URL 前缀格式）        │
│ (docs_source 下)          │ **仅用于 load_core_docs 过滤核心文档列表**       │
│                           │ 与 fetch_policy/url_strategy 配置独立            │
│                           │ 配置位于 config.yaml docs_source.                │
│                           │ allowed_doc_url_prefixes                        │
└───────────────────────────┴─────────────────────────────────────────────────┘

【两个策略配置的区别】

fetch_policy（在线抓取策略）:
    - 作用域：决定哪些 URL **可以被抓取**（网络请求层面）
    - allowed_path_prefixes：**路径前缀**格式（如 "docs", "cn/docs"）
    - external_link_mode：控制外链处理（record_only/skip_all/fetch_allowlist）
    - CLI 参数：--allowed-path-prefixes, --allowed-domains, --external-link-mode
    - [DEPRECATED] allowed_url_prefixes：废弃别名，请使用 allowed_path_prefixes
    - [DEPRECATED] CLI --allowed-url-prefixes：废弃别名，请使用 --allowed-path-prefixes

url_strategy（URL 选择策略）:
    - 作用域：决定从候选 URL 中**选择哪些**进行处理（数据处理层面）
    - allowed_url_prefixes：**完整 URL 前缀**格式（必须含 scheme/host）
      正确格式：["https://cursor.com/docs", "https://cursor.com/cn/docs"]
      废弃格式：["docs", "cn/docs"]（旧版路径前缀，会触发警告）
    - 包含去重、规范化、优先级排序等功能
    - CLI 参数：--url-allowed-prefixes, --url-allowed-domains, --url-deduplicate

【fetch_policy 在 run_iterate 链路的作用范围说明】

当前版本的实际行为：
    - fetch_policy 配置**已解析并在 update_from_analysis 主流程中调用**
    - apply_fetch_policy() 的返回结果**已用于覆盖** urls_to_fetch
    - 外链过滤（external_link_mode）**已生效**：
      * record_only: 外链从 urls_to_fetch 移除但记录
      * skip_all: 外链既不抓取也不记录
      * fetch_allowlist: 仅白名单命中的外链允许抓取
    - 内链路径前缀检查（enforce_path_prefixes）：
      * 默认 False（保持向后兼容）
      * 设为 True 时，内链必须匹配 allowed_path_prefixes 才允许抓取
      * 可通过 --enforce-path-prefixes CLI 参数启用

Phase B（计划中）启用后：
    - enforce_path_prefixes 默认值将改为 True
    - 不匹配 allowed_path_prefixes 的内链将默认被拒绝
    - external_link_mode 默认行为不变（record_only）

使用建议：
    - 外链过滤：已生效，无需额外配置即可保持最小抓取面
    - 内链路径限制：需显式启用 --enforce-path-prefixes
    - 测试验证：可通过日志中的 fetch_policy_filtered 字段观察过滤结果

【docs_source.allowed_doc_url_prefixes 用途说明】

此配置项**仅用于**过滤核心文档列表（load_core_docs），不影响：
    - fetch_policy 的抓取策略（使用 allowed_path_prefixes）
    - url_strategy 的 URL 选择（使用 allowed_url_prefixes 或 allowed_domains）

格式要求：**完整 URL 前缀**（含 scheme/host）
    正确：["https://cursor.com/cn/docs", "https://cursor.com/docs"]
    错误：["cn/docs", "docs"]（不含域名）

================================================================================

【1. URL 规范化契约 - normalize_url()】

    输入:
        - url: 原始 URL 字符串（可为空、相对路径、含锚点等）
        - base_url: 可选的基础 URL（用于解析相对路径）

    输出:
        - 规范化后的 URL 字符串

    保证的行为:
        1. 空输入返回空字符串
        2. 相对路径使用 base_url 补全
        3. 移除 fragment（锚点，如 #section）
        4. 移除末尾斜杠（根路径 "/" 除外）
        5. scheme 和 host 归一化为小写，path/query 保留原样
        6. 处理 ../ 和 ./ 路径段
        7. 合并重复斜杠（// → /）
        8. 非 HTTP(S) URL 原样返回

    示例:
        "https://EXAMPLE.COM/Docs/../api/"  →  "https://example.com/api"
        "/docs/guide" + base="https://a.com"  →  "https://a.com/docs/guide"
        "https://a.com/page#section"  →  "https://a.com/page"

【2. URL 过滤契约 - is_allowed_doc_url()】

    优先级规则（从高到低）:
        1. allowed_url_prefixes 不为空时，使用前缀匹配
        2. 仅当 allowed_url_prefixes 为空时，回退到 allowed_domains 检查
        3. 两者都为空时，允许所有 URL

    保证的行为:
        - 空 URL 返回 False
        - 匹配 exclude_patterns 的 URL 返回 False
        - 域名匹配支持子域名（如 api.example.com 匹配 example.com）
        - 前缀匹配和域名匹配均大小写不敏感

【3. URL 选择契约 - select_urls_to_fetch()】

    来源优先级（默认权重）:
        1. changelog_links     (weight: 3.0) - 变更日志链接
        2. llms_txt            (weight: 2.5) - llms.txt 中的链接
        3. related_doc_urls    (weight: 2.0) - 相关文档链接
        4. core_docs           (weight: 1.5) - 核心文档

    排序规则:
        1. 按优先级降序（priority desc）
        2. 相同优先级按来源顺序（changelog > llms_txt > related_doc > core_doc）
        3. 相同来源按 URL 字母序（确保确定性）

    关键词匹配:
        - URL 中包含关键词时，priority += match_score * keyword_boost_weight
        - match_score = 匹配关键词数 / 总关键词数

    去重策略（deduplicate=True 时）:
        - 基于规范化后的 URL 去重
        - 保留优先级最高的版本

    兜底策略:
        - 当结果数 < fallback_core_docs_count 时，从 core_docs 补充
        - 补充的 URL 仍需通过过滤检查

    截断策略:
        - 最终结果数 ≤ max_urls
        - 截断在排序和兜底补充之后执行

【4. 确定性保证】

    对于相同的输入参数，以下函数保证输出顺序完全一致:
        - select_urls_to_fetch()
        - deduplicate_urls()
        - filter_urls_by_keywords()

    确定性通过以下方式保证:
        - 排序时使用 URL 字母序作为最终排序键
        - 遍历顺序遵循输入顺序

【5. 配置参数契约 - DocURLStrategyConfig】

    参数                     | 类型        | 默认值 | 说明
    -------------------------|-------------|--------|--------------------------------
    allowed_domains          | list[str]   | ["cursor.com"] | 允许的域名（仅当 prefixes 为空时生效）
    allowed_url_prefixes     | list[str]   | []     | 允许的 URL 前缀（优先级高于 domains）
    max_urls                 | int         | 20     | 最大返回 URL 数量
    fallback_core_docs_count | int         | 5      | 兜底补充的 core_docs 数量
    prefer_changelog         | bool        | True   | 是否优先处理 changelog
    deduplicate              | bool        | True   | 是否启用去重
    normalize                | bool        | True   | 是否规范化 URL
    keyword_boost_weight     | float       | 1.2    | 关键词匹配权重提升倍数（与 config.yaml 一致）
    exclude_patterns         | list[str]   | [...]  | 排除的 URL 正则模式（与 config.yaml 一致）
    priority_weights         | dict        | {...}  | 各来源的优先级权重

    注意：默认值来源于 core/config.py 的 DEFAULT_URL_STRATEGY_* 常量，
    确保与 config.yaml 保持同步。

================================================================================

================================================================================
                           URL 裁决规则表 (Adjudication Rules)
================================================================================

本节定义 URL 在各阶段的裁决逻辑，明确输入→输出的判定规则。

【输入】
┌─────────────────────────────────────────────────────────────────────────────┐
│ 参数                        │ 来源                     │ 说明               │
├─────────────────────────────┼──────────────────────────┼────────────────────┤
│ candidate_url               │ changelog/llms.txt/...   │ 候选 URL           │
│ base_url                    │ llms_txt_url 或          │ 用于规范化相对路径 │
│                             │ changelog_url 推导       │                    │
│ url_strategy_config         │ config.yaml              │ URL 选择策略配置   │
│                             │ .url_strategy            │                    │
│ docs_source.allowed_doc_    │ config.yaml              │ 核心文档 URL 前缀  │
│ url_prefixes                │ .docs_source             │ (完整 URL 格式)    │
│ fetch_policy                │ config.yaml              │ 在线抓取策略配置   │
│                             │ .docs_source.fetch_policy│                    │
└─────────────────────────────┴──────────────────────────┴────────────────────┘

【输出】
┌─────────────────────────────────────────────────────────────────────────────┐
│ 输出                        │ 类型    │ 说明                               │
├─────────────────────────────┼─────────┼────────────────────────────────────┤
│ in_urls_to_fetch            │ bool    │ 是否进入 urls_to_fetch 列表       │
│ is_external_link            │ bool    │ 是否记录为 external_links          │
│ allow_actual_fetch          │ bool    │ 是否允许实际发起 HTTP 抓取         │
└─────────────────────────────┴─────────┴────────────────────────────────────┘

【裁决规则矩阵】

阶段 1：URL 选择 (url_strategy)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 规则                                           │ in_urls_to_fetch │ 说明   │
├────────────────────────────────────────────────┼──────────────────┼────────┤
│ 通过 url_strategy.exclude_patterns 排除       │ False            │ 被过滤 │
│ 通过 url_strategy.allowed_url_prefixes 匹配   │ True             │ 允许   │
│ 通过 url_strategy.allowed_domains 匹配(回退)  │ True             │ 允许   │
│ 两者都为空                                     │ True             │ 全允许 │
└─────────────────────────────────────────────────────────────────────────────┘

阶段 2：外链分类 (external_link 判定)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 规则                                           │ is_external_link │ 说明   │
├────────────────────────────────────────────────┼──────────────────┼────────┤
│ URL 域名匹配 fetch_policy.allowed_domains     │ False            │ 内链   │
│ URL 域名 == base_url 域名（主域名）            │ False            │ 内链   │
│ 其他域名                                       │ True             │ 外链   │
└─────────────────────────────────────────────────────────────────────────────┘

阶段 3：抓取策略 (fetch_policy gate)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 条件                           │ external_link_mode   │ allow_actual_fetch  │
├────────────────────────────────┼──────────────────────┼─────────────────────┤
│ is_external_link = False       │ (任意)               │ 由 allowed_path_    │
│ (内链)                         │                      │ prefixes 决定       │
├────────────────────────────────┼──────────────────────┼─────────────────────┤
│ is_external_link = True        │ skip_all             │ False (仅记录)      │
│ (外链)                         │                      │                     │
├────────────────────────────────┼──────────────────────┼─────────────────────┤
│ is_external_link = True        │ record_only          │ False (仅记录)      │
│ (外链, 默认)                   │                      │                     │
├────────────────────────────────┼──────────────────────┼─────────────────────┤
│ is_external_link = True        │ fetch_allowlist      │ 匹配 external_link_ │
│ (外链)                         │                      │ allowlist 时 True   │
└─────────────────────────────────────────────────────────────────────────────┘

阶段 4：内链路径前缀检查 (allowed_path_prefixes gate)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 条件                                          │ allow_actual_fetch │ 说明  │
├───────────────────────────────────────────────┼────────────────────┼───────┤
│ URL 路径匹配 allowed_path_prefixes            │ True               │ 允许  │
│ allowed_path_prefixes 为空                    │ True (全部允许)    │ 宽松  │
│ URL 路径不匹配任何前缀                        │ False              │ 拒绝  │
└─────────────────────────────────────────────────────────────────────────────┘

【兼容性说明】

旧字段/CLI 参数仍可用，但会触发 deprecation warning：

┌─────────────────────────────────────────────────────────────────────────────┐
│ 旧字段/参数                        │ 新字段/参数                │ 警告级别 │
├────────────────────────────────────┼────────────────────────────┼──────────┤
│ fetch_policy.allowed_url_prefixes  │ fetch_policy.allowed_path_ │ WARNING  │
│ (config.yaml)                      │ prefixes                   │          │
├────────────────────────────────────┼────────────────────────────┼──────────┤
│ --allowed-url-prefixes (CLI)       │ --allowed-path-prefixes    │ WARNING  │
├────────────────────────────────────┼────────────────────────────┼──────────┤
│ url_strategy.allowed_url_prefixes  │ (无变更，语义为完整 URL)   │ WARNING  │
│ (路径前缀格式，旧版本)             │ 需改为完整 URL 格式        │ (格式)   │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
                         两阶段迁移计划 (Migration Phases)
================================================================================

【Phase A：修正示例/警告/测试，不改变抓取行为】
状态：当前阶段
目标：保持向后兼容，仅改进文档和警告

变更内容：
1. 文档更新
   - 更新 config.yaml 示例，使用新字段名
   - 更新 CLI --help 文本，标注废弃参数
   - 添加裁决规则表文档

2. 警告增强
   - 使用旧字段时输出 deprecation warning
   - 格式不匹配时输出 warning（但不拒绝）

3. 测试覆盖
   - 添加兼容性测试用例
   - 添加裁决规则表的单元测试

验收标准：
- 所有现有测试通过（无回归）
- 使用旧配置时输出 warning，但行为不变
- 新字段名可正常工作

【Phase B：启用 fetch_policy gate 的行为变更，默认最小抓取面】
状态：计划中
目标：启用完整的抓取策略控制，默认保守策略

变更内容：
1. 行为变更
   - fetch_policy.allowed_path_prefixes 默认生效（非空时严格过滤）
   - 不匹配路径前缀的 URL 将被 **实际拒绝**（非仅 warning）
   - external_link_mode 默认为 "record_only"（外链仅记录）

2. 默认值调整
   - allowed_path_prefixes 默认值: ["docs", "cn/docs", "changelog", "cn/changelog"]
   - external_link_mode 默认值: "record_only"

3. 移除兼容支持
   - fetch_policy.allowed_url_prefixes 字段移除
   - --allowed-url-prefixes CLI 参数移除
   - 旧格式不再自动兼容

切换条件：
- Phase A 稳定运行 >= 2 个版本
- 用户反馈无 breaking 问题
- 发布 v2.0 major 版本时启用

回滚策略：
- 提供 --legacy-fetch-behavior 临时开关
- 开关在 v2.1 版本后移除

================================================================================
                         run_iterate 调用点说明
================================================================================

本模块在 scripts/run_iterate.py 中的实际调用点及其 Phase 状态：

┌─────────────────────────────────────────────────────────────────────────────┐
│ 调用点                              │ 使用的函数              │ Phase 状态  │
├─────────────────────────────────────┼─────────────────────────┼─────────────┤
│ ChangelogAnalyzer._extract_links_   │ is_allowed_doc_url()    │ Phase A ✓   │
│ from_html()                         │ normalize_url()         │ 已生效      │
│                                     │                         │             │
│ 说明：使用 doc_allowlist.config 对 HTML 中提取的链接进行分类              │
│      - allowed: 通过 is_allowed_doc_url() 检查的链接                      │
│      - external: 不通过检查的外部链接                                      │
├─────────────────────────────────────┼─────────────────────────┼─────────────┤
│ KnowledgeUpdater._build_urls_to_    │ select_urls_to_fetch()  │ Phase A ✓   │
│ fetch()                             │ parse_llms_txt_urls()   │ 已生效      │
│                                     │ is_allowed_doc_url()    │             │
│                                     │ normalize_url()         │             │
│                                     │                         │             │
│ 说明：构建 URL 抓取列表时的核心策略调用                                    │
│      - 优先级排序: changelog > llms_txt > related_doc > core_doc          │
│      - 关键词匹配: _calculate_keyword_score()                              │
│      - 去重截断: deduplicate + max_urls 限制                               │
├─────────────────────────────────────┼─────────────────────────┼─────────────┤
│ KnowledgeUpdater.update_from_       │ apply_fetch_policy()    │ 已生效 ✓    │
│ analysis()                          │ is_external_link()      │             │
│                                     │ derive_primary_domains()│             │
│                                     │                         │             │
│ 说明：外链策略控制与内链路径检查                                           │
│      - apply_fetch_policy() 返回结果已用于覆盖 urls_to_fetch              │
│      - external_link_mode 外链过滤已生效                                   │
│      - enforce_path_prefixes 内链路径检查（默认禁用，可 CLI 启用）         │
├─────────────────────────────────────┼─────────────────────────┼─────────────┤
│ KnowledgeUpdater.__init__()         │ derive_primary_domains()│ Phase A ✓   │
│                                     │                         │ 已生效      │
│                                     │                         │             │
│ 说明：从 llms_txt_url 和 changelog_url 推导主域名                         │
│      用于 is_external_link() 判定内链/外链                                 │
└─────────────────────────────────────┴─────────────────────────┴─────────────┘

【已生效功能】
- URL 规范化 (normalize_url): 所有 URL 处理流程中使用
- URL 过滤 (is_allowed_doc_url): 链接分类、URL 选择
- URL 选择 (select_urls_to_fetch): 优先级排序、关键词匹配、去重截断
- 外链分类 (is_external_link): 外链识别和记录
- 主域名推导 (derive_primary_domains): 内链/外链判定基础
- apply_fetch_policy(): 在 update_from_analysis() 中调用，结果覆盖 urls_to_fetch
- external_link_mode: 外链过滤已生效（record_only/skip_all/fetch_allowlist）

【可选功能（需显式启用）】
- enforce_path_prefixes: 内链路径前缀检查，默认 False
  * 启用方式: --enforce-path-prefixes CLI 参数
  * 启用后内链必须匹配 allowed_path_prefixes 才允许抓取

【Phase B 计划变更】
- enforce_path_prefixes 默认值将改为 True
- 不匹配 allowed_path_prefixes 的内链将默认被拒绝

================================================================================
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

from loguru import logger

# 从 core/config.py 导入 URL 策略默认值常量，确保与 config.yaml 保持同步
# 注意：core/config.py 不依赖 knowledge 模块，不会引入循环导入
from core.config import (
    DEFAULT_FALLBACK_CORE_DOCS_COUNT,
    DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS,
    DEFAULT_URL_STRATEGY_DEDUPLICATE,
    DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS,
    DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT,
    DEFAULT_URL_STRATEGY_MAX_URLS,
    DEFAULT_URL_STRATEGY_NORMALIZE,
    DEFAULT_URL_STRATEGY_PREFER_CHANGELOG,
    DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS,
)


@dataclass
class DocURLStrategyConfig:
    """文档 URL 策略配置

    控制 URL 选择的各项参数。
    默认值从 core/config.py 的 DEFAULT_URL_STRATEGY_* 常量导入，
    确保与 config.yaml 保持同步，避免配置漂移。

    Attributes:
        allowed_domains: 允许的域名列表（仅在 allowed_url_prefixes 为空时生效）
        allowed_url_prefixes: 允许的 URL 前缀列表（优先级高于 allowed_domains）
        max_urls: 最大返回 URL 数量
        fallback_core_docs_count: 当其他来源不足时，从 core_docs 补充的数量
        prefer_changelog: 是否优先处理 changelog 链接
        deduplicate: 是否启用去重
        normalize: 是否规范化 URL
        keyword_boost_weight: 关键词匹配时的权重提升倍数
    """

    allowed_domains: list[str] = field(default_factory=lambda: DEFAULT_URL_STRATEGY_ALLOWED_DOMAINS.copy())
    allowed_url_prefixes: list[str] = field(default_factory=list)
    max_urls: int = DEFAULT_URL_STRATEGY_MAX_URLS
    fallback_core_docs_count: int = DEFAULT_FALLBACK_CORE_DOCS_COUNT
    prefer_changelog: bool = DEFAULT_URL_STRATEGY_PREFER_CHANGELOG
    deduplicate: bool = DEFAULT_URL_STRATEGY_DEDUPLICATE
    normalize: bool = DEFAULT_URL_STRATEGY_NORMALIZE
    keyword_boost_weight: float = DEFAULT_URL_STRATEGY_KEYWORD_BOOST_WEIGHT

    # 过滤规则
    # 使用 core/config.py 的 DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS
    # 注意：normalize 行为会移除 fragment（锚点），所以无需 r".*#.*$" 规则
    exclude_patterns: list[str] = field(default_factory=lambda: DEFAULT_URL_STRATEGY_EXCLUDE_PATTERNS.copy())

    # 优先级权重（用于排序）
    priority_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_URL_STRATEGY_PRIORITY_WEIGHTS.copy())


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """规范化 URL

    处理相对路径、移除锚点、统一协议和路径格式。
    规范化规则：
    1. 相对路径补全（使用 base_url）
    2. 移除 fragment（锚点）
    3. 移除末尾斜杠（除非是根路径）
    4. scheme/host 归一化为小写（保留 path/query 原样）
    5. 处理 ../ 和 ./ 路径段
    6. 合并重复斜杠（// → /）

    Args:
        url: 原始 URL
        base_url: 基础 URL（用于解析相对路径）

    Returns:
        规范化后的 URL

    Examples:
        >>> normalize_url("https://example.com/docs/../api/")
        'https://example.com/api'
        >>> normalize_url("/docs/guide", "https://example.com")
        'https://example.com/docs/guide'
        >>> normalize_url("https://example.com/page#section")
        'https://example.com/page'
        >>> normalize_url("https://EXAMPLE.COM/Docs")
        'https://example.com/Docs'
        >>> normalize_url("https://example.com//docs//page")
        'https://example.com/docs/page'
    """
    if not url:
        return ""

    url = url.strip()

    # strip 后再次检查是否为空
    if not url:
        return ""

    # 处理相对 URL
    if base_url and not url.startswith(("http://", "https://")):
        url = urljoin(base_url, url)

    # 解析 URL
    try:
        parsed = urlparse(url)
    except Exception:
        return url

    # 跳过非 HTTP(S) URL
    if parsed.scheme not in ("http", "https", ""):
        return url

    # 规范化路径
    path = parsed.path

    # 合并重复斜杠（// → /）- 在路径处理前完成
    while "//" in path:
        path = path.replace("//", "/")

    # 规范化路径中的 ../ 和 ./，同时移除空段（已通过重复斜杠处理）
    path_parts: list[str] = []
    for part in path.split("/"):
        if part == "..":
            if path_parts:
                path_parts.pop()
        elif part and part != ".":
            path_parts.append(part)
    path = "/" + "/".join(path_parts) if path_parts else "/"

    # 移除末尾斜杠（除非是根路径）- 最终处理
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # 重建 URL（移除 fragment/锚点）
    # scheme 和 netloc（host）归一化为小写，path/query 保留原样
    normalized = urlunparse(
        (
            (parsed.scheme or "https").lower(),  # scheme 小写，默认 https
            parsed.netloc.lower(),  # host 小写
            path,  # 保留 path 原样
            "",  # params 清空
            parsed.query,  # 保留 query string 原样
            "",  # 移除 fragment
        )
    )

    return normalized


def _normalize_prefixes(
    prefixes: list[str],
    base_url: Optional[str] = None,
) -> list[str]:
    """规范化 URL 前缀列表

    对每个前缀应用 normalize_url 进行规范化处理，确保前缀匹配时的一致性。
    支持相对路径前缀，通过 base_url 参数补全为绝对路径。

    Args:
        prefixes: 原始前缀列表（可包含相对路径）
        base_url: 基础 URL（用于解析相对路径前缀）

    Returns:
        规范化后的前缀列表（去除空值和重复项，保持顺序）

    Examples:
        >>> _normalize_prefixes(["https://EXAMPLE.COM/docs/", "https://example.com/api"])
        ['https://example.com/docs', 'https://example.com/api']
        >>> _normalize_prefixes(["/docs", "/api"], "https://example.com")
        ['https://example.com/docs', 'https://example.com/api']
    """
    seen: set[str] = set()
    result: list[str] = []

    for prefix in prefixes:
        if not prefix:
            continue

        # 规范化前缀（使用 normalize_url，支持 base_url 补全相对路径）
        normalized = normalize_url(prefix, base_url)
        if not normalized:
            continue

        # 去重并保持顺序
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def is_allowed_doc_url(
    url: str,
    config: DocURLStrategyConfig,
    base_url: Optional[str] = None,
) -> bool:
    """检查 URL 是否在允许的范围内

    会先对 URL 进行规范化处理（如果 config.normalize 为 True），
    确保判断基于一致的 URL 格式。

    优先级规则：
    1. exclude_patterns 最先检查，匹配的 URL 直接返回 False
    2. allowed_url_prefixes 优先（不为空时使用前缀匹配）
    3. 仅当 allowed_url_prefixes 为空时，回退到 allowed_domains 检查
    4. 两者都为空时，允许所有 URL

    前缀匹配规则（allowed_url_prefixes）：
    - 精确匹配：URL 与前缀完全相同时匹配（如 URL="https://a.com/docs" 匹配前缀 "https://a.com/docs"）
    - 子路径匹配：URL 以前缀开头时匹配（如 URL="https://a.com/docs/guide" 匹配前缀 "https://a.com/docs"）
    - 前缀会被规范化（移除末尾斜杠、合并重复斜杠、域名小写化）
    - 域名大小写不敏感，路径大小写敏感

    Args:
        url: 待检查的 URL
        config: 策略配置
        base_url: 基础 URL（用于规范化相对路径）

    Returns:
        True 如果 URL 被允许，False 否则

    Examples:
        >>> # 使用 allowed_url_prefixes（优先级更高）
        >>> config = DocURLStrategyConfig(allowed_url_prefixes=["https://docs.python.org/3/"])
        >>> is_allowed_doc_url("https://docs.python.org/3/library/", config)
        True
        >>> is_allowed_doc_url("https://docs.python.org/2/library/", config)
        False

        >>> # 精确匹配（URL 与前缀规范化后相同）
        >>> config = DocURLStrategyConfig(allowed_url_prefixes=["https://docs.python.org/3"])
        >>> is_allowed_doc_url("https://docs.python.org/3", config)
        True
        >>> is_allowed_doc_url("https://docs.python.org/3/", config)  # 规范化后相同
        True

        >>> # 使用 allowed_domains（仅当 allowed_url_prefixes 为空时生效）
        >>> config = DocURLStrategyConfig(allowed_domains=["docs.python.org"])
        >>> is_allowed_doc_url("https://docs.python.org/3/library/", config)
        True
        >>> is_allowed_doc_url("https://other.com/docs", config)
        False
        >>> is_allowed_doc_url("https://DOCS.PYTHON.ORG/Guide", config)
        True

        >>> # exclude_patterns 始终生效（即使匹配前缀）
        >>> config = DocURLStrategyConfig(
        ...     allowed_url_prefixes=["https://example.com/docs"],
        ...     exclude_patterns=[r".*\\.png$"]
        ... )
        >>> is_allowed_doc_url("https://example.com/docs/image.png", config)
        False
    """
    if not url:
        return False

    # 规范化 URL（如果配置启用）
    check_url = url
    if config.normalize:
        check_url = normalize_url(url, base_url)

    # 检查是否匹配排除模式（对规范化后的 URL 进行检查）
    for pattern in config.exclude_patterns:
        try:
            if re.match(pattern, check_url, re.IGNORECASE):
                logger.debug(f"URL 被排除模式过滤: {check_url} (pattern: {pattern})")
                return False
        except re.error:
            continue

    # 优先级规则：
    # 1. allowed_url_prefixes 优先（不为空时使用前缀匹配）
    # 2. 仅当 allowed_url_prefixes 为空时，回退到 allowed_domains 检查
    # 3. 两者都为空时，允许所有 URL

    # 检查 URL 前缀（优先级最高）
    if config.allowed_url_prefixes:
        # 规范化前缀列表（支持相对路径前缀 + base_url 补全）
        normalized_prefixes = _normalize_prefixes(config.allowed_url_prefixes, base_url)

        # 检查 URL 是否以任一前缀开头
        for prefix in normalized_prefixes:
            if check_url.startswith(prefix):
                return True

        logger.debug(f"URL 不匹配任何允许的前缀: {check_url}")
        return False

    # 如果没有指定前缀，回退到域名检查
    if not config.allowed_domains:
        # 两者都为空，允许所有
        return True

    # 解析规范化后的 URL 获取域名（已经是小写）
    try:
        parsed = urlparse(check_url)
        domain = parsed.netloc.lower()
    except Exception:
        return False

    # 检查域名是否在允许列表中（支持子域名匹配）
    for allowed in config.allowed_domains:
        allowed_lower = allowed.lower()
        # 精确匹配或子域名匹配
        if domain == allowed_lower or domain.endswith("." + allowed_lower):
            return True

    logger.debug(f"URL 域名不在允许列表中: {check_url} (domain: {domain})")
    return False


def deduplicate_urls(
    urls: list[str],
    normalize_before_dedup: bool = True,
    base_url: Optional[str] = None,
) -> list[str]:
    """URL 去重

    保持原始顺序，移除重复的 URL。

    Args:
        urls: URL 列表
        normalize_before_dedup: 去重前是否规范化 URL
        base_url: 基础 URL（用于规范化相对路径）

    Returns:
        去重后的 URL 列表（保持原始顺序）

    Examples:
        >>> deduplicate_urls(["https://a.com/", "https://a.com", "https://b.com"])
        ['https://a.com', 'https://b.com']
    """
    seen: set[str] = set()
    result: list[str] = []

    for url in urls:
        if not url:
            continue

        # 用于去重的 key
        key = normalize_url(url, base_url) if normalize_before_dedup else url

        if key not in seen:
            seen.add(key)
            # 返回规范化后的 URL（如果启用规范化）
            result.append(key if normalize_before_dedup else url)

    return result


def parse_llms_txt_urls(
    content: str,
    base_url: Optional[str] = None,
) -> list[str]:
    """解析 llms.txt 文件中的 URL

    llms.txt 是一种约定格式，用于列出 LLM 友好的文档链接。
    格式通常为每行一个 URL，或 Markdown 链接格式。

    Args:
        content: llms.txt 文件内容
        base_url: 基础 URL（用于解析相对路径）

    Returns:
        解析出的 URL 列表（保持原始顺序）

    Examples:
        >>> parse_llms_txt_urls("# Docs\\nhttps://example.com/guide\\n[API](./api)")
        ['https://example.com/guide', 'https://example.com/api']
    """
    if not content:
        return []

    urls: list[str] = []

    # 正则模式
    # 1. Markdown 链接: [text](url)
    md_link_pattern = r"\[([^\]]*)\]\(([^)]+)\)"
    # 2. 纯 URL（http/https 开头）
    plain_url_pattern = r'https?://[^\s<>"\')\]]+(?=[)\]\s<>"\']|$)'

    # 提取 Markdown 链接
    for match in re.finditer(md_link_pattern, content):
        url = match.group(2).strip()
        if url and not url.startswith("#"):  # 跳过锚点链接
            urls.append(url)

    # 提取纯 URL（避免重复提取 Markdown 链接中的 URL）
    # 先移除已找到的 Markdown 链接部分
    content_without_md = re.sub(md_link_pattern, " ", content)
    for match in re.finditer(plain_url_pattern, content_without_md):
        url = match.group(0).strip()
        if url:
            urls.append(url)

    # 规范化相对路径
    if base_url:
        urls = [normalize_url(url, base_url) for url in urls]

    return urls


@dataclass
class URLWithPriority:
    """带优先级的 URL（内部使用）"""

    url: str
    priority: float
    source: str  # 来源标识
    keyword_matched: bool = False


def _calculate_keyword_score(
    url: str,
    keywords: list[str],
) -> float:
    """计算 URL 与关键词的匹配得分

    Args:
        url: URL
        keywords: 关键词列表

    Returns:
        匹配得分（0.0 - 1.0）
    """
    if not keywords:
        return 0.0

    url_lower = url.lower()
    matched = sum(1 for kw in keywords if kw.lower() in url_lower)
    return matched / len(keywords) if keywords else 0.0


def select_urls_to_fetch(
    changelog_links: list[str],
    related_doc_urls: list[str],
    llms_txt_content: Optional[str],
    core_docs: list[str],
    keywords: list[str],
    config: Optional[DocURLStrategyConfig] = None,
    base_url: Optional[str] = None,
) -> list[str]:
    """选择要获取的 URL

    根据优先级规则从多个来源选择 URL，返回确定性的有序列表。

    优先级规则（从高到低）：
    1. changelog_links（变更日志链接，通常包含最新信息）
    2. llms_txt 中的链接（专门为 LLM 准备的文档）
    3. related_doc_urls（相关文档链接）
    4. core_docs（核心文档，作为兜底）

    关键词匹配会提升 URL 的优先级权重。

    Args:
        changelog_links: 变更日志中的链接
        related_doc_urls: 相关文档链接
        llms_txt_content: llms.txt 文件内容（可为空）
        core_docs: 核心文档 URL 列表
        keywords: 用于优先级提升的关键词
        config: 策略配置（None 使用默认配置）
        base_url: 基础 URL（用于规范化相对路径）

    Returns:
        选中的 URL 列表（确定性顺序，按优先级降序）

    Examples:
        >>> select_urls_to_fetch(
        ...     changelog_links=["https://example.com/changelog"],
        ...     related_doc_urls=["https://example.com/guide"],
        ...     llms_txt_content="https://example.com/llms-guide",
        ...     core_docs=["https://example.com/core"],
        ...     keywords=["guide"],
        ...     config=DocURLStrategyConfig(max_urls=3),
        ... )
        ['https://example.com/changelog', 'https://example.com/llms-guide', 'https://example.com/guide']
    """
    if config is None:
        config = DocURLStrategyConfig()

    # 收集所有候选 URL 及其优先级
    candidates: list[URLWithPriority] = []

    # 1. 处理 changelog 链接（最高优先级）
    changelog_weight = config.priority_weights.get("changelog", 3.0)
    for url in changelog_links:
        if config.normalize:
            url = normalize_url(url, base_url)
        if is_allowed_doc_url(url, config, base_url):
            kw_score = _calculate_keyword_score(url, keywords)
            priority = changelog_weight
            if kw_score > 0:
                priority += kw_score * config.keyword_boost_weight
            candidates.append(
                URLWithPriority(
                    url=url,
                    priority=priority,
                    source="changelog",
                    keyword_matched=kw_score > 0,
                )
            )

    # 2. 解析并处理 llms.txt 链接
    llms_weight = config.priority_weights.get("llms_txt", 2.5)
    if llms_txt_content:
        llms_urls = parse_llms_txt_urls(llms_txt_content, base_url)
        for url in llms_urls:
            if config.normalize:
                url = normalize_url(url, base_url)
            if is_allowed_doc_url(url, config, base_url):
                kw_score = _calculate_keyword_score(url, keywords)
                priority = llms_weight
                if kw_score > 0:
                    priority += kw_score * config.keyword_boost_weight
                candidates.append(
                    URLWithPriority(
                        url=url,
                        priority=priority,
                        source="llms_txt",
                        keyword_matched=kw_score > 0,
                    )
                )

    # 3. 处理相关文档链接
    related_weight = config.priority_weights.get("related_doc", 2.0)
    for url in related_doc_urls:
        if config.normalize:
            url = normalize_url(url, base_url)
        if is_allowed_doc_url(url, config, base_url):
            kw_score = _calculate_keyword_score(url, keywords)
            priority = related_weight
            if kw_score > 0:
                priority += kw_score * config.keyword_boost_weight
            candidates.append(
                URLWithPriority(
                    url=url,
                    priority=priority,
                    source="related_doc",
                    keyword_matched=kw_score > 0,
                )
            )

    # 4. 处理核心文档（兜底来源）
    core_weight = config.priority_weights.get("core_doc", 1.5)
    for url in core_docs:
        if config.normalize:
            url = normalize_url(url, base_url)
        if is_allowed_doc_url(url, config, base_url):
            kw_score = _calculate_keyword_score(url, keywords)
            priority = core_weight
            if kw_score > 0:
                priority += kw_score * config.keyword_boost_weight
            candidates.append(
                URLWithPriority(
                    url=url,
                    priority=priority,
                    source="core_doc",
                    keyword_matched=kw_score > 0,
                )
            )

    # 去重（保留优先级最高的版本）
    if config.deduplicate:
        url_to_best: dict[str, URLWithPriority] = {}
        for cand in candidates:
            if cand.url not in url_to_best or cand.priority > url_to_best[cand.url].priority:
                url_to_best[cand.url] = cand
        candidates = list(url_to_best.values())

    # 排序：按优先级降序，相同优先级按来源权重和 URL 字母序（确保确定性）
    source_order = {"changelog": 0, "llms_txt": 1, "related_doc": 2, "core_doc": 3}
    candidates.sort(
        key=lambda c: (
            -c.priority,  # 优先级降序
            source_order.get(c.source, 99),  # 来源顺序
            c.url,  # URL 字母序（确保确定性）
        )
    )

    # 提取 URL 列表
    result = [c.url for c in candidates]

    # 兜底策略：如果结果不足，从 core_docs 补充（在截断之前）
    min_required = min(config.fallback_core_docs_count, config.max_urls)
    if len(result) < min_required and core_docs:
        existing = set(result)
        for url in core_docs:
            if len(result) >= min_required:
                break
            normalized = normalize_url(url, base_url) if config.normalize else url
            if normalized not in existing and is_allowed_doc_url(normalized, config, base_url):
                result.append(normalized)
                existing.add(normalized)
        logger.debug(f"从 core_docs 补充 URL，当前数量: {len(result)}")

    # 截断到 max_urls（确保不超过限制）
    if len(result) > config.max_urls:
        result = result[: config.max_urls]
        logger.debug(f"URL 列表被截断到 {config.max_urls} 个")

    logger.info(f"选中 {len(result)} 个 URL 待获取")
    return result


def filter_urls_by_keywords(
    urls: list[str],
    keywords: list[str],
    min_match_score: float = 0.0,
) -> list[str]:
    """根据关键词过滤 URL

    保留至少匹配一个关键词的 URL。

    Args:
        urls: URL 列表
        keywords: 关键词列表
        min_match_score: 最小匹配得分阈值（0.0-1.0）

    Returns:
        过滤后的 URL 列表（保持原始顺序）
    """
    if not keywords:
        return urls

    result = []
    for url in urls:
        score = _calculate_keyword_score(url, keywords)
        if score >= min_match_score:
            result.append(url)

    return result


def extract_domain(url: str) -> str:
    """从 URL 提取域名

    Args:
        url: URL

    Returns:
        域名（小写）

    Examples:
        >>> extract_domain("https://docs.python.org/3/library/")
        'docs.python.org'
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def derive_primary_domains(
    llms_txt_url: Optional[str] = None,
    changelog_url: Optional[str] = None,
) -> list[str]:
    """从配置的 URL 推导主域名列表

    从 llms_txt_url 和 changelog_url 提取域名，用于判断内链/外链。
    同域视为内部链接。

    Args:
        llms_txt_url: llms.txt 文件 URL
        changelog_url: changelog 页面 URL

    Returns:
        主域名列表（去重，小写）

    Examples:
        >>> derive_primary_domains("https://cursor.com/llms.txt", "https://cursor.com/cn/changelog")
        ['cursor.com']
        >>> derive_primary_domains("https://docs.cursor.com/llms.txt", "https://cursor.com/changelog")
        ['cursor.com', 'docs.cursor.com']
    """
    domains: set[str] = set()

    for url in [llms_txt_url, changelog_url]:
        if url:
            domain = extract_domain(url)
            if domain:
                domains.add(domain)

    return sorted(domains)


def is_external_link(
    url: str,
    primary_domains: list[str],
    allowed_domains: Optional[list[str]] = None,
) -> bool:
    """判断 URL 是否为外链

    判定规则（阶段 2：外链分类）：
    - URL 域名匹配 primary_domains → 内链
    - URL 域名匹配 allowed_domains → 内链
    - 其他 → 外链

    Args:
        url: 待检查的 URL
        primary_domains: 主域名列表（从 llms_txt_url/changelog_url 推导）
        allowed_domains: 额外允许的域名列表（可选）

    Returns:
        True 如果是外链，False 如果是内链

    Examples:
        >>> is_external_link("https://cursor.com/docs", ["cursor.com"])
        False
        >>> is_external_link("https://github.com/repo", ["cursor.com"])
        True
        >>> is_external_link("https://api.cursor.com/v1", ["cursor.com"])
        False
    """
    url_domain = extract_domain(url)
    if not url_domain:
        return True  # 无法解析的 URL 视为外链

    # 检查主域名（支持子域名匹配）
    for domain in primary_domains:
        domain_lower = domain.lower()
        if url_domain == domain_lower or url_domain.endswith("." + domain_lower):
            return False

    # 检查额外允许的域名（支持子域名匹配）
    if allowed_domains:
        for domain in allowed_domains:
            domain_lower = domain.lower()
            if url_domain == domain_lower or url_domain.endswith("." + domain_lower):
                return False

    return True


@dataclass
class FetchPolicyResult:
    """apply_fetch_policy 的返回结果

    Attributes:
        urls_to_fetch: 允许抓取的 URL 列表
        filtered_urls: 被过滤的 URL 及原因
        external_links_recorded: 被记录（但未抓取）的外链
    """

    urls_to_fetch: list[str]
    filtered_urls: list[dict[str, str]]  # [{"url": "...", "reason": "..."}]
    external_links_recorded: list[str]


def _matches_path_prefixes(url: str, allowed_path_prefixes: list[str]) -> bool:
    """检查 URL 路径是否匹配允许的路径前缀列表

    用于 fetch_policy 的阶段 4 路径前缀检查。

    Args:
        url: 待检查的 URL
        allowed_path_prefixes: 允许的路径前缀列表（如 ["docs", "cn/docs"]）

    Returns:
        True 如果 URL 路径匹配任一前缀，或 allowed_path_prefixes 为空（全部允许）

    Examples:
        >>> _matches_path_prefixes("https://cursor.com/docs/cli", ["docs", "cn/docs"])
        True
        >>> _matches_path_prefixes("https://cursor.com/pricing", ["docs", "cn/docs"])
        False
        >>> _matches_path_prefixes("https://cursor.com/any/path", [])
        True
    """
    if not allowed_path_prefixes:
        # 空列表表示全部允许
        return True

    try:
        parsed = urlparse(url)
        # 移除路径开头的斜杠，以便与前缀匹配
        path = parsed.path.lstrip("/")

        for prefix in allowed_path_prefixes:
            # 移除前缀开头的斜杠（如果有）
            prefix_clean = prefix.lstrip("/")
            if not prefix_clean:
                continue

            # 路径以前缀开头（精确匹配或子路径匹配）
            if path == prefix_clean or path.startswith(prefix_clean + "/"):
                return True

        return False
    except Exception:
        return False


def apply_fetch_policy(
    urls: list[str],
    fetch_policy_mode: str,
    base_url: Optional[str] = None,
    primary_domains: Optional[list[str]] = None,
    allowed_domains: Optional[list[str]] = None,
    external_link_allowlist: "Optional[list[str] | ExternalLinkAllowlist]" = None,
    allowed_path_prefixes: Optional[list[str]] = None,
    enforce_path_prefixes: bool = False,
) -> FetchPolicyResult:
    """根据 fetch_policy 过滤 URL 列表

    根据 external_link_mode 策略处理外链：
    - record_only（默认）：外链从 urls_to_fetch 移除，但记录到 external_links_recorded
    - skip_all：外链既不抓取也不记录
    - fetch_allowlist：仅允许白名单命中的外链抓取

    当 enforce_path_prefixes=True 时，对内链执行阶段 4 路径前缀检查：
    - 内链路径必须匹配 allowed_path_prefixes 中的任一前缀
    - 不匹配的内链将被拒绝并记录原因为 "internal_link_path_not_allowed"

    此函数应在 _build_urls_to_fetch 之后、_fetch_related_docs 之前调用，
    确保外链不会被实际抓取（保持最小抓取面）。

    Args:
        urls: 待过滤的 URL 列表
        fetch_policy_mode: 外链处理模式 (record_only/skip_all/fetch_allowlist)
        base_url: 基础 URL（用于规范化）
        primary_domains: 主域名列表（内链判定）
        allowed_domains: 额外允许的域名列表
        external_link_allowlist: 外链白名单（fetch_allowlist 模式专用）
            可以是原始列表或结构化的 ExternalLinkAllowlist 对象
        allowed_path_prefixes: 允许的路径前缀列表（用于内链路径 gate）
            格式：路径前缀（不含 scheme/host），如 ["docs", "cn/docs"]
        enforce_path_prefixes: 是否启用内链路径前缀检查
            默认 False（Phase A 行为不变）
            设为 True 启用阶段 4 路径 gate

    Returns:
        FetchPolicyResult 包含：
        - urls_to_fetch: 允许抓取的 URL 列表
        - filtered_urls: 被过滤的 URL 及原因
        - external_links_recorded: 被记录的外链

    Examples:
        >>> result = apply_fetch_policy(
        ...     urls=["https://cursor.com/docs", "https://github.com/repo"],
        ...     fetch_policy_mode="record_only",
        ...     primary_domains=["cursor.com"],
        ... )
        >>> result.urls_to_fetch
        ['https://cursor.com/docs']
        >>> result.external_links_recorded
        ['https://github.com/repo']

        >>> # 使用结构化 ExternalLinkAllowlist
        >>> allowlist = validate_external_link_allowlist(["github.com"])
        >>> result = apply_fetch_policy(
        ...     urls=["https://cursor.com/docs", "https://github.com/repo"],
        ...     fetch_policy_mode="fetch_allowlist",
        ...     primary_domains=["cursor.com"],
        ...     external_link_allowlist=allowlist,
        ... )
        >>> result.urls_to_fetch
        ['https://cursor.com/docs', 'https://github.com/repo']

        >>> # 启用内链路径前缀检查
        >>> result = apply_fetch_policy(
        ...     urls=["https://cursor.com/docs/cli", "https://cursor.com/pricing"],
        ...     fetch_policy_mode="record_only",
        ...     primary_domains=["cursor.com"],
        ...     allowed_path_prefixes=["docs", "cn/docs"],
        ...     enforce_path_prefixes=True,
        ... )
        >>> result.urls_to_fetch
        ['https://cursor.com/docs/cli']
        >>> result.filtered_urls[0]["reason"]
        'internal_link_path_not_allowed'
    """
    # 规范化 mode 值
    mode = validate_external_link_mode(fetch_policy_mode)

    # 准备主域名列表
    if primary_domains is None:
        primary_domains = []
    if base_url:
        base_domain = extract_domain(base_url)
        if base_domain and base_domain not in primary_domains:
            primary_domains = list(primary_domains) + [base_domain]

    urls_to_fetch: list[str] = []
    filtered_urls: list[dict[str, str]] = []
    external_links_recorded: list[str] = []

    for url in urls:
        # 检查是否为外链
        is_external = is_external_link(url, primary_domains, allowed_domains)

        if not is_external:
            # 内链：检查阶段 4 路径前缀 gate
            if (
                enforce_path_prefixes
                and allowed_path_prefixes
                and not _matches_path_prefixes(url, allowed_path_prefixes)
            ):
                # 内链路径不匹配，拒绝并记录
                filtered_urls.append(
                    {
                        "url": url,
                        "reason": "internal_link_path_not_allowed",
                    }
                )
                logger.debug(f"内链路径不匹配 allowed_path_prefixes: {url} (prefixes: {allowed_path_prefixes})")
                continue

            # 内链通过所有检查，允许抓取
            urls_to_fetch.append(url)
            continue

        # 外链处理
        if mode == "skip_all":
            # skip_all：既不抓取也不记录
            filtered_urls.append(
                {
                    "url": url,
                    "reason": "external_link_skip_all",
                }
            )
        elif mode == "fetch_allowlist":
            # fetch_allowlist：检查白名单
            if external_link_allowlist and _matches_allowlist(url, external_link_allowlist):
                urls_to_fetch.append(url)
            else:
                filtered_urls.append(
                    {
                        "url": url,
                        "reason": "external_link_not_in_allowlist",
                    }
                )
                external_links_recorded.append(url)
        else:
            # record_only（默认）：不抓取但记录
            filtered_urls.append(
                {
                    "url": url,
                    "reason": "external_link_record_only",
                }
            )
            external_links_recorded.append(url)

    logger.debug(
        f"apply_fetch_policy: mode={mode}, enforce_path_prefixes={enforce_path_prefixes}, "
        f"input={len(urls)}, output={len(urls_to_fetch)}, "
        f"filtered={len(filtered_urls)}, recorded={len(external_links_recorded)}"
    )

    return FetchPolicyResult(
        urls_to_fetch=urls_to_fetch,
        filtered_urls=filtered_urls,
        external_links_recorded=external_links_recorded,
    )


def _matches_allowlist(
    url: str,
    allowlist: "list[str] | ExternalLinkAllowlist",
) -> bool:
    """检查 URL 是否匹配白名单

    白名单项可以是域名或 URL 前缀。支持原始列表或结构化 ExternalLinkAllowlist。

    匹配规则（优先级从高到低）：
    1. prefixes 列表中的完整 URL 前缀匹配
    2. domains 列表中的域名匹配（支持子域名）

    Args:
        url: 待检查的 URL
        allowlist: 白名单列表或 ExternalLinkAllowlist 对象

    Returns:
        True 如果匹配白名单
    """
    # 如果是 ExternalLinkAllowlist，直接使用其 matches 方法
    if isinstance(allowlist, ExternalLinkAllowlist):
        return allowlist.matches(url)

    # 原始列表：向后兼容处理
    url_lower = url.lower()
    url_domain = extract_domain(url)

    for item in allowlist:
        item_lower = item.lower()

        # 检查是否为域名格式（不含 scheme）
        if not item_lower.startswith(("http://", "https://")):
            # 检查是否带路径（如 "github.com/org"）
            if "/" in item_lower:
                # 带路径的域名格式，转为 URL 前缀匹配
                full_prefix = f"https://{item_lower}"
                if url_lower.startswith(full_prefix):
                    return True
            else:
                # 纯域名匹配（支持子域名）
                if url_domain == item_lower or url_domain.endswith("." + item_lower):
                    return True
        else:
            # URL 前缀匹配
            if url_lower.startswith(item_lower):
                return True

    return False


# ============================================================
# 配置校验 helpers（供 core/config.py 和入口脚本复用）
# ============================================================


# 有效的 external_link_mode 值
VALID_EXTERNAL_LINK_MODES = frozenset({"record_only", "skip_all", "fetch_allowlist"})

# 有效的 execution_mode 值
VALID_EXECUTION_MODES = frozenset({"cli", "cloud", "auto"})


def is_valid_external_link_mode(value: str) -> bool:
    """检查 external_link_mode 值是否有效

    Args:
        value: 待检查的值

    Returns:
        True 如果值有效，False 否则

    Examples:
        >>> is_valid_external_link_mode("record_only")
        True
        >>> is_valid_external_link_mode("invalid")
        False
    """
    return value in VALID_EXTERNAL_LINK_MODES


def is_valid_execution_mode(value: str) -> bool:
    """检查 execution_mode 值是否有效

    Args:
        value: 待检查的值

    Returns:
        True 如果值有效，False 否则

    Examples:
        >>> is_valid_execution_mode("cli")
        True
        >>> is_valid_execution_mode("invalid")
        False
    """
    return value in VALID_EXECUTION_MODES


def is_full_url_prefix(value: str) -> bool:
    """检查值是否为完整 URL 前缀格式（包含 scheme）

    用于区分新版完整 URL 前缀格式和旧版路径前缀格式。

    Args:
        value: 待检查的值

    Returns:
        True 如果是完整 URL 前缀（以 http:// 或 https:// 开头），False 否则

    Examples:
        >>> is_full_url_prefix("https://example.com/docs")
        True
        >>> is_full_url_prefix("docs")
        False
        >>> is_full_url_prefix("/docs")
        False
    """
    return value.startswith("http://") or value.startswith("https://")


def is_path_prefix(value: str) -> bool:
    """检查值是否为路径前缀格式（不包含 scheme）

    用于 fetch_policy.allowed_url_prefixes 的格式校验。

    Args:
        value: 待检查的值

    Returns:
        True 如果是路径前缀（不以 http:// 或 https:// 开头），False 否则

    Examples:
        >>> is_path_prefix("docs")
        True
        >>> is_path_prefix("cn/docs")
        True
        >>> is_path_prefix("https://example.com/docs")
        False
    """
    return not is_full_url_prefix(value)


def validate_external_link_mode(
    value: str,
    context: str = "external_link_mode",
) -> str:
    """校验 external_link_mode 值

    如果值无效，记录 warning 并返回默认值。

    Args:
        value: 待校验的值
        context: 上下文描述（用于日志）

    Returns:
        如果值有效返回原值，否则返回默认值 "record_only"

    Examples:
        >>> validate_external_link_mode("record_only")
        'record_only'
        >>> validate_external_link_mode("invalid")  # 会记录 warning
        'record_only'
    """
    if is_valid_external_link_mode(value):
        return value

    valid_values = ", ".join(sorted(VALID_EXTERNAL_LINK_MODES))
    logger.warning(f"{context} 值无效: '{value}'。有效值为: {valid_values}。已回退到默认值 'record_only'。")
    return "record_only"


def validate_execution_mode(
    value: str,
    context: str = "execution_mode",
) -> str:
    """校验 execution_mode 值

    如果值无效，记录 warning 并返回默认值。

    Args:
        value: 待校验的值
        context: 上下文描述（用于日志）

    Returns:
        如果值有效返回原值，否则返回默认值 "auto"

    Examples:
        >>> validate_execution_mode("auto")
        'auto'
        >>> validate_execution_mode("invalid")  # 会记录 warning
        'auto'
    """
    if is_valid_execution_mode(value):
        return value

    valid_values = ", ".join(sorted(VALID_EXECUTION_MODES))
    logger.warning(f"{context} 值无效: '{value}'。有效值为: {valid_values}。已回退到默认值 'auto'。")
    return "auto"


def validate_url_strategy_prefixes(
    prefixes: list[str],
    context: str = "url_strategy.allowed_url_prefixes",
) -> list[str]:
    """校验 url_strategy.allowed_url_prefixes 格式

    新版本要求使用完整 URL 前缀格式（含 scheme）。
    如果检测到旧版路径前缀格式，记录 warning 但保留原值。

    Args:
        prefixes: 待校验的前缀列表
        context: 上下文描述（用于日志）

    Returns:
        原值（不修改，仅记录 warning）

    Examples:
        >>> validate_url_strategy_prefixes(["https://example.com/docs"])
        ['https://example.com/docs']
        >>> validate_url_strategy_prefixes(["docs"])  # 会记录 warning
        ['docs']
    """
    if not prefixes:
        return prefixes

    deprecated = [p for p in prefixes if is_path_prefix(p)]
    if deprecated:
        logger.warning(
            f"检测到旧版 {context} 格式（路径前缀）: {deprecated}。"
            f"新版本要求使用完整 URL 前缀（如 'https://example.com/docs'）。"
            f"请更新配置文件。当前值将保持不变，但可能无法正确过滤 URL。"
        )

    return prefixes


def validate_fetch_policy_path_prefixes(
    prefixes: list[str],
    context: str = "fetch_policy.allowed_path_prefixes",
) -> list[str]:
    """校验 fetch_policy.allowed_path_prefixes 格式

    fetch_policy 使用路径前缀格式（不含 scheme）。
    如果检测到完整 URL 格式，记录 warning。

    Args:
        prefixes: 待校验的前缀列表
        context: 上下文描述（用于日志）

    Returns:
        原值（不修改，仅记录 warning）

    Examples:
        >>> validate_fetch_policy_path_prefixes(["docs", "cn/docs"])
        ['docs', 'cn/docs']
        >>> validate_fetch_policy_path_prefixes(["https://example.com/docs"])  # 会记录 warning
        ['https://example.com/docs']
    """
    if not prefixes:
        return prefixes

    full_urls = [p for p in prefixes if is_full_url_prefix(p)]
    if full_urls:
        logger.warning(
            f"{context} 应使用路径前缀格式（如 'docs', 'cn/docs'），"
            f"而非完整 URL 格式: {full_urls}。"
            f"当前值将保持不变，但可能导致意外行为。"
        )

    return prefixes


# ============================================================
# Deprecated 警告机制（统一管理，每类警告仅输出一次）
# ============================================================
#
# 统一的 deprecated 警告 key 命名规则：
# - doc_url_strategy.func.*: 函数级别的废弃警告
# - doc_url_strategy.format.*: 格式校验的废弃警告
#
# 测试用关键文案片段（用于断言，防止未来修改破坏兼容提示）：
# - "[DEPRECATED]" (统一前缀)
# - "已废弃" (中文提示)
# - "将在 v2.0 版本中移除" (移除计划)
# ============================================================

# 记录已警告的 deprecated 函数（避免重复警告）
_deprecated_func_warned: set[str] = set()

# deprecated 警告的统一 key 常量（便于测试断言和维护）
DEPRECATED_KEY_FUNC_VALIDATE_FETCH_POLICY_PREFIXES = "doc_url_strategy.func.validate_fetch_policy_prefixes"

# deprecated 警告的关键文案片段（用于测试断言）
DEPRECATED_MSG_FUNC_PREFIX = "[DEPRECATED]"
DEPRECATED_MSG_FUNC_DEPRECATED = "已废弃"
DEPRECATED_MSG_FUNC_WILL_REMOVE_V2 = "将在 v2.0 版本中移除"


def reset_deprecated_func_warnings() -> None:
    """重置 deprecated 函数警告状态（仅供测试使用）

    清空已警告的函数集合，使警告可以再次触发。
    """
    _deprecated_func_warned.clear()


@dataclass
class ExternalLinkAllowlist:
    """外链白名单解析结果（统一结构）

    将原始 allowlist 字符串列表解析为结构化的域名列表和 URL 前缀列表。

    匹配规则优先级（从高到低）：
    1. prefixes 列表中的完整 URL 前缀匹配（精确前缀匹配）
    2. domains 列表中的域名匹配（支持子域名匹配）

    使用示例：
        allowlist = validate_external_link_allowlist(["github.com", "https://docs.python.org/3"])
        # allowlist.domains = ["github.com"]
        # allowlist.prefixes = ["https://docs.python.org/3"]

    Attributes:
        domains: 域名列表（不含 scheme，支持子域名匹配）
            例如 "github.com" 可匹配 "api.github.com"
        prefixes: 完整 URL 前缀列表（含 scheme/host/path）
            例如 "https://github.com/org" 仅匹配该路径前缀
        invalid_items: 无效项列表（格式不合法或为空的项）
    """

    domains: list[str] = field(default_factory=list)
    prefixes: list[str] = field(default_factory=list)
    invalid_items: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """检查白名单是否为空（无任何有效项）"""
        return not self.domains and not self.prefixes

    def matches(self, url: str) -> bool:
        """检查 URL 是否匹配白名单

        匹配规则：
        1. prefixes 优先：URL 以任一前缀开头则匹配
        2. domains 回退：URL 域名匹配任一域名（支持子域名）

        Args:
            url: 待检查的 URL

        Returns:
            True 如果匹配白名单，False 否则
        """
        if not url:
            return False

        url_lower = url.lower()
        url_domain = extract_domain(url)

        # 优先检查 URL 前缀
        for prefix in self.prefixes:
            if url_lower.startswith(prefix.lower()):
                return True

        # 回退到域名检查（支持子域名）
        for domain in self.domains:
            domain_lower = domain.lower()
            if url_domain == domain_lower or url_domain.endswith("." + domain_lower):
                return True

        return False


def validate_external_link_allowlist(
    allowlist: list[str],
    context: str = "external_link_allowlist",
) -> ExternalLinkAllowlist:
    """解析并校验 external_link_allowlist 配置

    将原始字符串列表解析为结构化的 ExternalLinkAllowlist 对象，
    区分域名项和完整 URL 前缀项，并校验格式合法性。

    解析规则：
    - 以 http:// 或 https:// 开头的项 → prefixes（完整 URL 前缀）
    - 不以 scheme 开头的项 → domains（域名，支持子域名匹配）
    - 空字符串或仅空白的项 → invalid_items（无效项）

    Args:
        allowlist: 原始白名单列表
        context: 上下文描述（用于日志）

    Returns:
        ExternalLinkAllowlist 对象，包含解析后的 domains、prefixes 和 invalid_items

    Examples:
        >>> result = validate_external_link_allowlist([
        ...     "github.com",
        ...     "https://docs.python.org/3",
        ...     "api.openai.com",
        ...     "",  # 无效项
        ... ])
        >>> result.domains
        ['github.com', 'api.openai.com']
        >>> result.prefixes
        ['https://docs.python.org/3']
        >>> result.invalid_items
        ['']
    """
    if not allowlist:
        return ExternalLinkAllowlist()

    domains: list[str] = []
    prefixes: list[str] = []
    invalid_items: list[str] = []

    for item in allowlist:
        # 去除首尾空白
        item_stripped = item.strip() if item else ""

        # 空项视为无效
        if not item_stripped:
            invalid_items.append(item if item is not None else "")
            continue

        # 判断是完整 URL 前缀还是域名
        if item_stripped.startswith("http://") or item_stripped.startswith("https://"):
            # 完整 URL 前缀
            # 规范化 URL 前缀（移除末尾斜杠等）
            normalized = normalize_url(item_stripped)
            if normalized:
                prefixes.append(normalized)
            else:
                invalid_items.append(item)
        else:
            # 域名格式
            # 简单校验：不能包含路径分隔符 "/" 在域名部分
            # 支持如 "github.com/org" 这样的格式（域名+路径前缀）
            if "/" in item_stripped:
                # 带路径的域名格式，转为完整 URL 前缀
                # 假设使用 https scheme
                full_url = f"https://{item_stripped}"
                normalized = normalize_url(full_url)
                if normalized:
                    prefixes.append(normalized)
                else:
                    invalid_items.append(item)
            else:
                # 纯域名
                domains.append(item_stripped.lower())

    # 记录无效项警告
    if invalid_items:
        logger.warning(
            f"{context} 包含无效项: {invalid_items}。"
            f"有效格式: 域名（如 'github.com'）或完整 URL 前缀（如 'https://github.com/org'）。"
            f"无效项已被忽略。"
        )

    return ExternalLinkAllowlist(
        domains=domains,
        prefixes=prefixes,
        invalid_items=invalid_items,
    )


def validate_fetch_policy_prefixes(
    prefixes: list[str],
    context: str = "fetch_policy.allowed_path_prefixes (兼容别名: allowed_url_prefixes)",
) -> list[str]:
    """校验 fetch_policy 路径前缀格式（兼容别名函数）

    .. deprecated::
        此函数已废弃，请使用 :func:`validate_fetch_policy_path_prefixes` 替代。
        将在 v2.0 版本中移除。

    **命名迁移说明**:
        - 新版本字段名: ``fetch_policy.allowed_path_prefixes``
        - 旧版本字段名: ``fetch_policy.allowed_url_prefixes`` (仅作为兼容别名保留)
        - CLI 参数: ``--allowed-path-prefixes`` (新) / ``--allowed-url-prefixes`` (废弃)

    fetch_policy 使用**路径前缀格式**（不含 scheme），如 "docs", "cn/docs"。
    如果检测到完整 URL 格式，记录 warning。

    **术语区分**:
        - ``fetch_policy.allowed_path_prefixes``: 路径前缀，如 "docs", "cn/docs"
        - ``url_strategy.allowed_url_prefixes``: 完整 URL 前缀，如 "https://example.com/docs"

    Args:
        prefixes: 待校验的前缀列表
        context: 上下文描述（用于日志）。默认值明确标注兼容别名关系。

    Returns:
        原值（不修改，仅记录 warning）

    Examples:
        >>> validate_fetch_policy_prefixes(["docs", "cn/docs"])
        ['docs', 'cn/docs']
        >>> validate_fetch_policy_prefixes(["https://example.com/docs"])  # 会记录 warning
        ['https://example.com/docs']
    """
    # 一次性 deprecation warning（使用统一 key）
    key = DEPRECATED_KEY_FUNC_VALIDATE_FETCH_POLICY_PREFIXES
    if key not in _deprecated_func_warned:
        _deprecated_func_warned.add(key)
        logger.warning(
            f"{DEPRECATED_MSG_FUNC_PREFIX} 'validate_fetch_policy_prefixes' {DEPRECATED_MSG_FUNC_DEPRECATED}，"
            f"请使用 'validate_fetch_policy_path_prefixes'。"
            f"此函数{DEPRECATED_MSG_FUNC_WILL_REMOVE_V2}。"
        )
    # 兼容性 wrapper：调用新函数，保持行为不变
    return validate_fetch_policy_path_prefixes(prefixes, context)
