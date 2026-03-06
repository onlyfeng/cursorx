"""tests/test_execution_policy.py - 执行策略模块单元测试

测试覆盖:
1. compute_side_effects 四种策略模式
2. SideEffectPolicy 属性和方法
3. 边界条件和组合场景
4. resolve_requested_mode_for_decision 与 should_use_mp_orchestrator 联动测试
5. has_ampersand_prefix=True 时 build_execution_decision 决策验证
"""

import pytest

from core.execution_policy import (
    VIRTUAL_PROMPT_FOR_PREFIX_DETECTION,
    DecisionInputs,
    SideEffectPolicy,
    build_execution_decision,
    compute_decision_inputs,
    compute_message_dedup_key,
    compute_side_effects,
    resolve_requested_mode_for_decision,
    should_use_mp_orchestrator,
    validate_requested_mode_invariant,
)

# ============================================================
# compute_side_effects 测试
# ============================================================


class TestComputeSideEffects:
    """测试 compute_side_effects 函数"""

    def test_normal_mode_default(self):
        """默认参数应返回 normal 模式（允许所有副作用）"""
        policy = compute_side_effects()

        assert policy.allow_network_fetch is True
        assert policy.allow_file_write is True
        assert policy.allow_cache_write is True
        assert policy.allow_git_operations is True
        assert policy.allow_directory_create is True
        assert policy.is_normal is True
        assert policy.is_minimal is False

    def test_normal_mode_explicit_false(self):
        """显式 False 参数应返回 normal 模式"""
        policy = compute_side_effects(
            skip_online=False,
            dry_run=False,
            minimal=False,
        )

        assert policy.allow_network_fetch is True
        assert policy.allow_file_write is True
        assert policy.allow_cache_write is True
        assert policy.is_normal is True

    def test_skip_online_mode(self):
        """skip_online 模式：禁止网络请求和缓存写入，允许其他"""
        policy = compute_side_effects(skip_online=True)

        assert policy.allow_network_fetch is False
        assert policy.allow_file_write is True
        assert policy.allow_cache_write is False  # 无新数据可写
        assert policy.allow_git_operations is True
        assert policy.allow_directory_create is True
        assert policy.skip_online is True
        assert policy.is_minimal is False  # 不是 minimal

    def test_dry_run_mode(self):
        """dry_run 模式：允许网络请求用于分析，禁止所有写入"""
        policy = compute_side_effects(dry_run=True)

        assert policy.allow_network_fetch is True  # 允许分析
        assert policy.allow_file_write is False
        assert policy.allow_cache_write is False
        assert policy.allow_git_operations is False
        assert policy.allow_directory_create is False
        assert policy.dry_run is True
        assert policy.is_minimal is False  # 不是 minimal

    def test_minimal_mode(self):
        """minimal 模式：禁止所有副作用"""
        policy = compute_side_effects(minimal=True)

        assert policy.allow_network_fetch is False
        assert policy.allow_file_write is False
        assert policy.allow_cache_write is False
        assert policy.allow_git_operations is False
        assert policy.allow_directory_create is False
        assert policy.is_minimal is True
        assert policy.is_normal is False
        # minimal 强制设置 skip_online 和 dry_run
        assert policy.skip_online is True
        assert policy.dry_run is True

    def test_skip_online_plus_dry_run_equals_minimal(self):
        """skip_online + dry_run 组合等效于 minimal"""
        policy = compute_side_effects(skip_online=True, dry_run=True)

        assert policy.allow_network_fetch is False
        assert policy.allow_file_write is False
        assert policy.allow_cache_write is False
        assert policy.allow_git_operations is False
        assert policy.allow_directory_create is False
        assert policy.is_minimal is True  # is_minimal 应为 True

    def test_minimal_overrides_other_params(self):
        """minimal=True 应覆盖其他参数"""
        # 即使 skip_online=False, dry_run=False，minimal=True 也应强制设置
        policy = compute_side_effects(
            skip_online=False,
            dry_run=False,
            minimal=True,
        )

        assert policy.skip_online is True  # 被 minimal 强制设置
        assert policy.dry_run is True  # 被 minimal 强制设置
        assert policy.is_minimal is True


# ============================================================
# SideEffectPolicy 属性测试
# ============================================================


class TestSideEffectPolicyProperties:
    """测试 SideEffectPolicy 属性"""

    def test_is_minimal_property(self):
        """is_minimal 属性应正确计算"""
        # 两者都为 True 时为 minimal
        policy = SideEffectPolicy(skip_online=True, dry_run=True)
        assert policy.is_minimal is True

        # 只有一个为 True 时不是 minimal
        policy = SideEffectPolicy(skip_online=True, dry_run=False)
        assert policy.is_minimal is False

        policy = SideEffectPolicy(skip_online=False, dry_run=True)
        assert policy.is_minimal is False

        policy = SideEffectPolicy(skip_online=False, dry_run=False)
        assert policy.is_minimal is False

    def test_is_normal_property(self):
        """is_normal 属性应正确计算"""
        # 所有 allow_* 都为 True 时为 normal
        policy = SideEffectPolicy(
            allow_network_fetch=True,
            allow_file_write=True,
            allow_cache_write=True,
            allow_git_operations=True,
            allow_directory_create=True,
        )
        assert policy.is_normal is True

        # 任一 allow_* 为 False 时不是 normal
        policy = SideEffectPolicy(
            allow_network_fetch=False,
            allow_file_write=True,
            allow_cache_write=True,
            allow_git_operations=True,
            allow_directory_create=True,
        )
        assert policy.is_normal is False

    def test_to_dict(self):
        """to_dict 应返回完整的字典表示"""
        policy = compute_side_effects(skip_online=True)
        d = policy.to_dict()

        assert "allow_network_fetch" in d
        assert "allow_file_write" in d
        assert "allow_cache_write" in d
        assert "allow_git_operations" in d
        assert "allow_directory_create" in d
        assert "skip_online" in d
        assert "dry_run" in d
        assert "minimal" in d
        assert "is_minimal" in d
        assert "is_normal" in d

        assert d["skip_online"] is True
        assert d["allow_network_fetch"] is False

    def test_repr_normal(self):
        """repr 应正确显示 normal 模式"""
        policy = compute_side_effects()
        assert "normal" in repr(policy)

    def test_repr_minimal(self):
        """repr 应正确显示 minimal 模式"""
        policy = compute_side_effects(minimal=True)
        assert "minimal=True" in repr(policy)

    def test_repr_skip_online(self):
        """repr 应正确显示 skip_online 模式"""
        policy = compute_side_effects(skip_online=True)
        assert "skip_online" in repr(policy)

    def test_repr_dry_run(self):
        """repr 应正确显示 dry_run 模式"""
        policy = compute_side_effects(dry_run=True)
        assert "dry_run" in repr(policy)


# ============================================================
# 策略矩阵一致性测试
# ============================================================


class TestSideEffectMatrix:
    """验证策略矩阵与模块文档一致"""

    @pytest.mark.parametrize(
        "skip_online,dry_run,minimal,expected",
        [
            # normal 模式
            (
                False,
                False,
                False,
                {
                    "allow_network_fetch": True,
                    "allow_file_write": True,
                    "allow_cache_write": True,
                    "allow_git_operations": True,
                    "allow_directory_create": True,
                },
            ),
            # skip_online 模式
            (
                True,
                False,
                False,
                {
                    "allow_network_fetch": False,
                    "allow_file_write": True,
                    "allow_cache_write": False,
                    "allow_git_operations": True,
                    "allow_directory_create": True,
                },
            ),
            # dry_run 模式
            (
                False,
                True,
                False,
                {
                    "allow_network_fetch": True,
                    "allow_file_write": False,
                    "allow_cache_write": False,
                    "allow_git_operations": False,
                    "allow_directory_create": False,
                },
            ),
            # minimal 模式
            (
                False,
                False,
                True,
                {
                    "allow_network_fetch": False,
                    "allow_file_write": False,
                    "allow_cache_write": False,
                    "allow_git_operations": False,
                    "allow_directory_create": False,
                },
            ),
            # skip_online + dry_run = minimal
            (
                True,
                True,
                False,
                {
                    "allow_network_fetch": False,
                    "allow_file_write": False,
                    "allow_cache_write": False,
                    "allow_git_operations": False,
                    "allow_directory_create": False,
                },
            ),
        ],
    )
    def test_matrix_consistency(self, skip_online, dry_run, minimal, expected):
        """验证策略矩阵与文档定义一致"""
        policy = compute_side_effects(
            skip_online=skip_online,
            dry_run=dry_run,
            minimal=minimal,
        )

        for field, expected_value in expected.items():
            actual_value = getattr(policy, field)
            assert actual_value == expected_value, (
                f"策略矩阵不一致: {field} 期望 {expected_value}, 实际 {actual_value} "
                f"(skip_online={skip_online}, dry_run={dry_run}, minimal={minimal})"
            )


# ============================================================
# 与下游模块参数映射测试
# ============================================================


class TestDownstreamMapping:
    """测试 SideEffectPolicy 到下游模块参数的映射"""

    def test_knowledge_updater_mapping_normal(self):
        """normal 模式映射到 KnowledgeUpdater 参数"""
        policy = compute_side_effects()

        # KnowledgeUpdater 构造参数映射
        offline = not policy.allow_network_fetch
        disable_cache_write = not policy.allow_cache_write
        dry_run = not policy.allow_file_write

        assert offline is False
        assert disable_cache_write is False
        assert dry_run is False

    def test_knowledge_updater_mapping_skip_online(self):
        """skip_online 模式映射到 KnowledgeUpdater 参数"""
        policy = compute_side_effects(skip_online=True)

        offline = not policy.allow_network_fetch
        disable_cache_write = not policy.allow_cache_write
        dry_run = not policy.allow_file_write

        assert offline is True
        assert disable_cache_write is True
        assert dry_run is False  # 仍允许文件写入

    def test_knowledge_updater_mapping_dry_run(self):
        """dry_run 模式映射到 KnowledgeUpdater 参数"""
        policy = compute_side_effects(dry_run=True)

        offline = not policy.allow_network_fetch
        disable_cache_write = not policy.allow_cache_write
        dry_run_param = not policy.allow_file_write

        assert offline is False  # 允许网络请求用于分析
        assert disable_cache_write is True
        assert dry_run_param is True

    def test_knowledge_updater_mapping_minimal(self):
        """minimal 模式映射到 KnowledgeUpdater 参数"""
        policy = compute_side_effects(minimal=True)

        offline = not policy.allow_network_fetch
        disable_cache_write = not policy.allow_cache_write
        dry_run_param = not policy.allow_file_write

        assert offline is True
        assert disable_cache_write is True
        assert dry_run_param is True

    def test_changelog_analyzer_should_skip(self):
        """测试 ChangelogAnalyzer.analyze 是否应该跳过"""
        # normal 模式不跳过
        policy = compute_side_effects()
        should_skip = not policy.allow_network_fetch
        assert should_skip is False

        # skip_online 模式跳过
        policy = compute_side_effects(skip_online=True)
        should_skip = not policy.allow_network_fetch
        assert should_skip is True

        # minimal 模式跳过
        policy = compute_side_effects(minimal=True)
        should_skip = not policy.allow_network_fetch
        assert should_skip is True

    def test_knowledge_updater_initialize_should_skip(self):
        """测试 KnowledgeUpdater.initialize 是否应该跳过"""
        # normal 模式不跳过
        policy = compute_side_effects()
        should_skip = not policy.allow_directory_create
        assert should_skip is False

        # dry_run 模式跳过
        policy = compute_side_effects(dry_run=True)
        should_skip = not policy.allow_directory_create
        assert should_skip is True

        # minimal 模式跳过
        policy = compute_side_effects(minimal=True)
        should_skip = not policy.allow_directory_create
        assert should_skip is True


# ============================================================
# resolve_requested_mode_for_decision 与 should_use_mp_orchestrator 联动测试
# ============================================================


class TestResolveRequestedModeForDecision:
    """测试 resolve_requested_mode_for_decision 函数及其与 should_use_mp_orchestrator 的联动"""

    @pytest.mark.parametrize(
        "cli_execution_mode,has_ampersand_prefix,config_execution_mode,expected_requested_mode,expected_can_use_mp",
        [
            # ================================================================
            # 场景 1: has_ampersand_prefix=False 且 cli_execution_mode=None
            # requested_mode_for_decision 必须等于 config_execution_mode
            # ================================================================
            # config=auto -> requested=auto -> can_use_mp=False
            (None, False, "auto", "auto", False),
            # config=cloud -> requested=cloud -> can_use_mp=False
            (None, False, "cloud", "cloud", False),
            # config=cli -> requested=cli -> can_use_mp=True
            (None, False, "cli", "cli", True),
            # config=plan -> requested=plan -> can_use_mp=True
            (None, False, "plan", "plan", True),
            # config=ask -> requested=ask -> can_use_mp=True
            (None, False, "ask", "ask", True),
            # config=None -> requested=DEFAULT_EXECUTION_MODE="auto" -> can_use_mp=False
            # 当 config_execution_mode=None 时，使用 DEFAULT_EXECUTION_MODE="auto"
            (None, False, None, "auto", False),
            # ================================================================
            # 场景 2: CLI 显式设置，优先级最高
            # ================================================================
            # CLI=cli 覆盖 config=auto
            ("cli", False, "auto", "cli", True),
            # CLI=cloud 覆盖 config=cli
            ("cloud", False, "cli", "cloud", False),
            # CLI=auto 覆盖 config=cli
            ("auto", False, "cli", "auto", False),
            # CLI 设置时 has_ampersand_prefix 不影响 requested_mode
            ("cli", True, "auto", "cli", True),
            ("cloud", True, "cli", "cloud", False),
            # ================================================================
            # 场景 3: has_ampersand_prefix=True 且 cli_execution_mode=None
            # 返回 None 是合法的，后续由 build_execution_decision 决策
            # ================================================================
            # 有 & 前缀，无 CLI 设置 -> 返回 None
            (None, True, "auto", None, True),
            (None, True, "cli", None, True),
            (None, True, "cloud", None, True),
            (None, True, None, None, True),
        ],
    )
    def test_requested_mode_and_mp_orchestrator(
        self,
        cli_execution_mode,
        has_ampersand_prefix,
        config_execution_mode,
        expected_requested_mode,
        expected_can_use_mp,
    ):
        """验证 resolve_requested_mode_for_decision 返回值及 should_use_mp_orchestrator 判断"""
        # Step 1: 解析 requested_mode_for_decision
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=cli_execution_mode,
            has_ampersand_prefix=has_ampersand_prefix,
            config_execution_mode=config_execution_mode,
        )

        # Step 2: 断言 requested_mode_for_decision
        assert requested_mode == expected_requested_mode, (
            f"resolve_requested_mode_for_decision 返回值不符预期: "
            f"cli={cli_execution_mode}, has_prefix={has_ampersand_prefix}, "
            f"config={config_execution_mode} -> 期望 {expected_requested_mode}, 实际 {requested_mode}"
        )

        # Step 3: 断言 should_use_mp_orchestrator
        can_use_mp = should_use_mp_orchestrator(requested_mode)
        assert can_use_mp == expected_can_use_mp, (
            f"should_use_mp_orchestrator 判断不符预期: "
            f"requested_mode={requested_mode} -> 期望 can_use_mp={expected_can_use_mp}, 实际 {can_use_mp}"
        )


# ============================================================
# has_ampersand_prefix=True 时 build_execution_decision 决策验证
# ============================================================


class TestAmpersandPrefixRoutingDecision:
    """测试 has_ampersand_prefix=True 场景下 build_execution_decision 的决策

    当 resolve_requested_mode_for_decision 返回 None 时（即 has_ampersand_prefix=True
    且无 CLI 显式设置），后续由 build_execution_decision 决策：
    - prefix_routed: & 前缀是否成功触发 Cloud
    - effective_mode: 有效执行模式
    - orchestrator: 编排器类型
    """

    @pytest.mark.parametrize(
        "prompt,requested_mode,cloud_enabled,has_api_key,"
        "expected_prefix_routed,expected_effective_mode,expected_orchestrator",
        [
            # ================================================================
            # 场景 1: & 前缀成功路由到 Cloud
            # ================================================================
            # 所有条件满足 -> prefix_routed=True, effective_mode=cloud, orchestrator=basic
            ("& 分析代码", None, True, True, True, "cloud", "basic"),
            # ================================================================
            # 场景 2: & 前缀存在但未成功路由（各种原因）
            # ================================================================
            # 根据 AGENTS.md：& 前缀表达 Cloud 意图，即使未成功路由也使用 basic
            # 无 API Key -> prefix_routed=False, effective_mode=cli, orchestrator=basic
            ("& 分析代码", None, True, False, False, "cli", "basic"),
            # cloud_enabled=False -> prefix_routed=False, effective_mode=cli, orchestrator=basic
            ("& 分析代码", None, False, True, False, "cli", "basic"),
            # cloud_enabled=False 且无 API Key -> prefix_routed=False, cli, basic
            ("& 分析代码", None, False, False, False, "cli", "basic"),
            # ================================================================
            # 场景 3: 显式 CLI 模式忽略 & 前缀
            # ================================================================
            # requested_mode=cli -> 忽略 & 前缀, effective_mode=cli, orchestrator=mp
            ("& 分析代码", "cli", True, True, False, "cli", "mp"),
            # ================================================================
            # 场景 4: 显式 Cloud/Auto 模式（& 前缀不算触发）
            # ================================================================
            # requested_mode=cloud（显式）-> prefix_routed=False, effective_mode=cloud, orchestrator=basic
            ("& 分析代码", "cloud", True, True, False, "cloud", "basic"),
            # requested_mode=auto（显式）-> prefix_routed=False, effective_mode=auto, orchestrator=basic
            ("& 分析代码", "auto", True, True, False, "auto", "basic"),
            # 显式 cloud 但无 API Key -> 回退 cli，但因 requested=cloud 仍强制 basic
            ("& 分析代码", "cloud", True, False, False, "cli", "basic"),
            # 显式 auto 但无 API Key -> 回退 cli，但因 requested=auto 仍强制 basic
            ("& 分析代码", "auto", True, False, False, "cli", "basic"),
            # ================================================================
            # 场景 5: 无 & 前缀的对照组
            # ================================================================
            # 无 & 前缀, requested=None -> cli, mp（函数级默认）
            ("分析代码", None, True, True, False, "cli", "mp"),
            # 无 & 前缀, requested=auto -> auto, basic
            ("分析代码", "auto", True, True, False, "auto", "basic"),
            # 无 & 前缀, requested=cli -> cli, mp
            ("分析代码", "cli", True, True, False, "cli", "mp"),
            # ================================================================
            # 场景 6: & 前缀 + plan/ask 只读模式（R-3 规则）
            # ================================================================
            # 只读模式不参与 Cloud 路由，& 前缀被忽略，允许 mp
            # requested_mode=plan + & 前缀 -> prefix_routed=False, effective_mode=plan, orchestrator=mp
            ("& 分析代码", "plan", True, True, False, "plan", "mp"),
            # requested_mode=plan + & 前缀 + 无 API Key -> 仍为 plan, mp（只读模式不受 API Key 影响）
            ("& 分析代码", "plan", True, False, False, "plan", "mp"),
            # requested_mode=plan + & 前缀 + cloud_disabled -> 仍为 plan, mp
            ("& 分析代码", "plan", False, True, False, "plan", "mp"),
            # requested_mode=ask + & 前缀 -> prefix_routed=False, effective_mode=ask, orchestrator=mp
            ("& 分析代码", "ask", True, True, False, "ask", "mp"),
            # requested_mode=ask + & 前缀 + 无 API Key -> 仍为 ask, mp
            ("& 分析代码", "ask", True, False, False, "ask", "mp"),
            # requested_mode=ask + & 前缀 + cloud_disabled -> 仍为 ask, mp
            ("& 分析代码", "ask", False, True, False, "ask", "mp"),
        ],
    )
    def test_build_execution_decision_with_ampersand_prefix(
        self,
        prompt,
        requested_mode,
        cloud_enabled,
        has_api_key,
        expected_prefix_routed,
        expected_effective_mode,
        expected_orchestrator,
    ):
        """验证 build_execution_decision 在 & 前缀场景下的决策"""
        decision = build_execution_decision(
            prompt=prompt,
            requested_mode=requested_mode,
            cloud_enabled=cloud_enabled,
            has_api_key=has_api_key,
        )

        # 断言 prefix_routed
        assert decision.prefix_routed == expected_prefix_routed, (
            f"prefix_routed 不符预期: prompt='{prompt}', requested={requested_mode}, "
            f"cloud_enabled={cloud_enabled}, has_api_key={has_api_key} -> "
            f"期望 {expected_prefix_routed}, 实际 {decision.prefix_routed}"
        )

        # 断言 effective_mode
        assert decision.effective_mode == expected_effective_mode, (
            f"effective_mode 不符预期: prompt='{prompt}', requested={requested_mode}, "
            f"cloud_enabled={cloud_enabled}, has_api_key={has_api_key} -> "
            f"期望 {expected_effective_mode}, 实际 {decision.effective_mode}"
        )

        # 断言 orchestrator
        assert decision.orchestrator == expected_orchestrator, (
            f"orchestrator 不符预期: prompt='{prompt}', requested={requested_mode}, "
            f"cloud_enabled={cloud_enabled}, has_api_key={has_api_key} -> "
            f"期望 {expected_orchestrator}, 实际 {decision.orchestrator}"
        )

        # 断言 has_ampersand_prefix（语法检测）与 prefix_routed（策略决策）的区分
        has_prefix = prompt.strip().startswith("&")
        assert decision.has_ampersand_prefix == has_prefix, (
            f"has_ampersand_prefix 不符预期: prompt='{prompt}' -> "
            f"期望 {has_prefix}, 实际 {decision.has_ampersand_prefix}"
        )

    def test_ampersand_prefix_return_none_is_valid_prefix_flow_only(self):
        """[prefix-flow] 验证 has_ampersand_prefix=True 时返回 None 是合法的

        ⚠ 此测试验证 None 仅在 & 前缀场景（prefix-flow）下是合法的：
        1. has_ampersand_prefix=True 且无 CLI 显式设置时，返回 None
        2. should_use_mp_orchestrator(None) 返回 True（允许 mp）
        3. 实际编排器由 build_execution_decision 根据 prefix_routed 决定

        **注意**：当 has_ampersand_prefix=False 且无 CLI 显式设置时，
        resolve_requested_mode_for_decision 返回 config.yaml 的默认值（如 "auto"），
        **不应为 None**。
        """
        # Step 1: 验证 & 前缀场景下返回 None
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        )
        assert requested_mode is None, (
            "has_ampersand_prefix=True 且无 CLI 设置时，应返回 None 让 build_execution_decision 处理"
        )

        # Step 2: 验证 should_use_mp_orchestrator(None) 返回 True
        # 这是 prefix-flow 场景下的合法调用
        assert should_use_mp_orchestrator(requested_mode) is True, (
            "[prefix-flow] should_use_mp_orchestrator(None) 应返回 True"
        )

        # Step 3: 验证实际编排器由 build_execution_decision 决定
        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=requested_mode,  # 来自 resolve_requested_mode_for_decision
            cloud_enabled=True,
            has_api_key=True,
        )
        assert decision.prefix_routed is True
        assert decision.orchestrator == "basic", "& 前缀成功路由时，即使 requested_mode=None，也应强制 basic 编排器"

        # Step 4: 对比验证 - 无 & 前缀时不应返回 None
        requested_mode_no_prefix = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        )
        assert requested_mode_no_prefix == "auto", "无 & 前缀时应返回 config.yaml 默认值，不应为 None"


# ============================================================
# resolve_requested_mode_for_decision 与 should_use_mp_orchestrator 核心断言专项测试
# ============================================================


class TestResolvedModeAndMpOrchestratorCoreAssertions:
    """resolve_requested_mode_for_decision 与 should_use_mp_orchestrator 核心断言专项测试

    本类是针对以下关键决策路径的专项显式测试：

    【断言组 1：无 & 前缀 + CLI 未指定 + config=auto 场景】
    - resolve_requested_mode_for_decision(None, False, 'auto') == 'auto'
    - should_use_mp_orchestrator('auto') is False（auto 模式强制 basic）

    【断言组 2：有 & 前缀 + CLI 未指定 场景（prefix-flow）】
    - resolve_requested_mode_for_decision(None, True, 'auto') is None
    - should_use_mp_orchestrator(None) is True（允许 mp 的前提条件之一）

    注意：should_use_mp_orchestrator(None) 返回 True 仅表示"不强制 basic"，
    实际编排器由 build_execution_decision 根据 prefix_routed 决定。
    """

    def test_no_ampersand_no_cli_config_auto_returns_auto(self):
        """【核心断言 1】无 & 前缀 + CLI 未指定 + config=auto → requested_mode='auto'

        验证：resolve_requested_mode_for_decision(None, False, 'auto') == 'auto'
        """
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        )
        assert requested_mode == "auto", "无 & 前缀 + CLI 未指定时，应返回 config.yaml 默认值 'auto'"

    def test_auto_mode_forces_basic_orchestrator(self):
        """【核心断言 2】should_use_mp_orchestrator('auto') is False

        验证：auto 模式强制使用 basic 编排器，不允许 MP。
        """
        can_use_mp = should_use_mp_orchestrator("auto")
        assert can_use_mp is False, "should_use_mp_orchestrator('auto') 应返回 False，强制 basic 编排器"

    def test_ampersand_prefix_no_cli_returns_none(self):
        """【核心断言 3】有 & 前缀 + CLI 未指定 → requested_mode=None（prefix-flow）

        验证：resolve_requested_mode_for_decision(None, True, 'auto') is None
        """
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        )
        assert requested_mode is None, "有 & 前缀 + CLI 未指定时，应返回 None（让 build_execution_decision 处理）"

    def test_none_mode_allows_mp_precondition(self):
        """【核心断言 4】should_use_mp_orchestrator(None) is True

        验证：requested_mode=None 时，should_use_mp_orchestrator 返回 True。
        这是允许 mp 的前提条件之一，但不代表最终一定使用 mp。
        实际编排器由 build_execution_decision 根据 prefix_routed 决定。
        """
        can_use_mp = should_use_mp_orchestrator(None)
        assert can_use_mp is True, "should_use_mp_orchestrator(None) 应返回 True（允许 mp 的前提条件）"

    def test_combined_flow_no_ampersand_auto_forces_basic(self):
        """【综合验证】无 & 前缀场景：完整流程验证 auto 模式强制 basic

        完整验证流程：
        1. resolve_requested_mode_for_decision(None, False, 'auto') == 'auto'
        2. should_use_mp_orchestrator('auto') is False
        """
        # Step 1: 解析 requested_mode
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="auto",
        )
        assert requested_mode == "auto"

        # Step 2: 验证强制 basic
        can_use_mp = should_use_mp_orchestrator(requested_mode)
        assert can_use_mp is False, "无 & 前缀 + CLI 未指定 + config=auto 场景：应强制 basic 编排器"

    def test_combined_flow_ampersand_prefix_allows_mp_precondition(self):
        """【综合验证】有 & 前缀场景：验证 None 允许 mp 的前提条件

        完整验证流程：
        1. resolve_requested_mode_for_decision(None, True, 'auto') is None
        2. should_use_mp_orchestrator(None) is True

        注意：虽然 should_use_mp_orchestrator(None) 返回 True，
        但实际编排器由 build_execution_decision 决定（通常为 basic）。
        """
        # Step 1: 解析 requested_mode（prefix-flow 场景）
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=True,
            config_execution_mode="auto",
        )
        assert requested_mode is None

        # Step 2: 验证 should_use_mp_orchestrator(None) 返回 True
        can_use_mp = should_use_mp_orchestrator(requested_mode)
        assert can_use_mp is True, "有 & 前缀场景：should_use_mp_orchestrator(None) 应返回 True"

        # Step 3: 验证实际编排器由 build_execution_decision 决定
        decision = build_execution_decision(
            prompt="& 分析代码",
            requested_mode=requested_mode,
            cloud_enabled=True,
            has_api_key=True,
        )
        # prefix_routed=True 时，实际编排器为 basic
        assert decision.prefix_routed is True
        assert decision.orchestrator == "basic", (
            "即使 should_use_mp_orchestrator(None)==True，& 前缀成功路由时仍强制 basic"
        )


# ============================================================
# 无 & 前缀 + CLI 未指定 + config=auto 关键场景测试
# ============================================================


class TestNoAmpersandNoCLIConfigAutoOrchestratorBasic:
    """测试无 & 前缀 + CLI 未指定 + config=auto 场景下 orchestrator=basic

    这是系统默认行为的核心测试场景：
    - 无 & 前缀：has_ampersand_prefix=False
    - CLI 未指定：cli_execution_mode=None
    - config=auto：config_execution_mode="auto"

    关键断言：
    - resolve_requested_mode_for_decision 返回 "auto"（来自 config.yaml）
    - build_execution_decision 返回 orchestrator="basic"

    回退场景：
    - requested_mode=auto/cloud 且 has_api_key=False → effective_mode=cli
    - 但 orchestrator 仍为 basic（基于 requested_mode 语义）
    """

    @pytest.mark.parametrize(
        "config_execution_mode,has_api_key,cloud_enabled,"
        "expected_requested_mode,expected_effective_mode,expected_orchestrator",
        [
            # ================================================================
            # 场景 1: 无 & 前缀 + CLI 未指定 + config=auto + 有 API Key
            # ================================================================
            ("auto", True, True, "auto", "auto", "basic"),
            # ================================================================
            # 场景 2: 无 & 前缀 + CLI 未指定 + config=auto + 无 API Key（回退场景）
            # 关键：effective_mode=cli（回退），但 orchestrator 仍为 basic
            # ================================================================
            ("auto", False, True, "auto", "cli", "basic"),
            # ================================================================
            # 场景 3: 无 & 前缀 + CLI 未指定 + config=cloud + 有 API Key
            # ================================================================
            ("cloud", True, True, "cloud", "cloud", "basic"),
            # ================================================================
            # 场景 4: 无 & 前缀 + CLI 未指定 + config=cloud + 无 API Key（回退场景）
            # 关键：effective_mode=cli（回退），但 orchestrator 仍为 basic
            # ================================================================
            ("cloud", False, True, "cloud", "cli", "basic"),
            # ================================================================
            # 场景 5: 无 & 前缀 + CLI 未指定 + config=cli
            # 此时可以使用 mp 编排器
            # ================================================================
            ("cli", True, True, "cli", "cli", "mp"),
            ("cli", False, True, "cli", "cli", "mp"),
        ],
    )
    def test_no_ampersand_no_cli_config_auto_orchestrator(
        self,
        config_execution_mode,
        has_api_key,
        cloud_enabled,
        expected_requested_mode,
        expected_effective_mode,
        expected_orchestrator,
    ):
        """验证无 & 前缀 + CLI 未指定场景下的决策

        复用 resolve_requested_mode_for_decision 与 build_execution_decision 函数，
        避免在测试里复制决策逻辑。
        """
        # Step 1: 使用 resolve_requested_mode_for_decision 解析 requested_mode
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,  # CLI 未指定
            has_ampersand_prefix=False,  # 无 & 前缀
            config_execution_mode=config_execution_mode,
        )

        # 断言 requested_mode 符合预期
        assert requested_mode == expected_requested_mode, (
            f"resolve_requested_mode_for_decision 返回值不符预期: "
            f"config={config_execution_mode} -> 期望 {expected_requested_mode}, 实际 {requested_mode}"
        )

        # Step 2: 使用 build_execution_decision 构建决策
        decision = build_execution_decision(
            prompt="普通任务描述",  # 无 & 前缀
            requested_mode=requested_mode,
            cloud_enabled=cloud_enabled,
            has_api_key=has_api_key,
        )

        # 断言 effective_mode
        assert decision.effective_mode == expected_effective_mode, (
            f"effective_mode 不符预期: requested={requested_mode}, "
            f"has_api_key={has_api_key} -> 期望 {expected_effective_mode}, 实际 {decision.effective_mode}"
        )

        # 断言 orchestrator（核心断言）
        assert decision.orchestrator == expected_orchestrator, (
            f"orchestrator 不符预期: requested={requested_mode}, "
            f"has_api_key={has_api_key}, effective={decision.effective_mode} -> "
            f"期望 {expected_orchestrator}, 实际 {decision.orchestrator}"
        )

        # 断言 prefix_routed=False（无 & 前缀）
        assert decision.prefix_routed is False, "无 & 前缀场景下 prefix_routed 应为 False"

        # 断言 has_ampersand_prefix=False
        assert decision.has_ampersand_prefix is False, "普通任务描述不应检测到 & 前缀"

    def test_fallback_scenario_auto_no_key_orchestrator_basic(self):
        """专项测试：requested_mode=auto + 无 API Key 回退场景

        场景：
        - resolve_requested_mode_for_decision(None, False, "auto") → "auto"
        - build_execution_decision(..., requested_mode="auto", has_api_key=False)
          → effective_mode="cli", orchestrator="basic"

        这是回退场景的核心测试：即使 effective_mode 回退到 cli，
        orchestrator 仍基于 requested_mode=auto 强制 basic。
        """
        # Step 1: 模拟入口脚本（run.py / scripts/run_iterate.py）的逻辑
        cli_execution_mode = None  # 用户未显式指定 --execution-mode
        has_ampersand_prefix = False  # 普通任务，无 & 前缀
        config_execution_mode = "auto"  # config.yaml 默认值

        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=cli_execution_mode,
            has_ampersand_prefix=has_ampersand_prefix,
            config_execution_mode=config_execution_mode,
        )

        # 断言：requested_mode 应为 "auto"（来自 config.yaml）
        assert requested_mode == "auto", "无 & 前缀且 CLI 未指定时，应使用 config.yaml 的 execution_mode"

        # Step 2: 构建执行决策
        decision = build_execution_decision(
            prompt="分析代码结构",
            requested_mode=requested_mode,
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key，触发回退
        )

        # 断言：effective_mode 应回退到 cli
        assert decision.effective_mode == "cli", "无 API Key 时，effective_mode 应回退到 cli"

        # 核心断言：orchestrator 仍为 basic（基于 requested_mode=auto）
        assert decision.orchestrator == "basic", (
            "回退场景关键断言：requested_mode=auto 时，即使回退到 cli，orchestrator 仍应为 basic"
        )

        # 断言：prefix_routed=False
        assert decision.prefix_routed is False

    def test_fallback_scenario_cloud_no_key_orchestrator_basic(self):
        """专项测试：requested_mode=cloud + 无 API Key 回退场景

        场景：
        - resolve_requested_mode_for_decision(None, False, "cloud") → "cloud"
        - build_execution_decision(..., requested_mode="cloud", has_api_key=False)
          → effective_mode="cli", orchestrator="basic"

        这是回退场景的核心测试：即使 effective_mode 回退到 cli，
        orchestrator 仍基于 requested_mode=cloud 强制 basic。
        """
        # Step 1: 模拟入口脚本逻辑
        requested_mode = resolve_requested_mode_for_decision(
            cli_execution_mode=None,
            has_ampersand_prefix=False,
            config_execution_mode="cloud",  # config.yaml 设置为 cloud
        )

        assert requested_mode == "cloud"

        # Step 2: 构建执行决策
        decision = build_execution_decision(
            prompt="长时间分析任务",
            requested_mode=requested_mode,
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key，触发回退
        )

        # 断言：effective_mode 应回退到 cli
        assert decision.effective_mode == "cli"

        # 核心断言：orchestrator 仍为 basic
        assert decision.orchestrator == "basic", (
            "回退场景关键断言：requested_mode=cloud 时，即使回退到 cli，orchestrator 仍应为 basic"
        )


# ============================================================
# compute_message_dedup_key 测试
# ============================================================


class TestComputeMessageDedupKey:
    """测试 compute_message_dedup_key 稳定哈希函数

    验证：
    1. 相同消息产生相同 dedup_key（稳定性）
    2. 不同消息产生不同 dedup_key（无误去重）
    3. 空字符串返回空字符串
    4. Unicode 字符正确处理
    """

    def test_same_message_same_key(self):
        """相同消息应产生相同 dedup_key"""
        msg = "⚠ 未设置 CURSOR_API_KEY，Cloud 模式不可用"
        key1 = compute_message_dedup_key(msg)
        key2 = compute_message_dedup_key(msg)

        assert key1 == key2, "相同消息应产生相同 dedup_key"
        assert len(key1) == 16, "dedup_key 应为 16 位十六进制字符串"

    def test_different_messages_different_keys(self):
        """不同消息应产生不同 dedup_key（无误去重）"""
        msg1 = "⚠ 未设置 CURSOR_API_KEY"
        msg2 = "ℹ 检测到 '&' 前缀但 cloud_enabled=False"
        msg3 = "Cloud 执行成功"

        key1 = compute_message_dedup_key(msg1)
        key2 = compute_message_dedup_key(msg2)
        key3 = compute_message_dedup_key(msg3)

        assert key1 != key2, "不同消息应产生不同 dedup_key"
        assert key2 != key3, "不同消息应产生不同 dedup_key"
        assert key1 != key3, "不同消息应产生不同 dedup_key"

    def test_empty_string_returns_empty(self):
        """空字符串应返回空字符串"""
        key = compute_message_dedup_key("")
        assert key == "", "空字符串应返回空 dedup_key"

    def test_unicode_support(self):
        """Unicode 字符应正确处理"""
        # 中文
        msg_cn = "这是一条中文消息"
        key_cn = compute_message_dedup_key(msg_cn)
        assert len(key_cn) == 16

        # 表情符号
        msg_emoji = "⚠ Warning 🚀"
        key_emoji = compute_message_dedup_key(msg_emoji)
        assert len(key_emoji) == 16

        # 验证不同 Unicode 消息产生不同 key
        assert key_cn != key_emoji

    def test_stability_across_calls(self):
        """多次调用应返回相同结果（稳定性）"""
        msg = "测试消息稳定性"
        keys = [compute_message_dedup_key(msg) for _ in range(100)]

        assert all(k == keys[0] for k in keys), "多次调用应返回相同 dedup_key"

    def test_whitespace_sensitive(self):
        """空白字符应影响 dedup_key"""
        msg1 = "消息"
        msg2 = " 消息"
        msg3 = "消息 "
        msg4 = "消 息"

        keys = [
            compute_message_dedup_key(msg1),
            compute_message_dedup_key(msg2),
            compute_message_dedup_key(msg3),
            compute_message_dedup_key(msg4),
        ]

        # 所有 key 应不同
        assert len(set(keys)) == 4, "不同空白的消息应产生不同 dedup_key"

    def test_dedup_key_format(self):
        """dedup_key 应为有效的十六进制字符串"""
        import re

        msg = "测试格式"
        key = compute_message_dedup_key(msg)

        # 应为 16 位十六进制字符串
        assert re.match(r"^[0-9a-f]{16}$", key), f"dedup_key 应为 16 位十六进制字符串，实际: {key}"


# ============================================================
# build_unified_overrides 与 ExecutionDecision 联动测试
# ============================================================


class TestBuildUnifiedOverridesWithExecutionDecision:
    """测试 build_unified_overrides 与 ExecutionDecision 联动

    验证当 execution_decision.orchestrator=basic 且 prefix_routed=False 时：
    - options.orchestrator == 'basic'
    - options.resolved['orchestrator'] == 'basic'

    这对应以下场景（见 AGENTS.md 决策矩阵）：
    1. & 前缀存在但因 cloud_enabled=False 未成功路由（R-2 规则）
    2. & 前缀存在但因无 API Key 未成功路由（R-2 规则）
    3. requested_mode=auto/cloud 但回退到 CLI（R-1 规则）
    """

    @pytest.fixture
    def mock_args(self):
        """创建模拟的 argparse.Namespace 对象"""
        import argparse

        return argparse.Namespace(
            task="测试任务",
            execution_mode=None,  # tri-state 未显式指定
            orchestrator=None,  # tri-state 未显式指定
            no_mp=False,
            workers=3,
            max_iterations="10",
            verbose=False,
            quiet=False,
            auto_commit=False,
            auto_push=False,
            dry_run=False,
            skip_online=False,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            _orchestrator_user_set=False,
            _directory_user_set=False,
            directory=".",
        )

    @pytest.fixture(autouse=True)
    def reset_config_manager(self):
        """每个测试前重置配置管理器"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.mark.parametrize(
        "prompt,cloud_enabled,has_api_key,expected_prefix_routed,expected_orchestrator,scenario_desc",
        [
            # ================================================================
            # 场景 1: & 前缀存在，cloud_enabled=False → prefix_routed=False, basic
            # ================================================================
            (
                "& 分析代码",
                False,  # cloud_enabled=False
                True,  # has_api_key=True（无影响，因为 cloud_enabled=False）
                False,  # expected_prefix_routed=False
                "basic",
                "& 前缀 + cloud_enabled=False → prefix_routed=False, orchestrator=basic",
            ),
            # ================================================================
            # 场景 2: & 前缀存在，无 API Key → prefix_routed=False, basic
            # ================================================================
            (
                "& 分析代码",
                True,  # cloud_enabled=True
                False,  # has_api_key=False
                False,  # expected_prefix_routed=False
                "basic",
                "& 前缀 + 无 API Key → prefix_routed=False, orchestrator=basic",
            ),
            # ================================================================
            # 场景 3: & 前缀存在，cloud_enabled=False 且无 API Key
            # ================================================================
            (
                "& 分析代码",
                False,  # cloud_enabled=False
                False,  # has_api_key=False
                False,  # expected_prefix_routed=False
                "basic",
                "& 前缀 + cloud_enabled=False + 无 API Key → prefix_routed=False, basic",
            ),
        ],
    )
    def test_build_unified_overrides_with_decision_basic_orchestrator_prefix_not_routed(
        self,
        mock_args,
        prompt,
        cloud_enabled,
        has_api_key,
        expected_prefix_routed,
        expected_orchestrator,
        scenario_desc,
    ):
        """验证 execution_decision.orchestrator=basic 且 prefix_routed=False 时
        build_unified_overrides 正确传播到 UnifiedOptions

        核心断言：
        - options.orchestrator == 'basic'
        - options.resolved['orchestrator'] == 'basic'
        """
        from unittest.mock import patch

        from core.config import build_unified_overrides

        # Step 1: 构建 ExecutionDecision
        decision = build_execution_decision(
            prompt=prompt,
            requested_mode=None,  # 无显式模式，由 & 前缀驱动
            cloud_enabled=cloud_enabled,
            has_api_key=has_api_key,
        )

        # 验证 ExecutionDecision 符合预期
        assert decision.prefix_routed == expected_prefix_routed, (
            f"[{scenario_desc}] ExecutionDecision.prefix_routed 不符预期: "
            f"期望 {expected_prefix_routed}, 实际 {decision.prefix_routed}"
        )
        assert decision.orchestrator == expected_orchestrator, (
            f"[{scenario_desc}] ExecutionDecision.orchestrator 不符预期: "
            f"期望 {expected_orchestrator}, 实际 {decision.orchestrator}"
        )

        # Step 2: 调用 build_unified_overrides，传入 execution_decision
        # Mock CloudClientFactory.resolve_api_key 以匹配测试场景
        api_key_value = "test_key" if has_api_key else None
        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key", return_value=api_key_value):
            options = build_unified_overrides(
                args=mock_args,
                nl_options=None,
                execution_decision=decision,
            )

        # Step 3: 断言 UnifiedOptions.orchestrator
        assert options.orchestrator == expected_orchestrator, (
            f"[{scenario_desc}] UnifiedOptions.orchestrator 不符预期: "
            f"期望 {expected_orchestrator}, 实际 {options.orchestrator}"
        )

        # Step 4: 断言 UnifiedOptions.resolved['orchestrator']
        assert options.resolved["orchestrator"] == expected_orchestrator, (
            f"[{scenario_desc}] UnifiedOptions.resolved['orchestrator'] 不符预期: "
            f"期望 {expected_orchestrator}, 实际 {options.resolved['orchestrator']}"
        )

        # Step 5: 断言 prefix_routed 字段传播正确
        assert options.prefix_routed == expected_prefix_routed, (
            f"[{scenario_desc}] UnifiedOptions.prefix_routed 不符预期: "
            f"期望 {expected_prefix_routed}, 实际 {options.prefix_routed}"
        )

    def test_build_unified_overrides_requested_auto_no_key_forces_basic(
        self,
        mock_args,
    ):
        """验证 requested_mode=auto 无 API Key 回退时仍强制 basic（R-1 规则）

        场景：requested_mode=auto, has_api_key=False → effective_mode=cli 回退
        但因 requested_mode=auto，orchestrator 仍强制 basic
        """
        from unittest.mock import patch

        from core.config import build_unified_overrides

        # 构建 ExecutionDecision: requested_mode=auto, 无 API Key
        decision = build_execution_decision(
            prompt="分析代码",  # 无 & 前缀
            requested_mode="auto",
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key，触发回退
        )

        # 验证 ExecutionDecision
        assert decision.prefix_routed is False, "无 & 前缀不应 prefix_routed"
        assert decision.effective_mode == "cli", "无 API Key 应回退到 cli"
        assert decision.orchestrator == "basic", "requested=auto 强制 basic（R-1）"

        # 调用 build_unified_overrides
        with patch(
            "cursor.cloud_client.CloudClientFactory.resolve_api_key",
            return_value=None,  # 模拟无 API Key
        ):
            options = build_unified_overrides(
                args=mock_args,
                nl_options=None,
                execution_decision=decision,
            )

        # 断言
        assert options.orchestrator == "basic", "requested_mode=auto 无 API Key 回退后，orchestrator 应为 basic"
        assert options.resolved["orchestrator"] == "basic", "resolved['orchestrator'] 应为 basic"
        assert options.prefix_routed is False, "无 & 前缀，prefix_routed 应为 False"

    def test_build_unified_overrides_requested_cloud_no_key_forces_basic(
        self,
        mock_args,
    ):
        """验证 requested_mode=cloud 无 API Key 回退时仍强制 basic（R-1 规则）

        场景：requested_mode=cloud, has_api_key=False → effective_mode=cli 回退
        但因 requested_mode=cloud，orchestrator 仍强制 basic
        """
        from unittest.mock import patch

        from core.config import build_unified_overrides

        # 构建 ExecutionDecision: requested_mode=cloud, 无 API Key
        decision = build_execution_decision(
            prompt="分析代码",  # 无 & 前缀
            requested_mode="cloud",
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key，触发回退
        )

        # 验证 ExecutionDecision
        assert decision.prefix_routed is False, "无 & 前缀不应 prefix_routed"
        assert decision.effective_mode == "cli", "无 API Key 应回退到 cli"
        assert decision.orchestrator == "basic", "requested=cloud 强制 basic（R-1）"

        # 调用 build_unified_overrides
        with patch(
            "cursor.cloud_client.CloudClientFactory.resolve_api_key",
            return_value=None,  # 模拟无 API Key
        ):
            options = build_unified_overrides(
                args=mock_args,
                nl_options=None,
                execution_decision=decision,
            )

        # 断言
        assert options.orchestrator == "basic", "requested_mode=cloud 无 API Key 回退后，orchestrator 应为 basic"
        assert options.resolved["orchestrator"] == "basic", "resolved['orchestrator'] 应为 basic"
        assert options.prefix_routed is False, "无 & 前缀，prefix_routed 应为 False"


# ============================================================
# build_unified_overrides 缺失 execution_decision 时 auto_detect_cloud_prefix 配置测试
# ============================================================


class TestBuildUnifiedOverridesNoDecisionAutoDetectConfig:
    """测试 build_unified_overrides 在 execution_decision 缺失时读取 config.auto_detect_cloud_prefix

    当 execution_decision=None 时，build_unified_overrides 会通过 compute_decision_inputs
    重建决策。此时 auto_detect_cloud_prefix 配置决定 & 前缀是否参与 Cloud 路由。

    关键场景（与 AGENTS.md R-3 规则一致）：
    - auto_detect_cloud_prefix=False 时，& 前缀被忽略，orchestrator 允许 mp
    - auto_detect_cloud_prefix=True（默认）时，& 前缀触发 Cloud 意图，orchestrator 为 basic

    此测试验证 build_unified_overrides 的重建路径正确读取 config.auto_detect_cloud_prefix。
    """

    @pytest.fixture
    def mock_args_with_ampersand(self):
        """创建模拟的 argparse.Namespace 对象（无 auto_detect_cloud_prefix CLI 参数）"""
        import argparse

        return argparse.Namespace(
            task="& 测试任务",  # 带 & 前缀的任务
            execution_mode=None,  # tri-state 未显式指定
            orchestrator=None,  # tri-state 未显式指定
            no_mp=False,
            workers=3,
            max_iterations="10",
            verbose=False,
            quiet=False,
            auto_commit=False,
            auto_push=False,
            dry_run=False,
            skip_online=False,
            cloud_timeout=None,
            cloud_auth_timeout=None,
            _orchestrator_user_set=False,
            _directory_user_set=False,
            directory=".",
            auto_detect_cloud_prefix=None,  # tri-state: None=未设置，使用 config.yaml
        )

    @pytest.fixture(autouse=True)
    def reset_config_manager(self):
        """每个测试前重置配置管理器"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def _create_mock_config(self, auto_detect_cloud_prefix: bool = True):
        """创建模拟配置对象"""
        from unittest.mock import MagicMock

        from core.config import DEFAULT_PLANNER_MODEL, DEFAULT_REVIEWER_MODEL, DEFAULT_WORKER_MODEL

        mock_config = MagicMock()
        mock_config.cloud_agent.enabled = True
        mock_config.cloud_agent.execution_mode = "cli"
        mock_config.cloud_agent.auto_detect_cloud_prefix = auto_detect_cloud_prefix
        mock_config.cloud_agent.timeout = 300
        mock_config.cloud_agent.auth_timeout = 30
        mock_config.system.max_iterations = 10
        mock_config.system.worker_pool_size = 3
        mock_config.system.enable_sub_planners = True
        mock_config.system.strict_review = False
        mock_config.models.planner = DEFAULT_PLANNER_MODEL
        mock_config.models.worker = DEFAULT_WORKER_MODEL
        mock_config.models.reviewer = DEFAULT_REVIEWER_MODEL
        return mock_config

    def test_no_decision_config_auto_detect_false_allows_mp(
        self,
        mock_args_with_ampersand,
    ):
        """验证 execution_decision=None 且 config.auto_detect_cloud_prefix=False 时允许 mp

        场景：
        - execution_decision 未提供（触发重建）
        - config.yaml 设置 auto_detect_cloud_prefix=False
        - nl_options 包含带 & 前缀的 goal

        期望：
        - & 前缀被忽略（auto_detect=False）
        - orchestrator=mp（不强制 basic）
        - prefix_routed=False
        """
        from unittest.mock import patch

        from core.config import build_unified_overrides

        # 模拟配置：auto_detect_cloud_prefix=False
        mock_config = self._create_mock_config(auto_detect_cloud_prefix=False)

        # nl_options 包含带 & 前缀的任务
        nl_options = {
            "_original_goal": "& 分析代码架构",
            "goal": "分析代码架构",
        }

        # Mock get_config（在 core.config 模块级别，影响 compute_decision_inputs 的延迟导入）
        # 以及 CloudClientFactory.resolve_api_key
        with (
            patch("core.config.get_config", return_value=mock_config),
            patch(
                "cursor.cloud_client.CloudClientFactory.resolve_api_key",
                return_value="test_key",
            ),
        ):
            options = build_unified_overrides(
                args=mock_args_with_ampersand,
                nl_options=nl_options,
                execution_decision=None,  # 关键：不提供 execution_decision
            )

        # 核心断言：auto_detect_cloud_prefix=False 时，& 前缀被忽略，允许 mp
        assert options.orchestrator == "mp", (
            "config.auto_detect_cloud_prefix=False 时，& 前缀应被忽略，orchestrator=mp\n"
            "实际 orchestrator: " + options.orchestrator
        )
        assert options.resolved["orchestrator"] == "mp", "resolved['orchestrator'] 应为 mp"
        assert options.prefix_routed is False, "auto_detect_cloud_prefix=False 时，prefix_routed 应为 False"

    def test_no_decision_config_auto_detect_true_forces_basic(
        self,
        mock_args_with_ampersand,
    ):
        """验证 execution_decision=None 且 config.auto_detect_cloud_prefix=True（默认）时强制 basic

        场景：
        - execution_decision 未提供（触发重建）
        - config.yaml 设置 auto_detect_cloud_prefix=True（默认值）
        - nl_options 包含带 & 前缀的 goal

        期望：
        - & 前缀触发 Cloud 意图（auto_detect=True）
        - orchestrator=basic（& 前缀表达 Cloud 意图）
        - prefix_routed 取决于 has_api_key 和 cloud_enabled
        """
        from unittest.mock import patch

        from core.config import build_unified_overrides

        # 模拟配置：auto_detect_cloud_prefix=True（默认）
        mock_config = self._create_mock_config(auto_detect_cloud_prefix=True)

        # nl_options 包含带 & 前缀的任务
        nl_options = {
            "_original_goal": "& 分析代码架构",
            "goal": "分析代码架构",
        }

        # Mock get_config 和 CloudClientFactory（无 API Key，触发 prefix_routed=False）
        with (
            patch("core.config.get_config", return_value=mock_config),
            patch(
                "cursor.cloud_client.CloudClientFactory.resolve_api_key",
                return_value=None,  # 无 API Key
            ),
        ):
            options = build_unified_overrides(
                args=mock_args_with_ampersand,
                nl_options=nl_options,
                execution_decision=None,  # 关键：不提供 execution_decision
            )

        # 核心断言：auto_detect_cloud_prefix=True 时，& 前缀触发 Cloud 意图，强制 basic
        assert options.orchestrator == "basic", (
            "config.auto_detect_cloud_prefix=True 时，& 前缀应触发 Cloud 意图，orchestrator=basic\n"
            "实际 orchestrator: " + options.orchestrator
        )
        assert options.resolved["orchestrator"] == "basic", "resolved['orchestrator'] 应为 basic"
        # 无 API Key，prefix_routed=False（但仍强制 basic，因为 & 前缀表达 Cloud 意图）
        assert options.prefix_routed is False, "无 API Key 时 prefix_routed 应为 False"

    @pytest.mark.parametrize(
        "config_auto_detect,has_api_key,expected_orchestrator,scenario_desc",
        [
            # config auto_detect=False 场景：& 前缀被忽略，允许 mp
            (False, True, "mp", "auto_detect=False + has_key → mp"),
            (False, False, "mp", "auto_detect=False + no_key → mp"),
            # config auto_detect=True 场景：& 前缀触发 Cloud 意图，强制 basic
            (True, True, "basic", "auto_detect=True + has_key → basic (prefix_routed=True)"),
            (True, False, "basic", "auto_detect=True + no_key → basic (prefix_routed=False)"),
        ],
    )
    def test_no_decision_config_auto_detect_matrix(
        self,
        mock_args_with_ampersand,
        config_auto_detect,
        has_api_key,
        expected_orchestrator,
        scenario_desc,
    ):
        """矩阵测试：验证 execution_decision 缺失时 config.auto_detect_cloud_prefix 的影响

        此测试覆盖 R-3 规则（AGENTS.md）：
        - auto_detect_cloud_prefix=false 时，& 前缀被忽略，orchestrator 允许 mp
        - auto_detect_cloud_prefix=true 时，& 前缀表达 Cloud 意图，orchestrator 强制 basic
        """
        from unittest.mock import patch

        from core.config import build_unified_overrides

        # 模拟配置
        mock_config = self._create_mock_config(auto_detect_cloud_prefix=config_auto_detect)

        nl_options = {
            "_original_goal": "& 任务",
            "goal": "任务",
        }

        api_key_value = "test_key" if has_api_key else None

        with (
            patch("core.config.get_config", return_value=mock_config),
            patch(
                "cursor.cloud_client.CloudClientFactory.resolve_api_key",
                return_value=api_key_value,
            ),
        ):
            options = build_unified_overrides(
                args=mock_args_with_ampersand,
                nl_options=nl_options,
                execution_decision=None,
            )

        assert options.orchestrator == expected_orchestrator, (
            f"[{scenario_desc}] orchestrator 不符预期: 期望 {expected_orchestrator}, 实际 {options.orchestrator}"
        )
        assert options.resolved["orchestrator"] == expected_orchestrator, (
            f"[{scenario_desc}] resolved['orchestrator'] 不符预期"
        )


# ============================================================
# validate_requested_mode_invariant 测试
# ============================================================


class TestValidateRequestedModeInvariant:
    """测试 validate_requested_mode_invariant 函数

    验证：
    1. 正常场景（不变式满足）不产生警告
    2. 违反不变式时输出 warning 级别日志
    3. raise_on_violation=True 时抛出 ValueError
    4. has_ampersand_prefix=True 时不触发不变式检查
    """

    def test_invariant_satisfied_no_warning(self, caplog):
        """不变式满足时不应产生警告"""
        import logging

        caplog.set_level(logging.WARNING)

        # 正常场景：无 & 前缀且无 CLI 设置，requested_mode 来自 config
        validate_requested_mode_invariant(
            has_ampersand_prefix=False,
            cli_execution_mode=None,
            requested_mode_for_decision="auto",  # 非 None
            config_execution_mode="auto",
            caller_name="test",
        )

        # 不应有警告日志
        assert len(caplog.records) == 0, "不变式满足时不应产生警告"

    def test_invariant_violated_logs_warning(self, caplog):
        """违反不变式时应输出 warning 级别日志"""
        import logging

        caplog.set_level(logging.WARNING)

        # 违反不变式场景
        validate_requested_mode_invariant(
            has_ampersand_prefix=False,
            cli_execution_mode=None,
            requested_mode_for_decision=None,  # 违反不变式
            config_execution_mode=None,
            caller_name="test_caller",
        )

        # 应有 warning 日志
        assert len(caplog.records) == 1, "违反不变式时应产生警告"
        assert caplog.records[0].levelno == logging.WARNING
        assert "test_caller" in caplog.records[0].message
        assert "不变式违反" in caplog.records[0].message

    def test_invariant_violated_raises_with_flag(self):
        """raise_on_violation=True 时应抛出 ValueError"""
        with pytest.raises(ValueError) as excinfo:
            validate_requested_mode_invariant(
                has_ampersand_prefix=False,
                cli_execution_mode=None,
                requested_mode_for_decision=None,  # 违反不变式
                config_execution_mode=None,
                caller_name="test_caller",
                raise_on_violation=True,
            )

        assert "不变式违反" in str(excinfo.value)
        assert "test_caller" in str(excinfo.value)

    def test_ampersand_prefix_skips_invariant_check(self, caplog):
        """has_ampersand_prefix=True 时不触发不变式检查"""
        import logging

        caplog.set_level(logging.WARNING)

        # 有 & 前缀时，requested_mode=None 是合法的
        validate_requested_mode_invariant(
            has_ampersand_prefix=True,  # 有 & 前缀
            cli_execution_mode=None,
            requested_mode_for_decision=None,  # 合法的 None
            config_execution_mode="auto",
            caller_name="test",
        )

        # 不应有警告日志
        assert len(caplog.records) == 0, "有 & 前缀时 requested_mode=None 是合法的"

    def test_cli_explicit_skips_invariant_check(self, caplog):
        """cli_execution_mode 有值时不触发不变式检查"""
        import logging

        caplog.set_level(logging.WARNING)

        # CLI 显式设置时，requested_mode 来自 CLI，不触发不变式检查
        # 即使 requested_mode_for_decision=None（非预期但不触发）
        validate_requested_mode_invariant(
            has_ampersand_prefix=False,
            cli_execution_mode="cli",  # 有 CLI 设置
            requested_mode_for_decision=None,  # 理论上不应为 None，但不触发检查
            config_execution_mode="auto",
            caller_name="test",
        )

        # 不应有警告日志（不变式检查跳过）
        assert len(caplog.records) == 0

    def test_invariant_message_includes_config_and_default(self):
        """警告消息应包含 config 值和 DEFAULT_EXECUTION_MODE"""
        from core.config import DEFAULT_EXECUTION_MODE

        with pytest.raises(ValueError) as excinfo:
            validate_requested_mode_invariant(
                has_ampersand_prefix=False,
                cli_execution_mode=None,
                requested_mode_for_decision=None,
                config_execution_mode="cloud",  # 明确的 config 值
                caller_name="my_caller",
                raise_on_violation=True,
            )

        error_msg = str(excinfo.value)
        assert "my_caller" in error_msg, "错误消息应包含 caller_name"
        assert "cloud" in error_msg, "错误消息应包含 config_execution_mode 值"
        assert DEFAULT_EXECUTION_MODE in error_msg, "错误消息应包含 DEFAULT_EXECUTION_MODE"


# ============================================================
# resolve_orchestrator_settings 与 build_execution_decision 设计差异专项测试
# ============================================================


class TestResolveOrchestratorSettingsDesignBoundary:
    """resolve_orchestrator_settings 职责边界专项测试

    ================================================================================
    设计差异说明（这是设计而非 bug）
    ================================================================================

    【背景】
    当 `&` 前缀存在但未成功路由（prefix_routed=False, auto_detect_cloud_prefix=True）时，
    两个函数的行为存在设计差异：

    1. `resolve_orchestrator_settings(prefix_routed=False)`:
       - **不会**单独因 & 前缀未成功路由而强制 basic
       - 仅在 execution_mode=auto/cloud 或 prefix_routed=True 时强制 basic
       - 职责边界：不包含 prompt 解析或 & 前缀语法检测

    2. `build_execution_decision` (R-2 规则):
       - **会**因 & 前缀表达 Cloud 意图而返回 orchestrator=basic
       - 即使 prefix_routed=False（因无 API Key 或 cloud_disabled）
       - 将 orchestrator=basic 通过 overrides 传递给下游函数

    【正确调用流程】
    1. 调用 build_execution_decision() 获取决策
    2. 将 decision.orchestrator 写入 overrides
    3. 调用 resolve_orchestrator_settings(overrides) 应用配置
    4. 最终 orchestrator=basic（通过 overrides 传播）

    【本测试类目的】
    验证 resolve_orchestrator_settings 的职责边界：
    - 仅调用 resolve_orchestrator_settings(prefix_routed=False) 不应强制 basic
    - 强制 basic 的责任在 build_execution_decision（R-2 规则）
    """

    @pytest.fixture(autouse=True)
    def reset_config(self):
        """每个测试前重置配置单例"""
        from core.config import ConfigManager

        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def test_resolve_settings_prefix_routed_false_cli_mode_returns_mp(self):
        """设计断言：prefix_routed=False + execution_mode=cli → orchestrator=mp

        验证 resolve_orchestrator_settings 的职责边界：
        - 不包含 & 前缀语法检测
        - prefix_routed=False 不触发强制 basic（除非 execution_mode=auto/cloud）
        - R-2 规则（& 前缀表达 Cloud 意图）由 build_execution_decision 实现

        ⚠ 这是设计而非 bug：
        仅调用 resolve_orchestrator_settings 不足以实现 R-2 规则
        """
        from core.config import resolve_orchestrator_settings

        result = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,
        )

        # 核心断言：prefix_routed=False + cli 模式 → mp
        assert result["orchestrator"] == "mp", (
            "设计边界：resolve_orchestrator_settings(prefix_routed=False, execution_mode=cli) "
            "返回 orchestrator=mp，不负责 & 前缀语法检测"
        )

    def test_build_decision_ampersand_not_routed_returns_basic(self):
        """设计断言：build_execution_decision 对 & 前缀未成功路由返回 orchestrator=basic

        验证 build_execution_decision 的职责：
        - 包含 & 前缀语法检测
        - 实现 R-2 规则：& 前缀表达 Cloud 意图
        - 即使 prefix_routed=False 也返回 orchestrator=basic

        ⚠ 这与 resolve_orchestrator_settings 形成对比：
        - resolve_orchestrator_settings(prefix_routed=False) 不强制 basic
        - build_execution_decision 会因 & 前缀而强制 basic
        """
        decision = build_execution_decision(
            prompt="& 后台任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=False,  # 无 API Key → prefix_routed=False
            auto_detect_cloud_prefix=True,
        )

        # 验证 prefix_routed=False
        assert decision.has_ampersand_prefix is True, "应检测到 & 前缀"
        assert decision.prefix_routed is False, "无 API Key → prefix_routed=False"

        # 核心断言：orchestrator=basic（R-2 规则）
        assert decision.orchestrator == "basic", (
            "设计边界：build_execution_decision 对 & 前缀未成功路由返回 orchestrator=basic\n"
            "原因：R-2 规则 - & 前缀表达 Cloud 意图"
        )

    def test_design_difference_combined_flow_forces_basic(self):
        """综合测试：正确的组合调用方式产生 orchestrator=basic

        验证正确的调用流程：
        1. build_execution_decision → orchestrator=basic
        2. 将 orchestrator=basic 写入 overrides
        3. resolve_orchestrator_settings(overrides) → 读取 overrides["orchestrator"]=basic

        【关键差异对比】

        | 调用方式                                      | orchestrator |
        |-----------------------------------------------|--------------|
        | resolve_orchestrator_settings(prefix_routed=False, cli) | mp     |
        | build_execution_decision(& + no_key) + overrides传播    | basic  |

        这是设计而非 bug：分层设计保持职责清晰。
        """
        from core.config import resolve_orchestrator_settings

        # Step 1: build_execution_decision 返回 orchestrator=basic
        decision = build_execution_decision(
            prompt="& 分析",
            requested_mode=None,
            cloud_enabled=False,  # cloud_disabled → prefix_routed=False
            has_api_key=True,
            auto_detect_cloud_prefix=True,
        )

        assert decision.orchestrator == "basic", "build_execution_decision 应返回 basic"

        # Step 2: 仅调用 resolve_orchestrator_settings（对比测试）
        result_alone = resolve_orchestrator_settings(
            overrides={"execution_mode": "cli"},
            prefix_routed=False,
        )
        assert result_alone["orchestrator"] == "mp", (
            "对比：仅调用 resolve_orchestrator_settings(prefix_routed=False) → mp"
        )

        # Step 3: 正确的组合调用（通过 overrides 传播）
        result_combined = resolve_orchestrator_settings(
            overrides={
                "execution_mode": decision.effective_mode,
                "orchestrator": decision.orchestrator,  # 关键：传播 basic
            },
            prefix_routed=decision.prefix_routed,
        )
        assert result_combined["orchestrator"] == "basic", (
            "综合：通过 overrides 传播 orchestrator=basic（正确的调用方式）"
        )


# ============================================================
# compute_decision_inputs 测试
# ============================================================


class TestComputeDecisionInputs:
    """测试 compute_decision_inputs 函数

    验证此 helper 函数正确封装了决策输入构建逻辑：
    - 从 args 提取 CLI 参数
    - 从 nl_options 提取原始 prompt
    - 检测 & 前缀
    - 计算 requested_mode 和 mode_source
    """

    def _make_args(self, **kwargs):
        """创建模拟的 argparse.Namespace 对象"""
        import argparse

        args = argparse.Namespace()
        args.execution_mode = kwargs.get("execution_mode")
        args.orchestrator = kwargs.get("orchestrator")
        args.no_mp = kwargs.get("no_mp", False)
        args._orchestrator_user_set = kwargs.get("_orchestrator_user_set", False)
        # auto_detect_cloud_prefix: tri-state (None/True/False)
        # None 表示未设置，使用配置默认值
        args.auto_detect_cloud_prefix = kwargs.get("auto_detect_cloud_prefix")
        return args

    def test_basic_cli_mode(self):
        """测试 CLI 显式指定 execution_mode=cli"""
        from unittest.mock import patch

        args = self._make_args(execution_mode="cli")

        # Mock CloudClientFactory（在 cursor.cloud_client 模块中定义）和 get_config
        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                inputs = compute_decision_inputs(args, original_prompt="任务")

        assert inputs.requested_mode == "cli"
        assert inputs.mode_source == "cli"
        assert inputs.has_ampersand_prefix is False
        assert inputs.prompt == "任务"
        assert inputs.cloud_enabled is True
        assert inputs.has_api_key is True

    def test_ampersand_prefix_detected(self):
        """测试 & 前缀检测"""
        from unittest.mock import patch

        args = self._make_args()

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                inputs = compute_decision_inputs(args, original_prompt="& 分析代码")

        assert inputs.has_ampersand_prefix is True
        # 有 & 前缀时，requested_mode 应该为 None，让 build_execution_decision 处理
        assert inputs.requested_mode is None
        assert inputs.prompt == "& 分析代码"

    def test_nl_options_original_goal(self):
        """测试从 nl_options 提取 _original_goal"""
        from unittest.mock import patch

        args = self._make_args()
        nl_options = {"_original_goal": "& 后台任务", "goal": "后台任务"}

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                inputs = compute_decision_inputs(args, nl_options=nl_options)

        assert inputs.has_ampersand_prefix is True
        assert inputs.original_prompt == "& 后台任务"
        assert inputs.prompt == "& 后台任务"

    def test_nl_options_has_ampersand_prefix_override(self):
        """测试 nl_options 中已有的 has_ampersand_prefix 覆盖检测结果"""
        from unittest.mock import patch

        args = self._make_args()
        # prompt 没有 & 前缀，但 nl_options 中明确指定 has_ampersand_prefix=True
        nl_options = {"goal": "任务", "has_ampersand_prefix": True}

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                inputs = compute_decision_inputs(args, nl_options=nl_options)

        assert inputs.has_ampersand_prefix is True
        # goal="任务" 不是 & 前缀，所以原始 prompt 会被使用
        assert inputs.prompt == "任务"

    def test_virtual_prompt_when_no_original(self):
        """测试当原始 prompt 不可用但 has_ampersand_prefix=True 时使用虚拟 prompt"""
        from unittest.mock import patch

        args = self._make_args()
        nl_options = {"has_ampersand_prefix": True}  # 无 goal，仅有标记

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                inputs = compute_decision_inputs(args, nl_options=nl_options)

        assert inputs.has_ampersand_prefix is True
        assert inputs.prompt == VIRTUAL_PROMPT_FOR_PREFIX_DETECTION
        assert inputs.original_prompt is None

    def test_no_mp_flag_sets_orchestrator(self):
        """测试 --no-mp 标志设置 user_requested_orchestrator"""
        from unittest.mock import patch

        args = self._make_args(no_mp=True)

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                inputs = compute_decision_inputs(args, original_prompt="任务")

        assert inputs.user_requested_orchestrator == "basic"

    def test_orchestrator_user_set_flag(self):
        """测试 _orchestrator_user_set 标志"""
        from unittest.mock import patch

        args = self._make_args(
            orchestrator="mp",
            _orchestrator_user_set=True,
        )

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                inputs = compute_decision_inputs(args, original_prompt="任务")

        assert inputs.user_requested_orchestrator == "mp"

    def test_build_decision_method(self):
        """测试 DecisionInputs.build_decision() 方法"""
        from unittest.mock import patch

        args = self._make_args(execution_mode="cli")

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                inputs = compute_decision_inputs(args, original_prompt="任务")
                decision = inputs.build_decision()

        assert decision.effective_mode == "cli"
        assert decision.orchestrator == "mp"
        assert decision.has_ampersand_prefix is False

    def test_config_execution_mode_fallback(self):
        """测试无 CLI 参数时使用 config.yaml 的 execution_mode"""
        from unittest.mock import patch

        args = self._make_args()  # 无 execution_mode

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "cloud"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                inputs = compute_decision_inputs(args, original_prompt="任务")

        # 无 & 前缀，使用 config.yaml 的 execution_mode
        assert inputs.requested_mode == "cloud"
        assert inputs.mode_source == "config"

    def test_auto_detect_cloud_prefix_from_config(self):
        """测试从 config.yaml 读取 auto_detect_cloud_prefix"""
        from unittest.mock import patch

        args = self._make_args()  # auto_detect_cloud_prefix=None（未设置）

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = False  # 配置禁用

                inputs = compute_decision_inputs(args, original_prompt="任务")

        # 应使用配置值
        assert inputs.auto_detect_cloud_prefix is False

    def test_auto_detect_cloud_prefix_config_default_true(self):
        """测试 config.yaml 默认 auto_detect_cloud_prefix=True"""
        from unittest.mock import patch

        args = self._make_args()

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True  # 配置启用

                inputs = compute_decision_inputs(args, original_prompt="任务")

        assert inputs.auto_detect_cloud_prefix is True

    def test_cli_auto_detect_true_overrides_config_false(self):
        """测试 CLI 参数 auto_detect_cloud_prefix=True 覆盖配置 False"""
        from unittest.mock import patch

        # CLI 显式设置 auto_detect_cloud_prefix=True
        args = self._make_args(auto_detect_cloud_prefix=True)

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = False  # 配置禁用

                inputs = compute_decision_inputs(args, original_prompt="任务")

        # CLI 覆盖配置
        assert inputs.auto_detect_cloud_prefix is True

    def test_cli_auto_detect_false_overrides_config_true(self):
        """测试 CLI 参数 auto_detect_cloud_prefix=False 覆盖配置 True"""
        from unittest.mock import patch

        # CLI 显式设置 auto_detect_cloud_prefix=False
        args = self._make_args(auto_detect_cloud_prefix=False)

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True  # 配置启用

                inputs = compute_decision_inputs(args, original_prompt="任务")

        # CLI 覆盖配置
        assert inputs.auto_detect_cloud_prefix is False

    def test_cli_auto_detect_none_uses_config(self):
        """测试 CLI 参数 auto_detect_cloud_prefix=None 时使用配置值"""
        from unittest.mock import patch

        # CLI 未设置（tri-state None）
        args = self._make_args(auto_detect_cloud_prefix=None)

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = False  # 配置禁用

                inputs = compute_decision_inputs(args, original_prompt="任务")

        # None 回退到配置
        assert inputs.auto_detect_cloud_prefix is False

    def test_build_decision_uses_auto_detect_from_inputs(self):
        """测试 build_decision 传递 auto_detect_cloud_prefix 参数"""
        from unittest.mock import patch

        # CLI 显式禁用 auto_detect_cloud_prefix
        args = self._make_args(auto_detect_cloud_prefix=False)

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "cli"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True

                # 有 & 前缀的 prompt
                inputs = compute_decision_inputs(args, original_prompt="& 分析代码")

        # 验证 inputs 使用 CLI 覆盖
        assert inputs.auto_detect_cloud_prefix is False
        assert inputs.has_ampersand_prefix is True

        # 验证 build_decision 正确传递参数
        decision = inputs.build_decision()
        # 因为 auto_detect_cloud_prefix=False，& 前缀被忽略
        assert decision.prefix_routed is False
        # 因为 config.execution_mode=cli，effective_mode=cli
        assert decision.effective_mode == "cli"
        # 因为 & 前缀被忽略（auto_detect=False），允许 mp
        assert decision.orchestrator == "mp"

    def test_config_auto_detect_false_ignores_ampersand_prefix(self):
        """测试 config.yaml 设置 auto_detect_cloud_prefix=False 时 & 前缀被忽略

        验证当 config.yaml 中 cloud_agent.auto_detect_cloud_prefix=False 时：
        1. compute_decision_inputs 从配置读取该值（不再硬编码）
        2. inputs.auto_detect_cloud_prefix 为 False
        3. build_decision 返回 prefix_routed=False
        4. 编排器允许使用 mp（& 前缀被忽略）
        5. effective_mode 由 config.execution_mode 决定

        这证明 auto_detect_cloud_prefix 不再硬编码为 True，
        而是正确从 config.yaml 读取。
        """
        from unittest.mock import patch

        # CLI 未设置 auto_detect_cloud_prefix（tri-state None），使用配置值
        args = self._make_args(auto_detect_cloud_prefix=None)

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "cli"
                # 关键：配置 auto_detect_cloud_prefix=False
                mock_config.cloud_agent.auto_detect_cloud_prefix = False

                # 有 & 前缀的 prompt
                inputs = compute_decision_inputs(args, original_prompt="& x")

        # 核心断言 1：inputs.auto_detect_cloud_prefix 为 False（证明不再硬编码）
        assert inputs.auto_detect_cloud_prefix is False, (
            "inputs.auto_detect_cloud_prefix 应从 config.yaml 读取为 False，证明不再硬编码为 True"
        )

        # 验证语法层面检测到 & 前缀
        assert inputs.has_ampersand_prefix is True, "语法层面应检测到 & 前缀"

        # 构建决策
        decision = inputs.build_decision()

        # 核心断言 2：prefix_routed=False（& 前缀被忽略）
        assert decision.prefix_routed is False, "auto_detect_cloud_prefix=False 时，& 前缀应被忽略，prefix_routed=False"

        # 核心断言 3：orchestrator='mp'（允许 MP）
        assert decision.orchestrator == "mp", "& 前缀被忽略时，编排器应允许 mp"

        # 核心断言 4：effective_mode='cli'（由 config.execution_mode 决定）
        assert decision.effective_mode == "cli", "effective_mode 应由 config.execution_mode='cli' 决定"


class TestDecisionInputsDataclass:
    """测试 DecisionInputs 数据类"""

    def test_dataclass_fields(self):
        """测试数据类字段"""
        inputs = DecisionInputs(
            prompt="测试",
            requested_mode="cli",
            cloud_enabled=True,
            has_api_key=True,
        )

        assert inputs.prompt == "测试"
        assert inputs.requested_mode == "cli"
        assert inputs.cloud_enabled is True
        assert inputs.has_api_key is True
        assert inputs.auto_detect_cloud_prefix is True  # 默认值
        assert inputs.user_requested_orchestrator is None  # 默认值
        assert inputs.mode_source is None  # 默认值
        assert inputs.has_ampersand_prefix is False  # 默认值
        assert inputs.original_prompt is None  # 默认值

    def test_dataclass_with_all_fields(self):
        """测试数据类所有字段"""
        inputs = DecisionInputs(
            prompt="& 任务",
            requested_mode=None,
            cloud_enabled=True,
            has_api_key=True,
            auto_detect_cloud_prefix=True,
            user_requested_orchestrator="basic",
            mode_source="cli",
            has_ampersand_prefix=True,
            original_prompt="& 任务",
        )

        assert inputs.has_ampersand_prefix is True
        assert inputs.user_requested_orchestrator == "basic"
        assert inputs.mode_source == "cli"
        assert inputs.original_prompt == "& 任务"


# ============================================================
# auto_detect_cloud_prefix 优先级链测试: CLI > config > default
# ============================================================


class TestAutoDetectCloudPrefixPriorityChain:
    """测试 auto_detect_cloud_prefix 参数的优先级链: CLI > config > default

    验证配置优先级：
    1. CLI 参数（--auto-detect-cloud-prefix / --no-auto-detect-cloud-prefix）优先级最高
    2. config.yaml 中的 cloud_agent.auto_detect_cloud_prefix 次之
    3. 默认值 True 优先级最低

    此测试类与 TestComputeDecisionInputs 中的测试互补：
    - TestComputeDecisionInputs 测试单个场景
    - 本类提供显式的优先级链断言
    """

    def _make_args(self, auto_detect_cloud_prefix=None, **kwargs):
        """创建模拟的 argparse.Namespace 对象"""
        import argparse

        args = argparse.Namespace()
        args.execution_mode = kwargs.get("execution_mode")
        args.orchestrator = kwargs.get("orchestrator")
        args.no_mp = kwargs.get("no_mp", False)
        args._orchestrator_user_set = kwargs.get("_orchestrator_user_set", False)
        # auto_detect_cloud_prefix: tri-state (None/True/False)
        args.auto_detect_cloud_prefix = auto_detect_cloud_prefix
        return args

    def test_priority_1_cli_true_overrides_config_false(self):
        """优先级断言 1: CLI True 覆盖 config False

        CLI: --auto-detect-cloud-prefix (True)
        config.yaml: auto_detect_cloud_prefix=False
        期望: inputs.auto_detect_cloud_prefix=True (CLI 优先)
        """
        from unittest.mock import patch

        args = self._make_args(auto_detect_cloud_prefix=True)

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = False  # config 禁用

                inputs = compute_decision_inputs(args, original_prompt="任务")

        # 核心断言: CLI True 覆盖 config False
        assert inputs.auto_detect_cloud_prefix is True, "优先级 CLI > config: CLI True 应覆盖 config False"

    def test_priority_2_cli_false_overrides_config_true(self):
        """优先级断言 2: CLI False 覆盖 config True

        CLI: --no-auto-detect-cloud-prefix (False)
        config.yaml: auto_detect_cloud_prefix=True
        期望: inputs.auto_detect_cloud_prefix=False (CLI 优先)
        """
        from unittest.mock import patch

        args = self._make_args(auto_detect_cloud_prefix=False)

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True  # config 启用

                inputs = compute_decision_inputs(args, original_prompt="任务")

        # 核心断言: CLI False 覆盖 config True
        assert inputs.auto_detect_cloud_prefix is False, "优先级 CLI > config: CLI False 应覆盖 config True"

    def test_priority_3_cli_none_uses_config_true(self):
        """优先级断言 3: CLI None 时使用 config True

        CLI: 未指定 (None)
        config.yaml: auto_detect_cloud_prefix=True
        期望: inputs.auto_detect_cloud_prefix=True (来自 config)
        """
        from unittest.mock import patch

        args = self._make_args(auto_detect_cloud_prefix=None)  # CLI 未指定

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True  # config 启用

                inputs = compute_decision_inputs(args, original_prompt="任务")

        # 核心断言: CLI None 时使用 config True
        assert inputs.auto_detect_cloud_prefix is True, "优先级 config > default: CLI None 时应使用 config True"

    def test_priority_4_cli_none_uses_config_false(self):
        """优先级断言 4: CLI None 时使用 config False

        CLI: 未指定 (None)
        config.yaml: auto_detect_cloud_prefix=False
        期望: inputs.auto_detect_cloud_prefix=False (来自 config)
        """
        from unittest.mock import patch

        args = self._make_args(auto_detect_cloud_prefix=None)  # CLI 未指定

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = False  # config 禁用

                inputs = compute_decision_inputs(args, original_prompt="任务")

        # 核心断言: CLI None 时使用 config False
        assert inputs.auto_detect_cloud_prefix is False, "优先级 config > default: CLI None 时应使用 config False"

    def test_priority_5_default_value_is_true(self):
        """优先级断言 5: 默认值为 True

        验证 DecisionInputs 数据类和 CloudAgentConfig 的默认值为 True
        """
        from core.config import CloudAgentConfig

        # 验证 DecisionInputs 默认值
        inputs = DecisionInputs(
            prompt="任务",
            requested_mode="cli",
            cloud_enabled=True,
            has_api_key=True,
            # 不指定 auto_detect_cloud_prefix，使用默认值
        )
        assert inputs.auto_detect_cloud_prefix is True, "DecisionInputs.auto_detect_cloud_prefix 默认值应为 True"

        # 验证 CloudAgentConfig 默认值
        config = CloudAgentConfig()
        assert config.auto_detect_cloud_prefix is True, "CloudAgentConfig.auto_detect_cloud_prefix 默认值应为 True"

    @pytest.mark.parametrize(
        "cli_value,config_value,expected,description",
        [
            (True, True, True, "CLI True + config True → True"),
            (True, False, True, "CLI True + config False → True (CLI 优先)"),
            (False, True, False, "CLI False + config True → False (CLI 优先)"),
            (False, False, False, "CLI False + config False → False"),
            (None, True, True, "CLI None + config True → True (config)"),
            (None, False, False, "CLI None + config False → False (config)"),
        ],
    )
    def test_priority_matrix(self, cli_value, config_value, expected, description):
        """优先级矩阵测试: 验证所有 CLI × config 组合"""
        from unittest.mock import patch

        args = self._make_args(auto_detect_cloud_prefix=cli_value)

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "auto"
                mock_config.cloud_agent.auto_detect_cloud_prefix = config_value

                inputs = compute_decision_inputs(args, original_prompt="任务")

        assert inputs.auto_detect_cloud_prefix is expected, (
            f"[{description}] 期望 {expected}, 实际 {inputs.auto_detect_cloud_prefix}"
        )

    def test_priority_affects_prefix_routing(self):
        """验证优先级影响 & 前缀路由决策

        场景：config 启用，CLI 禁用 → & 前缀应被忽略
        """
        from unittest.mock import patch

        args = self._make_args(auto_detect_cloud_prefix=False)  # CLI 禁用

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "cli"
                mock_config.cloud_agent.auto_detect_cloud_prefix = True  # config 启用

                inputs = compute_decision_inputs(args, original_prompt="& 分析代码")
                decision = inputs.build_decision()

        # 验证 CLI 禁用生效
        assert inputs.auto_detect_cloud_prefix is False
        assert inputs.has_ampersand_prefix is True  # 语法检测到 & 前缀
        assert decision.prefix_routed is False  # 但因 auto_detect=False 未路由
        assert decision.orchestrator == "mp"  # 允许 mp（& 前缀被忽略）

    def test_priority_affects_orchestrator_selection(self):
        """验证优先级影响编排器选择

        场景：config 禁用，CLI 启用 → & 前缀应触发 Cloud 意图，强制 basic
        """
        from unittest.mock import patch

        args = self._make_args(auto_detect_cloud_prefix=True)  # CLI 启用

        with patch("cursor.cloud_client.CloudClientFactory.resolve_api_key") as mock_resolve:
            mock_resolve.return_value = "fake-key"
            with patch("core.config.get_config") as mock_get_config:
                mock_config = mock_get_config.return_value
                mock_config.cloud_agent.enabled = True
                mock_config.cloud_agent.execution_mode = "cli"
                mock_config.cloud_agent.auto_detect_cloud_prefix = False  # config 禁用

                inputs = compute_decision_inputs(args, original_prompt="& 分析代码")
                decision = inputs.build_decision()

        # 验证 CLI 启用生效
        assert inputs.auto_detect_cloud_prefix is True
        assert inputs.has_ampersand_prefix is True
        assert decision.prefix_routed is True  # 成功路由
        assert decision.orchestrator == "basic"  # 强制 basic
