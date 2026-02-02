"""tests/test_execution_policy.py - SideEffectPolicy 和 compute_side_effects 单元测试

测试覆盖:
1. compute_side_effects 四种策略模式
2. SideEffectPolicy 属性和方法
3. 边界条件和组合场景
"""

import pytest

from core.execution_policy import (
    SideEffectPolicy,
    compute_side_effects,
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
        assert policy.dry_run is True      # 被 minimal 强制设置
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
            (False, False, False, {
                "allow_network_fetch": True,
                "allow_file_write": True,
                "allow_cache_write": True,
                "allow_git_operations": True,
                "allow_directory_create": True,
            }),
            # skip_online 模式
            (True, False, False, {
                "allow_network_fetch": False,
                "allow_file_write": True,
                "allow_cache_write": False,
                "allow_git_operations": True,
                "allow_directory_create": True,
            }),
            # dry_run 模式
            (False, True, False, {
                "allow_network_fetch": True,
                "allow_file_write": False,
                "allow_cache_write": False,
                "allow_git_operations": False,
                "allow_directory_create": False,
            }),
            # minimal 模式
            (False, False, True, {
                "allow_network_fetch": False,
                "allow_file_write": False,
                "allow_cache_write": False,
                "allow_git_operations": False,
                "allow_directory_create": False,
            }),
            # skip_online + dry_run = minimal
            (True, True, False, {
                "allow_network_fetch": False,
                "allow_file_write": False,
                "allow_cache_write": False,
                "allow_git_operations": False,
                "allow_directory_create": False,
            }),
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
