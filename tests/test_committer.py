"""测试 CommitterAgent 类

测试 Git 操作相关功能：检查状态、生成提交信息、提交、回退
使用临时 git 仓库进行隔离测试
"""
import subprocess
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.committer import CommitResult, CommitterAgent, CommitterConfig
from core.base import AgentRole, AgentStatus


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """创建临时 git 仓库用于测试

    Yields:
        临时仓库路径
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # 初始化 git 仓库
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path, check=True, capture_output=True
    )

    # 创建初始提交
    initial_file = repo_path / "README.md"
    initial_file.write_text("# Test Repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path, check=True, capture_output=True
    )

    yield repo_path


@pytest.fixture
def committer_config(temp_git_repo: Path) -> CommitterConfig:
    """创建使用临时仓库的 CommitterConfig"""
    return CommitterConfig(
        name="test-committer",
        working_directory=str(temp_git_repo),
    )


@pytest.fixture
def committer(committer_config: CommitterConfig) -> CommitterAgent:
    """创建 CommitterAgent 实例"""
    return CommitterAgent(committer_config)


class TestCommitterConfig:
    """测试 CommitterConfig 配置类"""

    def test_default_values(self):
        """测试默认值"""
        config = CommitterConfig()

        assert config.name == "committer"
        assert config.working_directory == "."
        assert config.auto_push is False
        assert config.commit_message_style == "conventional"
        assert config.include_files == []
        assert ".env" in config.exclude_files
        assert "*.key" in config.exclude_files

    def test_custom_values(self):
        """测试自定义值"""
        config = CommitterConfig(
            name="my-committer",
            working_directory="/tmp/repo",
            auto_push=True,
            commit_message_style="simple",
            include_files=["*.py"],
            exclude_files=[".env", "*.log"],
        )

        assert config.name == "my-committer"
        assert config.working_directory == "/tmp/repo"
        assert config.auto_push is True
        assert config.commit_message_style == "simple"
        assert config.include_files == ["*.py"]
        assert config.exclude_files == [".env", "*.log"]


class TestCommitterAgentInit:
    """测试 CommitterAgent 初始化"""

    def test_default_init(self, committer_config: CommitterConfig):
        """测试默认配置初始化"""
        agent = CommitterAgent(committer_config)

        assert agent.role == AgentRole.WORKER
        assert agent.name == "test-committer"
        assert agent.status == AgentStatus.IDLE
        assert agent.committer_config == committer_config

    def test_id_format(self, committer_config: CommitterConfig):
        """测试 Agent ID 格式"""
        agent = CommitterAgent(committer_config)

        # ID 格式：{role}-{random_hex}
        assert agent.id.startswith("worker-")
        assert len(agent.id) > len("worker-")


class TestCheckStatus:
    """测试 check_status() 方法"""

    def test_clean_repo_returns_no_changes(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试干净仓库返回空更改列表"""
        status = committer.check_status()

        assert status["is_repo"] is True
        assert status["has_changes"] is False
        assert status["staged"] == []
        assert status["modified"] == []
        assert status["untracked"] == []
        assert "branch" in status

    def test_modified_file_detected(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试修改文件时返回正确的文件列表"""
        # 修改已有文件
        readme = temp_git_repo / "README.md"
        readme.write_text("# Updated Test Repo\nWith more content.\n")

        status = committer.check_status()

        assert status["is_repo"] is True
        assert status["has_changes"] is True
        assert "README.md" in status["modified"]

    def test_new_file_detected(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试新文件被检测为未跟踪"""
        # 创建新文件
        new_file = temp_git_repo / "new_file.txt"
        new_file.write_text("New content\n")

        status = committer.check_status()

        assert status["is_repo"] is True
        assert status["has_changes"] is True
        assert "new_file.txt" in status["untracked"]

    def test_staged_file_detected(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试已暂存的文件"""
        # 创建并暂存新文件
        new_file = temp_git_repo / "staged.txt"
        new_file.write_text("Staged content\n")
        subprocess.run(
            ["git", "add", "staged.txt"],
            cwd=temp_git_repo, check=True, capture_output=True
        )

        status = committer.check_status()

        assert status["is_repo"] is True
        assert status["has_changes"] is True
        assert "staged.txt" in status["staged"]

    def test_not_a_git_repo(self, tmp_path: Path):
        """测试非 git 仓库目录"""
        config = CommitterConfig(working_directory=str(tmp_path))
        agent = CommitterAgent(config)

        status = agent.check_status()

        assert status["is_repo"] is False
        assert "error" in status


class TestGenerateCommitMessage:
    """测试 generate_commit_message() 方法"""

    @pytest.mark.asyncio
    async def test_generate_message_from_diff(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试根据 diff 生成合理的提交信息"""
        # 修改文件创建 diff
        readme = temp_git_repo / "README.md"
        readme.write_text("# Updated Test Repo\n\nNew feature added.\n")

        # Mock cursor_client 返回生成的提交信息
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "feat(docs): 更新 README 文档"

        with patch.object(
            committer.cursor_client, "execute", new=AsyncMock(return_value=mock_result)
        ):
            message = await committer.generate_commit_message()

        assert message == "feat(docs): 更新 README 文档"

    @pytest.mark.asyncio
    async def test_generate_message_with_provided_diff(
        self, committer: CommitterAgent
    ):
        """测试使用提供的 diff 生成提交信息"""
        diff = """
diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1,2 @@
+def hello(): pass
"""
        files = ["test.py"]

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "feat: 添加 hello 函数"

        with patch.object(
            committer.cursor_client, "execute", new=AsyncMock(return_value=mock_result)
        ):
            message = await committer.generate_commit_message(diff=diff, files=files)

        assert message == "feat: 添加 hello 函数"

    @pytest.mark.asyncio
    async def test_fallback_message_when_cursor_fails(
        self, committer: CommitterAgent
    ):
        """测试 cursor 调用失败时返回默认信息"""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.output = None

        with patch.object(
            committer.cursor_client, "execute", new=AsyncMock(return_value=mock_result)
        ):
            message = await committer.generate_commit_message(diff="some diff")

        # 默认使用 conventional 风格
        assert message == "chore: 更新"

    @pytest.mark.asyncio
    async def test_fallback_message_simple_style(
        self, temp_git_repo: Path
    ):
        """测试 simple 风格的默认提交信息"""
        config = CommitterConfig(
            working_directory=str(temp_git_repo),
            commit_message_style="simple",
        )
        agent = CommitterAgent(config)

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.output = None

        with patch.object(
            agent.cursor_client, "execute", new=AsyncMock(return_value=mock_result)
        ):
            message = await agent.generate_commit_message(diff="some diff")

        assert message == "更新"

    @pytest.mark.asyncio
    async def test_empty_diff_returns_default(
        self, committer: CommitterAgent
    ):
        """测试空 diff 返回默认提交信息"""
        # 不 mock cursor_client，直接测试空 diff 路径
        with patch.object(
            committer, "_get_diff", return_value=""
        ):
            message = await committer.generate_commit_message()

        assert message == "chore: 更新"


class TestCommit:
    """测试 commit() 方法"""

    def test_commit_success(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试正常提交流程"""
        # 创建新文件
        new_file = temp_git_repo / "feature.py"
        new_file.write_text("def feature(): pass\n")

        result = committer.commit("feat: 添加新功能")

        assert result.success is True
        assert result.commit_hash != ""
        assert result.message == "feat: 添加新功能"
        assert "feature.py" in result.files_changed

    def test_commit_multiple_files(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试提交多个文件"""
        # 创建多个文件
        (temp_git_repo / "file1.py").write_text("# file1\n")
        (temp_git_repo / "file2.py").write_text("# file2\n")
        (temp_git_repo / "file3.py").write_text("# file3\n")

        result = committer.commit("feat: 添加多个文件")

        assert result.success is True
        assert len(result.files_changed) == 3

    def test_commit_specific_files(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试只提交指定文件"""
        # 创建多个文件
        (temp_git_repo / "include.py").write_text("# include\n")
        (temp_git_repo / "exclude.py").write_text("# exclude\n")

        result = committer.commit("feat: 只提交 include.py", files=["include.py"])

        assert result.success is True
        assert "include.py" in result.files_changed
        # exclude.py 应该仍未提交
        status = committer.check_status()
        assert "exclude.py" in status["untracked"]

    def test_commit_with_exclude_pattern(
        self, temp_git_repo: Path
    ):
        """测试排除文件模式"""
        config = CommitterConfig(
            working_directory=str(temp_git_repo),
            exclude_files=[".env", "*.secret", "*.log"],
        )
        agent = CommitterAgent(config)

        # 创建多个文件，包括应被排除的
        (temp_git_repo / "code.py").write_text("# code\n")
        (temp_git_repo / ".env").write_text("SECRET=xxx\n")
        (temp_git_repo / "data.secret").write_text("secret data\n")
        (temp_git_repo / "app.log").write_text("log content\n")

        result = agent.commit("feat: 添加代码")

        assert result.success is True
        assert "code.py" in result.files_changed
        # 排除的文件不应被提交
        assert ".env" not in result.files_changed
        assert "data.secret" not in result.files_changed
        assert "app.log" not in result.files_changed

    def test_commit_no_changes(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试无变更时提交失败"""
        result = committer.commit("test: 无变更提交")

        assert result.success is False
        assert "没有需要提交的变更" in result.error

    def test_commit_all_excluded(
        self, temp_git_repo: Path
    ):
        """测试所有文件都被排除时提交失败"""
        config = CommitterConfig(
            working_directory=str(temp_git_repo),
            exclude_files=["*.txt"],
        )
        agent = CommitterAgent(config)

        # 只创建会被排除的文件
        (temp_git_repo / "file.txt").write_text("content\n")

        result = agent.commit("test: 所有文件被排除")

        assert result.success is False
        assert "没有可提交的文件" in result.error


class TestRollback:
    """测试 rollback() 方法"""

    def test_rollback_soft(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试软回退（保留工作区和暂存区）"""
        # 创建并提交一个文件
        (temp_git_repo / "rollback_test.py").write_text("# test\n")
        committer.commit("feat: 测试回退")

        # 获取当前 HEAD
        initial_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_git_repo, capture_output=True, text=True
        ).stdout.strip()

        # 执行软回退
        result = committer.rollback(mode="soft", commit="HEAD~1")

        assert result.success is True
        # 检查 HEAD 已变更
        new_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_git_repo, capture_output=True, text=True
        ).stdout.strip()
        assert new_head != initial_head
        assert result.commit_hash == new_head

    def test_rollback_mixed(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试混合回退（保留工作区，清空暂存区）"""
        # 创建并提交一个文件
        (temp_git_repo / "mixed_test.py").write_text("# mixed\n")
        committer.commit("feat: mixed 测试")

        result = committer.rollback(mode="mixed", commit="HEAD~1")

        assert result.success is True

    def test_rollback_hard(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试硬回退（清空工作区和暂存区）"""
        # 创建并提交一个文件
        test_file = temp_git_repo / "hard_test.py"
        test_file.write_text("# hard\n")
        committer.commit("feat: hard 测试")

        # 确认文件存在
        assert test_file.exists()

        result = committer.rollback(mode="hard", commit="HEAD~1")

        assert result.success is True
        # 文件应该被删除
        assert not test_file.exists()

    def test_rollback_checkout_files(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试还原指定文件"""
        # 修改已有文件
        readme = temp_git_repo / "README.md"
        original_content = readme.read_text()
        readme.write_text("Modified content\n")

        # 使用 checkout 模式还原
        result = committer.rollback(mode="checkout", files=["README.md"])

        assert result.success is True
        assert readme.read_text() == original_content
        assert "README.md" in result.files_changed

    def test_rollback_invalid_mode(
        self, committer: CommitterAgent
    ):
        """测试无效的回退模式"""
        result = committer.rollback(mode="invalid")

        assert result.success is False
        assert "不支持的回退模式" in result.error

    def test_rollback_to_specific_commit(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试回退到指定提交"""
        # 创建多个提交
        (temp_git_repo / "commit1.py").write_text("# 1\n")
        committer.commit("feat: commit 1")

        # 记录这个提交的 hash
        first_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_git_repo, capture_output=True, text=True
        ).stdout.strip()

        (temp_git_repo / "commit2.py").write_text("# 2\n")
        committer.commit("feat: commit 2")

        (temp_git_repo / "commit3.py").write_text("# 3\n")
        committer.commit("feat: commit 3")

        # 回退到第一个提交
        result = committer.rollback(mode="soft", commit=first_commit)

        assert result.success is True
        assert result.commit_hash == first_commit


class TestMatchExcludePattern:
    """测试 _match_exclude_pattern() 方法"""

    def test_match_exact_filename(self, temp_git_repo: Path):
        """测试精确文件名匹配"""
        config = CommitterConfig(
            working_directory=str(temp_git_repo),
            exclude_files=[".env"],
        )
        agent = CommitterAgent(config)

        assert agent._match_exclude_pattern(".env") is True
        assert agent._match_exclude_pattern("src/.env") is True
        assert agent._match_exclude_pattern(".env.local") is False

    def test_match_glob_pattern(self, temp_git_repo: Path):
        """测试 glob 模式匹配"""
        config = CommitterConfig(
            working_directory=str(temp_git_repo),
            exclude_files=["*.key", "*.pem"],
        )
        agent = CommitterAgent(config)

        assert agent._match_exclude_pattern("server.key") is True
        assert agent._match_exclude_pattern("ssl/cert.pem") is True
        assert agent._match_exclude_pattern("keyfile.txt") is False

    def test_match_nested_path(self, temp_git_repo: Path):
        """测试嵌套路径匹配"""
        config = CommitterConfig(
            working_directory=str(temp_git_repo),
            exclude_files=["*.secret"],
        )
        agent = CommitterAgent(config)

        assert agent._match_exclude_pattern("config/db.secret") is True
        assert agent._match_exclude_pattern("deep/nested/path/api.secret") is True


class TestCommitResult:
    """测试 CommitResult 数据类"""

    def test_default_values(self):
        """测试默认值"""
        result = CommitResult(success=True)

        assert result.success is True
        assert result.commit_hash == ""
        assert result.message == ""
        assert result.files_changed == []
        assert result.pushed is False
        assert result.error == ""

    def test_full_result(self):
        """测试完整结果"""
        result = CommitResult(
            success=True,
            commit_hash="abc123",
            message="feat: new feature",
            files_changed=["file1.py", "file2.py"],
            pushed=True,
        )

        assert result.success is True
        assert result.commit_hash == "abc123"
        assert result.message == "feat: new feature"
        assert result.files_changed == ["file1.py", "file2.py"]
        assert result.pushed is True

    def test_error_result(self):
        """测试错误结果"""
        result = CommitResult(
            success=False,
            error="提交失败: 权限不足",
        )

        assert result.success is False
        assert result.error == "提交失败: 权限不足"


class TestAgentReset:
    """测试 Agent 重置功能"""

    @pytest.mark.asyncio
    async def test_committer_reset(self, committer: CommitterAgent):
        """测试 CommitterAgent 重置"""
        # 设置状态
        committer.set_context("test_key", "test_value")
        committer.update_status(AgentStatus.RUNNING)

        # 重置
        await committer.reset()

        assert committer.status == AgentStatus.IDLE
        assert committer.get_context("test_key") is None


class TestCommitIteration:
    """测试迭代提交流程 (execute 方法)"""

    @pytest.mark.asyncio
    async def test_commit_iteration_success(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试正常迭代提交"""
        # 创建新文件
        (temp_git_repo / "iteration.py").write_text("# iteration test\n")

        # Mock cursor_client 生成提交信息
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "feat: 添加迭代功能"

        with patch.object(
            committer.cursor_client, "execute", new=AsyncMock(return_value=mock_result)
        ):
            result = await committer.execute("提交迭代更新")

        assert result["success"] is True
        assert result["commit_hash"] != ""
        assert "iteration.py" in result["files_changed"]
        assert result["pushed"] is False

    @pytest.mark.asyncio
    async def test_commit_iteration_with_auto_push(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试带自动推送的迭代提交"""
        # 创建新文件
        (temp_git_repo / "push_test.py").write_text("# push test\n")

        # Mock cursor_client 生成提交信息
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "feat: 添加推送测试"

        # Mock push 方法（因为没有远程仓库）
        mock_push_result = CommitResult(
            success=True,
            commit_hash="abc123",
            pushed=True,
        )

        with patch.object(
            committer.cursor_client, "execute", new=AsyncMock(return_value=mock_result)
        ), patch.object(
            committer, "push", return_value=mock_push_result
        ):
            result = await committer.execute(
                "提交并推送",
                context={"auto_push": True}
            )

        assert result["success"] is True
        assert result["pushed"] is True

    @pytest.mark.asyncio
    async def test_commit_iteration_no_changes(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试无变更时的处理"""
        # 不创建任何新文件，仓库是干净的
        result = await committer.execute("尝试提交空变更")

        assert result["success"] is False
        assert "没有需要提交的变更" in result["error"]


class TestGetCommitSummary:
    """测试 get_commit_summary() 方法"""

    def test_get_commit_summary_empty(
        self, committer: CommitterAgent
    ):
        """测试无提交时的摘要"""
        summary = committer.get_commit_summary()

        assert summary["total_commits"] == 0
        assert summary["successful_commits"] == 0
        assert summary["failed_commits"] == 0
        assert summary["pushed_commits"] == 0
        assert summary["commit_hashes"] == []
        assert summary["files_changed"] == []

    def test_get_commit_summary_after_commits(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试多次提交后的统计"""
        # 执行多次提交
        (temp_git_repo / "file1.py").write_text("# file 1\n")
        committer.commit("feat: 添加 file1")

        (temp_git_repo / "file2.py").write_text("# file 2\n")
        committer.commit("feat: 添加 file2")

        (temp_git_repo / "file3.py").write_text("# file 3\n")
        committer.commit("feat: 添加 file3")

        summary = committer.get_commit_summary()

        assert summary["total_commits"] == 3
        assert summary["successful_commits"] == 3
        assert summary["failed_commits"] == 0
        assert len(summary["commit_hashes"]) == 3
        # 验证每个提交的 hash 都不同
        assert len(set(summary["commit_hashes"])) == 3
        # 验证变更文件
        assert "file1.py" in summary["files_changed"]
        assert "file2.py" in summary["files_changed"]
        assert "file3.py" in summary["files_changed"]

    def test_get_commit_summary_with_failures(
        self, committer: CommitterAgent, temp_git_repo: Path
    ):
        """测试包含失败提交的统计"""
        # 成功提交
        (temp_git_repo / "success.py").write_text("# success\n")
        committer.commit("feat: 成功提交")

        # 尝试无变更提交（会失败）
        result = committer.commit("feat: 空提交")
        assert result.success is False

        # 再次成功提交
        (temp_git_repo / "success2.py").write_text("# success 2\n")
        committer.commit("feat: 再次成功")

        summary = committer.get_commit_summary()

        assert summary["total_commits"] == 3
        assert summary["successful_commits"] == 2
        assert summary["failed_commits"] == 1
        assert len(summary["commit_hashes"]) == 2
        # 验证变更文件（失败的提交不应有文件在 commit_hashes 中）
        assert "success.py" in summary["files_changed"]
        assert "success2.py" in summary["files_changed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
