"""提交者 Agent

负责 Git 操作：检查状态、生成提交信息、提交、推送、回退
"""

import fnmatch
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from core.base import AgentConfig, AgentRole, AgentStatus, BaseAgent
from cursor.client import CursorAgentClient, CursorAgentConfig


class CommitterConfig(BaseModel):
    """提交者配置"""

    name: str = "committer"
    working_directory: str = "."
    auto_push: bool = False  # 是否自动推送
    commit_message_style: str = "conventional"  # 提交信息风格: conventional, simple, detailed
    include_files: list[str] = Field(default_factory=list)  # 仅提交指定文件，空则提交所有
    exclude_files: list[str] = Field(default_factory=lambda: [".env", "*.key", "*.pem", "*.secret"])  # 排除的文件模式
    cursor_config: CursorAgentConfig = Field(default_factory=CursorAgentConfig)


@dataclass
class CommitResult:
    """提交结果"""

    success: bool
    commit_hash: str = ""
    message: str = ""
    files_changed: list[str] = field(default_factory=list)
    pushed: bool = False
    error: str = ""


class CommitterAgent(BaseAgent):
    """提交者 Agent

    职责:
    1. 检查 Git 状态
    2. 根据 diff 生成提交信息
    3. 执行 git add + commit
    4. 可选推送到远程
    5. 支持回退操作
    """

    # 提交信息生成系统提示
    COMMIT_MESSAGE_PROMPT = """你是一个 Git 提交信息生成器。根据以下 diff 内容生成规范的提交信息。

## 风格要求: {style}

### conventional 风格
格式: <type>(<scope>): <description>

类型包括:
- feat: 新功能
- fix: 修复 bug
- docs: 文档变更
- style: 代码格式（不影响功能）
- refactor: 重构
- test: 测试相关
- chore: 构建/工具变更

### simple 风格
一句话简洁描述变更内容

### detailed 风格
第一行: 简短摘要（50字符以内）
空行
正文: 详细描述变更内容、原因和影响

## Diff 内容
```
{diff}
```

## 变更文件
{files}

请直接输出提交信息，不要包含任何解释或额外文本。使用中文描述。"""

    def __init__(self, config: CommitterConfig):
        agent_config = AgentConfig(
            role=AgentRole.WORKER,  # 复用 WORKER 角色
            name=config.name,
            working_directory=config.working_directory,
        )
        super().__init__(agent_config)
        self.committer_config = config
        self._commit_history: list[dict] = []  # 提交历史记录
        self._apply_stream_config(self.committer_config.cursor_config)
        self.cursor_client = CursorAgentClient(config.cursor_config)

    def _apply_stream_config(self, cursor_config: CursorAgentConfig) -> None:
        """注入流式日志配置与 Agent 标识"""
        cursor_config.stream_agent_id = self.id
        cursor_config.stream_agent_role = self.role.value
        cursor_config.stream_agent_name = self.name
        if cursor_config.stream_events_enabled:
            cursor_config.output_format = "stream-json"
            cursor_config.stream_partial_output = True

    def _run_git_command(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """执行 Git 命令

        Args:
            args: Git 命令参数列表
            check: 是否检查返回码

        Returns:
            命令执行结果
        """
        cmd = ["git"] + args
        logger.debug(f"[{self.id}] 执行: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=self.committer_config.working_directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            logger.error(f"[{self.id}] Git 命令失败: {result.stderr}")
        return result

    def _match_exclude_pattern(self, file_path: str) -> bool:
        """检查文件是否匹配排除模式

        Args:
            file_path: 文件路径

        Returns:
            是否应被排除
        """
        for pattern in self.committer_config.exclude_files:
            if fnmatch.fnmatch(file_path, pattern):
                return True
            # 也检查文件名
            if fnmatch.fnmatch(file_path.split("/")[-1], pattern):
                return True
        return False

    def check_status(self) -> dict[str, Any]:
        """检查 Git 状态

        Returns:
            状态信息，包含:
            - is_repo: 是否为 Git 仓库
            - branch: 当前分支
            - staged: 已暂存的文件列表
            - modified: 已修改但未暂存的文件列表
            - untracked: 未跟踪的文件列表
            - has_changes: 是否有变更
        """
        # 检查是否为 Git 仓库
        result = self._run_git_command(["rev-parse", "--is-inside-work-tree"], check=False)
        if result.returncode != 0:
            return {
                "is_repo": False,
                "error": "当前目录不是 Git 仓库",
            }

        # 获取当前分支
        branch_result = self._run_git_command(["branch", "--show-current"])
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

        # 获取状态
        status_result = self._run_git_command(["status", "--porcelain"])

        staged = []
        modified = []
        untracked = []

        # 使用 splitlines() 保留每行开头的状态码空格
        for line in status_result.stdout.splitlines():
            if not line or len(line) < 3:
                continue
            status_code = line[:2]
            file_path = line[3:]

            # 解析状态码
            if status_code[0] in "MADRCT":  # 已暂存
                staged.append(file_path)
            if status_code[1] in "MADRCT":  # 已修改但未暂存
                modified.append(file_path)
            if status_code == "??":  # 未跟踪
                untracked.append(file_path)

        return {
            "is_repo": True,
            "branch": branch,
            "staged": staged,
            "modified": modified,
            "untracked": untracked,
            "has_changes": bool(staged or modified or untracked),
        }

    def _get_diff(self, staged_only: bool = False) -> str:
        """获取 diff 内容

        Args:
            staged_only: 是否只获取已暂存的 diff

        Returns:
            diff 内容
        """
        args = ["diff"]
        if staged_only:
            args.append("--staged")
        args.append("--no-color")

        result = self._run_git_command(args)
        return result.stdout if result.returncode == 0 else ""

    async def generate_commit_message(
        self,
        diff: Optional[str] = None,
        files: Optional[list[str]] = None,
    ) -> str:
        """根据 diff 生成提交信息

        Args:
            diff: diff 内容，如为空则自动获取
            files: 变更文件列表

        Returns:
            生成的提交信息
        """
        self.update_status(AgentStatus.RUNNING)

        try:
            # 获取 diff
            if diff is None:
                diff = self._get_diff(staged_only=True)
                if not diff:
                    diff = self._get_diff(staged_only=False)

            if not diff:
                logger.warning(f"[{self.id}] 没有可用的 diff 内容")
                return "chore: 更新" if self.committer_config.commit_message_style == "conventional" else "更新"

            # 获取文件列表
            if files is None:
                status = self.check_status()
                files = status.get("staged", []) + status.get("modified", [])

            # 构建 prompt
            prompt = self.COMMIT_MESSAGE_PROMPT.format(
                style=self.committer_config.commit_message_style,
                diff=diff[:5000],  # 限制 diff 长度
                files="\n".join(f"- {f}" for f in files[:20]),  # 限制文件数量
            )

            # 调用 Cursor Agent 生成提交信息
            result = await self.cursor_client.execute(
                instruction=prompt,
                working_directory=self.committer_config.working_directory,
            )

            if result.success and result.output:
                # 清理输出，去除可能的代码块标记
                message = result.output.strip()
                message = re.sub(r"^```\w*\n?", "", message)
                message = re.sub(r"\n?```$", "", message)
                return message.strip()
            else:
                logger.warning(f"[{self.id}] 生成提交信息失败，使用默认信息")
                return "chore: 更新" if self.committer_config.commit_message_style == "conventional" else "更新"

        except Exception as e:
            logger.exception(f"[{self.id}] 生成提交信息异常: {e}")
            return "chore: 更新" if self.committer_config.commit_message_style == "conventional" else "更新"
        finally:
            self.update_status(AgentStatus.IDLE)

    def _filter_files(self, files: list[str]) -> list[str]:
        """过滤文件列表

        Args:
            files: 原始文件列表

        Returns:
            过滤后的文件列表
        """
        filtered = []
        for f in files:
            # 排除匹配排除模式的文件
            if self._match_exclude_pattern(f):
                logger.info(f"[{self.id}] 排除文件: {f}")
                continue

            # 如果指定了 include_files，只包含匹配的文件
            if self.committer_config.include_files:
                included = False
                for pattern in self.committer_config.include_files:
                    if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(f.split("/")[-1], pattern):
                        included = True
                        break
                if not included:
                    continue

            filtered.append(f)

        return filtered

    def commit(self, message: str, files: Optional[list[str]] = None) -> CommitResult:
        """执行 git add + commit

        Args:
            message: 提交信息
            files: 要提交的文件列表，为空则提交所有变更

        Returns:
            提交结果
        """
        self.update_status(AgentStatus.RUNNING)

        try:
            # 获取当前状态
            status = self.check_status()
            if not status.get("is_repo"):
                return CommitResult(
                    success=False,
                    error=status.get("error", "不是 Git 仓库"),
                )

            if not status.get("has_changes"):
                # 记录失败的提交到历史
                self._commit_history.append(
                    {
                        "commit_hash": "",
                        "message": message,
                        "files_changed": [],
                        "success": False,
                        "pushed": False,
                        "error": "没有需要提交的变更",
                    }
                )
                return CommitResult(
                    success=False,
                    error="没有需要提交的变更",
                )

            # 确定要添加的文件
            if files:
                files_to_add = self._filter_files(files)
            else:
                # 添加所有变更的文件
                all_files = status.get("modified", []) + status.get("untracked", [])
                files_to_add = self._filter_files(all_files)

            if not files_to_add:
                return CommitResult(
                    success=False,
                    error="没有可提交的文件（可能都被排除了）",
                )

            # 执行 git add
            for f in files_to_add:
                result = self._run_git_command(["add", f], check=False)
                if result.returncode != 0:
                    logger.warning(f"[{self.id}] 添加文件失败: {f} - {result.stderr}")

            # 执行 git commit
            commit_result = self._run_git_command(["commit", "-m", message], check=False)

            if commit_result.returncode != 0:
                # 记录失败的提交
                self._commit_history.append(
                    {
                        "commit_hash": "",
                        "message": message,
                        "files_changed": files_to_add,
                        "success": False,
                        "pushed": False,
                        "error": commit_result.stderr.strip(),
                    }
                )
                return CommitResult(
                    success=False,
                    error=commit_result.stderr.strip(),
                    files_changed=files_to_add,
                )

            # 获取 commit hash
            hash_result = self._run_git_command(["rev-parse", "HEAD"])
            commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else ""

            logger.info(f"[{self.id}] 提交成功: {commit_hash[:8]} - {message[:50]}")

            # 记录提交历史
            self._commit_history.append(
                {
                    "commit_hash": commit_hash,
                    "message": message,
                    "files_changed": files_to_add,
                    "success": True,
                    "pushed": False,
                }
            )

            return CommitResult(
                success=True,
                commit_hash=commit_hash,
                message=message,
                files_changed=files_to_add,
            )

        except Exception as e:
            logger.exception(f"[{self.id}] 提交异常: {e}")
            return CommitResult(
                success=False,
                error=str(e),
            )
        finally:
            self.update_status(AgentStatus.IDLE)

    def push(self, remote: str = "origin", branch: Optional[str] = None) -> CommitResult:
        """执行 git push

        Args:
            remote: 远程仓库名称
            branch: 分支名称，为空则使用当前分支

        Returns:
            推送结果
        """
        self.update_status(AgentStatus.RUNNING)

        try:
            # 获取当前分支
            if branch is None:
                status = self.check_status()
                branch = status.get("branch", "main")

            # 执行 push
            result = self._run_git_command(["push", "-u", remote, branch], check=False)

            if result.returncode != 0:
                return CommitResult(
                    success=False,
                    error=result.stderr.strip(),
                )

            # 获取最新 commit hash
            hash_result = self._run_git_command(["rev-parse", "HEAD"])
            commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else ""

            logger.info(f"[{self.id}] 推送成功: {remote}/{branch}")

            # 更新历史中对应 commit 的 pushed 状态
            for record in self._commit_history:
                if record.get("commit_hash") == commit_hash:
                    record["pushed"] = True
                    break

            return CommitResult(
                success=True,
                commit_hash=commit_hash,
                pushed=True,
            )

        except Exception as e:
            logger.exception(f"[{self.id}] 推送异常: {e}")
            return CommitResult(
                success=False,
                error=str(e),
            )
        finally:
            self.update_status(AgentStatus.IDLE)

    def rollback(
        self,
        mode: str = "soft",
        commit: str = "HEAD~1",
        files: Optional[list[str]] = None,
    ) -> CommitResult:
        """执行回退操作

        Args:
            mode: 回退模式
                - soft: git reset --soft（保留工作区和暂存区）
                - mixed: git reset --mixed（保留工作区，清空暂存区）
                - hard: git reset --hard（清空工作区和暂存区）
                - checkout: git checkout（还原指定文件）
            commit: 回退目标（默认 HEAD~1）
            files: 要还原的文件列表（仅 checkout 模式使用）

        Returns:
            回退结果
        """
        self.update_status(AgentStatus.RUNNING)

        try:
            if mode == "checkout" and files:
                # 还原指定文件
                for f in files:
                    result = self._run_git_command(["checkout", "HEAD", "--", f], check=False)
                    if result.returncode != 0:
                        return CommitResult(
                            success=False,
                            error=f"还原文件失败: {f} - {result.stderr}",
                        )

                logger.info(f"[{self.id}] 还原文件成功: {files}")
                return CommitResult(
                    success=True,
                    message=f"已还原 {len(files)} 个文件",
                    files_changed=files,
                )

            elif mode in ("soft", "mixed", "hard"):
                # 执行 reset
                result = self._run_git_command(["reset", f"--{mode}", commit], check=False)

                if result.returncode != 0:
                    return CommitResult(
                        success=False,
                        error=result.stderr.strip(),
                    )

                # 获取当前 HEAD
                hash_result = self._run_git_command(["rev-parse", "HEAD"])
                commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else ""

                logger.info(f"[{self.id}] 回退成功 ({mode}): {commit_hash[:8]}")

                return CommitResult(
                    success=True,
                    commit_hash=commit_hash,
                    message=f"已回退到 {commit} (模式: {mode})",
                )

            else:
                return CommitResult(
                    success=False,
                    error=f"不支持的回退模式: {mode}",
                )

        except Exception as e:
            logger.exception(f"[{self.id}] 回退异常: {e}")
            return CommitResult(
                success=False,
                error=str(e),
            )
        finally:
            self.update_status(AgentStatus.IDLE)

    async def execute(self, instruction: str, context: Optional[dict] = None) -> dict[str, Any]:
        """执行完整提交流程

        Args:
            instruction: 提交说明或指令
            context: 上下文信息，可包含:
                - files: 要提交的文件列表
                - message: 自定义提交信息
                - auto_push: 是否推送（覆盖配置）

        Returns:
            执行结果
        """
        self.update_status(AgentStatus.RUNNING)
        logger.info(f"[{self.id}] 开始提交流程: {instruction[:100]}...")

        try:
            context = context or {}

            # 1. 检查状态
            status = self.check_status()
            if not status.get("is_repo"):
                return {
                    "success": False,
                    "error": status.get("error", "不是 Git 仓库"),
                }

            if not status.get("has_changes"):
                return {
                    "success": False,
                    "error": "没有需要提交的变更",
                }

            # 2. 确定要提交的文件
            files = context.get("files") or None

            # 3. 生成或使用提供的提交信息
            message = context.get("message")
            if not message:
                # 根据 diff 生成提交信息
                diff = self._get_diff()
                all_files = status.get("staged", []) + status.get("modified", []) + status.get("untracked", [])
                message = await self.generate_commit_message(diff, all_files)

            # 4. 执行提交
            commit_result = self.commit(message, files)

            if not commit_result.success:
                self.update_status(AgentStatus.FAILED)
                return {
                    "success": False,
                    "error": commit_result.error,
                    "files_changed": commit_result.files_changed,
                }

            # 5. 可选推送
            auto_push = context.get("auto_push", self.committer_config.auto_push)
            if auto_push:
                push_result = self.push()
                if not push_result.success:
                    logger.warning(f"[{self.id}] 推送失败: {push_result.error}")
                    # 提交成功但推送失败，仍返回成功但标记推送失败
                    self.update_status(AgentStatus.COMPLETED)
                    return {
                        "success": True,
                        "commit_hash": commit_result.commit_hash,
                        "message": commit_result.message,
                        "files_changed": commit_result.files_changed,
                        "pushed": False,
                        "push_error": push_result.error,
                    }
                commit_result.pushed = True

            self.update_status(AgentStatus.COMPLETED)
            logger.info(f"[{self.id}] 提交流程完成: {commit_result.commit_hash[:8]}")

            return {
                "success": True,
                "commit_hash": commit_result.commit_hash,
                "message": commit_result.message,
                "files_changed": commit_result.files_changed,
                "pushed": commit_result.pushed,
            }

        except Exception as e:
            logger.exception(f"[{self.id}] 提交流程异常: {e}")
            self.update_status(AgentStatus.FAILED)
            return {
                "success": False,
                "error": str(e),
            }

    async def commit_iteration(
        self,
        iteration_id: int,
        tasks_completed: list[dict],
        review_decision: str,
        auto_push: bool = False,
    ) -> dict[str, Any]:
        """提交迭代结果

        根据迭代信息生成规范的提交消息并执行提交。

        Args:
            iteration_id: 迭代 ID
            tasks_completed: 已完成的任务列表，每个任务包含 task_id, description 等字段
            review_decision: 评审决策（如 'approved', 'needs_revision' 等）
            auto_push: 是否自动推送到远程

        Returns:
            提交结果字典，包含:
            - success: 是否成功
            - commit_hash: 提交哈希
            - message: 提交消息
            - files_changed: 变更的文件列表
            - pushed: 是否已推送
            - iteration_id: 迭代 ID
            - tasks_count: 完成的任务数量
        """
        self.update_status(AgentStatus.RUNNING)
        logger.info(f"[{self.id}] 开始提交迭代 {iteration_id}，共 {len(tasks_completed)} 个任务")

        try:
            # 1. 检查 Git 状态
            status = self.check_status()
            if not status.get("is_repo"):
                return {
                    "success": False,
                    "error": status.get("error", "不是 Git 仓库"),
                    "iteration_id": iteration_id,
                }

            if not status.get("has_changes"):
                # 没有变更，但仍记录到历史
                history_entry = {
                    "iteration_id": iteration_id,
                    "tasks_completed": tasks_completed,
                    "review_decision": review_decision,
                    "success": False,
                    "error": "没有需要提交的变更",
                    "commit_hash": "",
                    "message": "",
                    "files_changed": [],
                    "pushed": False,
                }
                self._commit_history.append(history_entry)
                return {
                    "success": False,
                    "error": "没有需要提交的变更",
                    "iteration_id": iteration_id,
                }

            # 2. 构建上下文信息
            task_descriptions = []
            for task in tasks_completed:
                task_id = task.get("task_id", task.get("id", "unknown"))
                # description 回退策略：description -> title -> name -> 空
                description = task.get("description") or task.get("title") or task.get("name") or ""
                if description:
                    task_descriptions.append(f"  - [{task_id}] {description}")
                else:
                    task_descriptions.append(f"  - [{task_id}]")

            # 3. 生成包含迭代信息的提交消息
            tasks_count = len(tasks_completed)
            commit_message = f"chore(iter-{iteration_id}): 完成 {tasks_count} 个任务"

            # 添加任务详情到提交消息正文
            if task_descriptions:
                commit_message += "\n\n完成的任务:\n" + "\n".join(task_descriptions)

            # 添加评审决策
            commit_message += f"\n\n评审决策: {review_decision}"

            # 4. 收集变更文件
            all_files = status.get("staged", []) + status.get("modified", []) + status.get("untracked", [])
            files_to_commit = self._filter_files(all_files)

            # 5. 调用现有的 commit() 方法执行提交
            commit_result = self.commit(commit_message, files_to_commit if files_to_commit else None)

            if not commit_result.success:
                # 记录失败到历史
                history_entry = {
                    "iteration_id": iteration_id,
                    "tasks_completed": tasks_completed,
                    "review_decision": review_decision,
                    "success": False,
                    "error": commit_result.error,
                    "commit_hash": "",
                    "message": commit_message,
                    "files_changed": commit_result.files_changed,
                    "pushed": False,
                }
                self._commit_history.append(history_entry)
                self.update_status(AgentStatus.FAILED)
                return {
                    "success": False,
                    "error": commit_result.error,
                    "message": commit_message,
                    "files_changed": commit_result.files_changed,
                    "pushed": False,
                    "iteration_id": iteration_id,
                    "tasks_count": tasks_count,
                }

            # 6. 根据 auto_push 参数决定是否推送
            pushed = False
            push_error = ""
            if auto_push:
                push_result = self.push()
                if push_result.success:
                    pushed = True
                    logger.info(f"[{self.id}] 迭代 {iteration_id} 推送成功")
                else:
                    push_error = push_result.error
                    logger.warning(f"[{self.id}] 迭代 {iteration_id} 推送失败: {push_error}")

            # 7. 记录提交历史
            history_entry = {
                "iteration_id": iteration_id,
                "tasks_completed": tasks_completed,
                "review_decision": review_decision,
                "success": True,
                "commit_hash": commit_result.commit_hash,
                "message": commit_message,
                "files_changed": commit_result.files_changed,
                "pushed": pushed,
                "push_error": push_error if push_error else None,
            }
            self._commit_history.append(history_entry)

            self.update_status(AgentStatus.COMPLETED)
            logger.info(
                f"[{self.id}] 迭代 {iteration_id} 提交成功: {commit_result.commit_hash[:8]} - {tasks_count} 个任务"
            )

            # 8. 返回结果
            result = {
                "success": True,
                "commit_hash": commit_result.commit_hash,
                "message": commit_message,
                "files_changed": commit_result.files_changed,
                "pushed": pushed,
                "iteration_id": iteration_id,
                "tasks_count": tasks_count,
            }
            if push_error:
                result["push_error"] = push_error

            return result

        except Exception as e:
            logger.exception(f"[{self.id}] 迭代 {iteration_id} 提交异常: {e}")
            # 记录异常到历史
            history_entry = {
                "iteration_id": iteration_id,
                "tasks_completed": tasks_completed,
                "review_decision": review_decision,
                "success": False,
                "error": str(e),
                "commit_hash": "",
                "message": "",
                "files_changed": [],
                "pushed": False,
            }
            self._commit_history.append(history_entry)
            self.update_status(AgentStatus.FAILED)
            return {
                "success": False,
                "error": str(e),
                "iteration_id": iteration_id,
            }

    def get_commit_summary(self) -> dict[str, Any]:
        """获取提交统计摘要

        Returns:
            包含以下字段的字典:
            - total_commits: 总提交次数
            - successful_commits: 成功提交次数
            - failed_commits: 失败提交次数
            - pushed_commits: 已推送次数
            - commit_hashes: 提交哈希列表
            - files_changed: 所有变更文件去重列表
            - iterations: 迭代提交历史列表（供 commit_iteration 使用）
            - last_commit: 最近一次提交信息
        """
        successful = [r for r in self._commit_history if r.get("success")]
        failed = [r for r in self._commit_history if not r.get("success")]
        pushed = [r for r in self._commit_history if r.get("pushed")]

        # 收集所有提交哈希（仅成功的）
        commit_hashes = [r["commit_hash"] for r in successful if r.get("commit_hash")]

        # 收集所有变更文件并去重
        all_files: set[str] = set()
        for record in self._commit_history:
            files = record.get("files_changed", [])
            all_files.update(files)

        # 收集迭代提交记录（仅包含 iteration_id 的记录）
        iteration_commits = [r for r in self._commit_history if r.get("iteration_id") is not None]

        return {
            "total_commits": len(self._commit_history),
            "successful_commits": len(successful),
            "failed_commits": len(failed),
            "pushed_commits": len(pushed),
            "commit_hashes": commit_hashes,
            "files_changed": list(all_files),
            "iterations": iteration_commits,
            "last_commit": self._commit_history[-1] if self._commit_history else None,
        }

    async def reset(self) -> None:
        """重置提交者状态"""
        self.update_status(AgentStatus.IDLE)
        self.clear_context()
        logger.debug(f"[{self.id}] 状态已重置")
