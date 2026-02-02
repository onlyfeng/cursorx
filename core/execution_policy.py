"""执行策略模块 - Cloud 与 CLI 执行模式决策的权威实现

本模块提供执行模式决策、Cloud 错误分类、回退策略的统一接口。
所有涉及执行模式选择的逻辑都应通过本模块的函数实现，以确保一致性。

================================================================================
参数级别优先级表 (Parameter Priority Matrix)
================================================================================

本表定义各入口脚本（run.py、scripts/run_iterate.py）中配置参数的解析优先级，
确保 `build_execution_decision` 和 `resolve_orchestrator_settings` 的调用一致。

【优先级层级】（从高到低）

+-------+----------------------+------------------------------------------+
| 优先级 | 来源                 | 说明                                     |
+=======+======================+==========================================+
| 1     | CLI 显式参数         | --execution-mode, --workers, --orchestrator 等 |
+-------+----------------------+------------------------------------------+
| 2     | 自然语言分析结果     | TaskAnalyzer 从任务描述中提取的参数        |
|       | (仅 run.py)          |                                          |
+-------+----------------------+------------------------------------------+
| 3     | & 前缀路由           | 当 has_ampersand_prefix=True 且满足路由条件时 |
+-------+----------------------+------------------------------------------+
| 4     | 环境变量             | CURSOR_API_KEY, CURSOR_CLOUD_API_KEY      |
+-------+----------------------+------------------------------------------+
| 5     | config.yaml          | cloud_agent.execution_mode, system.* 等    |
+-------+----------------------+------------------------------------------+
| 6     | DEFAULT_* 常量       | core/config.py 中的默认值常量             |
+-------+----------------------+------------------------------------------+

【各参数优先级详表】

+----------------------+--------+--------+--------+--------+--------+--------+
| 参数                 | CLI(1) | NL(2)  | &(3)   | ENV(4) | YAML(5)| DEF(6) |
+======================+========+========+========+========+========+========+
| execution_mode       | ✓      | ✓      | ✓(*)   | -      | ✓      | auto   |
+----------------------+--------+--------+--------+--------+--------+--------+
| orchestrator         | ✓      | ✓      | ✓(**)  | -      | -      | mp(**) |
+----------------------+--------+--------+--------+--------+--------+--------+
| workers              | ✓      | ✓      | -      | -      | ✓      | 3      |
+----------------------+--------+--------+--------+--------+--------+--------+
| max_iterations       | ✓      | ✓      | -      | -      | ✓      | 10     |
+----------------------+--------+--------+--------+--------+--------+--------+
| cloud_timeout        | ✓      | -      | -      | -      | ✓      | 300    |
+----------------------+--------+--------+--------+--------+--------+--------+
| cloud_auth_timeout   | ✓      | -      | -      | -      | ✓      | 30     |
+----------------------+--------+--------+--------+--------+--------+--------+
| api_key              | ✓      | -      | -      | ✓      | ✓      | None   |
+----------------------+--------+--------+--------+--------+--------+--------+
| dry_run              | ✓      | ✓      | -      | -      | -      | False  |
+----------------------+--------+--------+--------+--------+--------+--------+
| skip_online          | ✓      | ✓      | -      | -      | -      | False  |
+----------------------+--------+--------+--------+--------+--------+--------+
| auto_commit          | ✓      | ✓      | -      | -      | -      | False  |
+----------------------+--------+--------+--------+--------+--------+--------+

(*) execution_mode: 默认 auto（来自 config.yaml）；& 前缀成功路由时等效于 cloud
(**) orchestrator: 默认 mp，但 requested_mode=auto/cloud 或 & 前缀成功路由时**强制 basic**
    ⚠ 如需使用 MP 编排器，请显式指定 --execution-mode cli --orchestrator mp

================================================================================
系统级默认 vs 函数级默认
================================================================================

【系统级默认】（入口脚本层面）

    入口脚本（run.py、scripts/run_iterate.py）在无显式 --execution-mode 时：
    1. 读取 config.yaml 中的 cloud_agent.execution_mode（默认 auto）
    2. 将此值作为 requested_mode 传递给 build_execution_decision

    因此，**整体默认表现为 auto 模式**：
    - 有 API Key：使用 Cloud 执行，编排器强制 basic
    - 无 API Key：回退到 CLI 执行，**编排器仍为 basic**（基于 requested_mode=auto）

    示例（无显式 --execution-mode）：
    | config.yaml default | has_api_key | effective_mode | orchestrator |
    |---------------------|-------------|----------------|--------------|
    | auto                | True        | auto/cloud     | basic        |
    | auto                | False       | cli (回退)     | basic        |

    ⚠ 如需使用 MP 编排器，必须显式指定 --execution-mode cli

【函数级默认】（build_execution_decision / resolve_effective_execution_mode 层面）

    当 requested_mode=None 且无 & 前缀时，函数内部默认返回 effective_mode=cli。
    但实际运行时，入口脚本通常会传入 config.yaml 的默认值（auto），
    所以函数级默认（cli）很少直接生效。

    函数级默认触发条件：
    - requested_mode=None（入口脚本未传递任何值，通常不会发生）
    - has_ampersand_prefix=False（无 & 前缀）
    - 此时返回: effective_mode=cli, orchestrator=mp

    示例（requested_mode=None，函数级默认）：
    | requested_mode | & 前缀 | effective_mode | orchestrator |
    |----------------|--------|----------------|--------------|
    | None           | False  | cli            | mp           |
    | None           | True   | 取决于路由结果 | 取决于路由   |

【优先级总结】

    1. CLI 显式参数（--execution-mode）         → 最高优先级
    2. 自然语言分析 / & 前缀路由                → 次高优先级
    3. config.yaml (cloud_agent.execution_mode) → 系统级默认（通常为 auto）
    4. 函数内部硬编码默认                       → 函数级默认（cli）

【关键规则】

规则 1: requested_mode vs effective_mode
    - requested_mode: 用户通过 CLI 或 config.yaml 请求的执行模式
    - effective_mode: 经过路由决策后实际使用的执行模式
    - 编排器选择基于 **requested_mode**（不是 effective_mode）

    示例：
    | 场景 | requested_mode | has_api_key | effective_mode | orchestrator |
    |------|----------------|-------------|----------------|--------------|
    | A    | auto           | False       | cli (回退)     | basic        |
    | B    | cli            | True        | cli            | mp           |

    场景 A 中，虽然 effective_mode 回退到 cli，但因为 requested_mode=auto，
    编排器仍强制使用 basic。

================================================================================
警告与日志策略决策 (Warning & Logging Policy Decisions)
================================================================================

【决策 1】requested_mode=auto 且无 API Key 时的警告策略

    **问题**：如果 config.yaml 默认 execution_mode=auto，每次运行都会警告

    **结论**：区分"显式配置"和"隐式默认"两种情况

    | 情况               | 日志级别  | 说明                                        |
    |--------------------|-----------|---------------------------------------------|
    | CLI 显式参数       | WARNING   | 用户显式 --execution-mode auto/cloud        |
    | config.yaml 显式   | WARNING   | config.yaml 中显式设置 execution_mode: auto |
    | 隐式默认           | INFO      | 使用配置文件默认值，仅信息提示              |

    **理由**：
    - 避免"每次都警告"导致的警告疲劳
    - 保留用户显式意图时的明确反馈
    - INFO 级别仍可通过 --verbose 查看

【决策 2】编排器选择是否基于 requested_mode（结论：是，保持现有设计）

    **问题**：是否对"默认 auto 且回退到 cli"的场景放开 MP 编排器？

    **结论**：**否**，保持现有设计——编排器选择严格基于 requested_mode

    **理由**：
    a. 语义一致性：用户请求 auto 意味着期望云端执行，即使回退也应保持 basic
    b. 避免混淆：requested_mode=auto 的语义是"优先云端"，不应在回退时改变编排器
    c. 可预测性：用户可通过显式指定 `--execution-mode cli` 获得 MP 编排器

    **用户指引**：如需使用 MP 编排器，请显式指定 --execution-mode cli

【决策 3】scripts/run_mp.py 的日志级别策略

    **问题**：当 config 默认变为 auto 时，当前逻辑会每次 WARNING 并强制回退

    **结论**：将日志级别降级为 INFO（因当前无法区分显式/隐式来源）

    **未来优化**：
    - 传递"配置来源"信息（CLI/config.yaml/default）
    - 根据来源动态决定日志级别
    - 仅在用户显式配置时使用 WARNING

    **当前实现**：
    - run_mp.py 使用 INFO 级别提示 Cloud/Auto 模式不兼容
    - 用户需使用 --verbose 查看详细信息
    - WARNING 仅用于真正需要用户注意的情况（如推送失败）

规则 2: & 前缀路由优先于 config.yaml 的 execution_mode
    - 当用户输入 "& 任务描述" 且满足路由条件时，& 前缀的优先级高于 config.yaml
    - 但低于 CLI 显式 --execution-mode 参数

    示例：
    | 场景 | CLI execution_mode | config.yaml | 输入        | effective |
    |------|--------------------|-------------|-------------|-----------|
    | A    | None               | cli         | "& 任务"    | cloud     |
    | B    | cli                | cloud       | "& 任务"    | cli       |
    | C    | None               | auto        | "任务"      | auto      |

规则 3: prefix_routed 传递要求
    - `resolve_orchestrator_settings()` 必须接收 `prefix_routed` 参数（推荐）
    - 当 & 前缀成功触发时，传递 `prefix_routed=True`
    - `triggered_by_prefix` 作为兼容别名保留，语义等同于 `prefix_routed`
    - 这确保编排器选择逻辑的一致性

【已识别的冲突点清单】

+------+------------------+---------------------------------------------------+
| 编号 | 位置             | 问题描述                                          |
+======+==================+===================================================+
| C-1  | run.py           | _merge_options 正确传递 triggered_by_prefix       |
|      | _merge_options   | 给 resolve_orchestrator_settings                  |
|      | (已修复)         |                                                   |
+------+------------------+---------------------------------------------------+
| C-2  | scripts/         | _resolve_config_settings 调用                     |
|      | run_iterate.py   | resolve_orchestrator_settings 时需传递            |
|      | _resolve_config  | triggered_by_prefix 参数                          |
|      | _settings        |                                                   |
|      | (已修复)         | 修复: 传递 self._triggered_by_prefix              |
+------+------------------+---------------------------------------------------+
| C-3  | scripts/         | _resolve_config_settings 返回的 orchestrator      |
|      | run_iterate.py   | 可能与 _execution_decision.orchestrator 不一致    |
|      |                  | 因为一个使用 resolve_orchestrator_settings，       |
|      | (非冲突)         | 另一个使用 build_execution_decision               |
|      |                  | 实际上 _get_orchestrator_type 使用的是            |
|      |                  | _execution_decision.orchestrator，所以不冲突      |
+------+------------------+---------------------------------------------------+
| C-4  | core/config.py   | resolve_orchestrator_settings 内部使用            |
|      | resolve_         | effective_execution_mode 判断强制 basic           |
|      | orchestrator_    | 这与 AGENTS.md 描述的"基于 requested_mode"一致    |
|      | settings         | 但 overrides["execution_mode"] 本身就是           |
|      | (无冲突)         | requested_mode（来自 CLI 或 config.yaml）          |
+------+------------------+---------------------------------------------------+
| C-5  | tests/test_      | 测试矩阵中的 _compute_run_py_snapshot 函数        |
|      | iterate_...py    | 正确传递 triggered_by_prefix 给                   |
|      | (已正确)         | resolve_orchestrator_settings                     |
+------+------------------+---------------------------------------------------+

【修复记录】

C-2 修复（scripts/run_iterate.py _resolve_config_settings）:

    修复前:
        resolved = resolve_orchestrator_settings(overrides=cli_overrides)

    修复后:
        resolved = resolve_orchestrator_settings(
            overrides=cli_overrides,
            triggered_by_prefix=self._triggered_by_prefix,
        )

    说明: 虽然 _get_orchestrator_type() 实际使用 _execution_decision.orchestrator，
    而非 _resolve_config_settings() 的返回值，修复此问题可确保语义一致性，
    避免未来重构时引入 bug。

================================================================================
副作用控制策略矩阵 (Side Effect Control Strategy Matrix)
================================================================================

本系统提供三种副作用控制策略，用于在不同场景下控制系统行为：

+---------------+----------------+----------------+----------------+-----------------+
| 策略          | 网络请求       | 文件写入       | Git 操作       | 适用场景        |
+===============+================+================+================+=================+
| normal        | 允许           | 允许           | 允许           | 正常执行        |
| (默认)        |                |                |                |                 |
+---------------+----------------+----------------+----------------+-----------------+
| skip-online   | 禁止在线检查   | 允许           | 允许           | 离线环境        |
|               | 本地缓存优先   |                |                | CI/CD 加速      |
+---------------+----------------+----------------+----------------+-----------------+
| dry-run       | 允许           | 禁止           | 禁止           | 预览/调试       |
|               | (用于分析)     | (仅日志输出)   | (仅日志输出)   | 安全检查        |
+---------------+----------------+----------------+----------------+-----------------+
| minimal       | 禁止           | 禁止           | 禁止           | 最小副作用      |
|               |                |                |                | 纯分析场景      |
+---------------+----------------+----------------+----------------+-----------------+

【策略详解】

1. **normal (默认)**
   - 完整功能模式，允许所有副作用
   - 网络: 在线检查文档更新、获取 changelog、抓取网页
   - 文件: 更新知识库、保存文档、写入缓存
   - Git: 自动提交（需显式 --auto-commit）、推送

2. **skip-online (--skip-online)**
   - 跳过在线文档检查，使用本地缓存
   - 适用场景: 离线环境、CI/CD 加速、网络不稳定时
   - 副作用:
     * 禁止: 在线文档检查、changelog 更新、外部 URL 抓取
     * 允许: 本地知识库读写、Git 操作、本地缓存使用
   - 触发方式:
     * CLI: `--skip-online`
     * 自然语言: "跳过在线", "离线", "不检查更新"

3. **dry-run (--dry-run)**
   - 仅分析不执行，不修改任何文件
   - 适用场景: 预览变更、调试任务分解、安全检查
   - 副作用:
     * 禁止: 文件创建/修改/删除、Git commit/push
     * 允许: 读取文件、网络请求（用于分析）、日志输出
   - 触发方式:
     * CLI: `--dry-run`
     * 自然语言: "仅分析", "预览", "不执行"

4. **minimal (skip-online + dry-run)**
   - 最小副作用模式，跳过网络请求且不修改文件
   - 适用场景: 纯本地分析、单元测试、快速验证
   - 副作用:
     * 禁止: 所有网络请求、所有文件写入、所有 Git 操作
     * 允许: 本地文件读取、内存计算、日志输出
   - 触发方式:
     * CLI: `--skip-online --dry-run`
     * 自然语言: "跳过在线 仅分析"

【各模块副作用点】

| 模块                     | 副作用类型     | normal | skip-online | dry-run | minimal |
|--------------------------|----------------|--------|-------------|---------|---------|
| knowledge/fetcher.py     | 网络请求       | ✓      | ✗           | ✓       | ✗       |
| knowledge/storage.py     | 知识库写入     | ✓      | ✓           | ✗       | ✗       |
| agents/committer.py      | Git 提交       | ✓(显式)| ✓(显式)     | ✗       | ✗       |
| coordinator/*            | 文件修改       | ✓      | ✓           | ✗       | ✗       |
| run.py                   | 编排执行       | ✓      | ✓           | ✗       | ✗       |
| scripts/run_iterate.py   | 迭代执行       | ✓      | ✓           | ✗       | ✗       |

【CLI 参数映射】

```bash
# normal 模式（默认）
python run.py "任务描述"
python scripts/run_iterate.py "任务描述"

# skip-online 模式
python run.py --skip-online "任务描述"
python scripts/run_iterate.py --skip-online "任务描述"

# dry-run 模式
python run.py --dry-run "任务描述"
python scripts/run_iterate.py --dry-run "任务描述"

# minimal 模式
python run.py --skip-online --dry-run "任务描述"
python scripts/run_iterate.py --skip-online --dry-run "任务描述"
```

【实现契约】

各模块应遵循以下契约实现副作用控制：

1. fetcher 模块 (knowledge/fetcher.py):
   - 当 skip_online=True 时，fetch() 应直接返回缓存内容或空结果
   - 不应抛出异常，而是返回 FetchResult(success=False, ...)

2. storage 模块 (knowledge/storage.py):
   - 当 dry_run=True 时，save_document() 应只验证不写入
   - 返回 (True, "dry-run: 跳过写入") 而非实际写入

3. committer 模块 (agents/committer.py):
   - 当 dry_run=True 时，commit() 应只生成 commit message 不执行
   - 日志输出: "[dry-run] 将执行: git commit -m '...'"

4. 入口脚本 (run.py, scripts/run_iterate.py):
   - 解析 --skip-online 和 --dry-run 参数
   - 将策略传递给下游模块
   - 在 dry-run 模式下输出清晰的 "[DRY-RUN]" 标记

================================================================================
职责边界定义
================================================================================

本模块 (core/execution_policy.py) 的职责：
1. 【决策】执行模式解析、& 前缀路由判断
2. 【分类】Cloud 错误分类，返回结构化 CloudFailureInfo
3. 【构建】构建用户友好消息字符串，但 **不直接打印**
4. 【清理】sanitize prompt 以避免回退时再次触发 Cloud

本模块 **只返回结构化元数据**，打印职责由上层决定：
- cursor/executor.py: 编排层，可使用 logger 打印关键状态（DEBUG/INFO）
- 入口脚本 (run.py, scripts/run_iterate.py): 负责打印用户可见的提示消息

示例 - 谁打印 vs 谁只返回元数据：

    # ✓ execution_policy.py - 只返回元数据，不打印
    failure_info = classify_cloud_failure(exception)
    message = build_user_facing_fallback_message(...)  # 构建字符串但不输出

    # ✓ executor.py - 可打印技术性日志（DEBUG/INFO 级别）
    logger.info(f"Cloud 执行失败，冷却 {remaining}s，回退到 CLI")

    # ✓ 入口脚本 - 打印用户可见的提示消息
    # 使用 CooldownInfoFields 常量访问子字段
    if result.cooldown_info and result.cooldown_info.get(CooldownInfoFields.USER_MESSAGE):
        print(result.cooldown_info[CooldownInfoFields.USER_MESSAGE])

    # ✗ cursor/client.py - 【需要改造】避免多行用户提示
    # 当前违规行为：logger.warning(fallback_msg)  # 多行用户提示
    # 改造方案：将消息附加到返回结果中，由入口脚本决定是否打印

================================================================================
& 前缀语义：has_ampersand_prefix vs prefix_routed (triggered_by_prefix)
================================================================================

【关键区分】两个容易混淆的概念：

1. has_ampersand_prefix (检测层面)
   - 定义：原始 prompt 文本是否以 '&' 开头（语法检测）
   - 来源：is_cloud_request(prompt) 的返回值
   - 语义：纯文本层面的前缀检测，与配置/认证无关
   - 示例：
       >>> is_cloud_request("& 分析代码")
       True   # has_ampersand_prefix=True
       >>> is_cloud_request("分析代码")
       False  # has_ampersand_prefix=False

2. prefix_routed / triggered_by_prefix (路由层面)
   - 定义：& 前缀是否**成功触发** Cloud 模式（策略决策）
   - 条件：has_ampersand_prefix=True 且满足以下全部条件：
       a. cloud_enabled=True（Cloud 功能已启用）
       b. has_api_key=True（有有效 API Key）
       c. auto_detect_cloud_prefix=True（未禁用自动检测）
       d. 未显式指定 execution_mode=cli（CLI 模式忽略 & 前缀）
   - 语义：表示本次请求实际使用 Cloud 模式执行
   - 示例：
       # 场景 1: & 前缀成功路由到 Cloud
       has_ampersand_prefix=True, cloud_enabled=True, has_api_key=True
       => prefix_routed=True, effective_mode="cloud"

       # 场景 2: & 前缀存在但条件不满足
       has_ampersand_prefix=True, cloud_enabled=True, has_api_key=False
       => prefix_routed=False, effective_mode="cli"（回退）

       # 场景 3: 显式 CLI 模式忽略 & 前缀
       has_ampersand_prefix=True, requested_mode="cli"
       => prefix_routed=False, effective_mode="cli"

【命名迁移说明】

旧字段名 `triggered_by_prefix` 存在历史语义歧义，部分代码将其用于表示
has_ampersand_prefix，部分用于表示 prefix_routed。

**迁移已完成**: 全仓 `triggered_by_prefix` 现已统一为 `prefix_routed` 语义。

推荐用法：
- 检测层面使用：has_ampersand_prefix（来自 is_cloud_request）
- 决策层面使用：prefix_routed（来自 build_execution_decision）
- 兼容性：triggered_by_prefix 作为 prefix_routed 的别名保留，值始终来自 prefix_routed

【迁移检查点 - 已完成】

以下位置已完成迁移，`triggered_by_prefix` 值均来自 `prefix_routed`：

run.py:
  - analysis.options["triggered_by_prefix"]: 值来自 decision.prefix_routed ✓
  - options["triggered_by_prefix"]: 值来自 execution_decision.prefix_routed ✓

scripts/run_iterate.py:
  - self._triggered_by_prefix: 值来自 self._prefix_routed ✓

core/execution_policy.py:
  - ExecutionPolicyContext.triggered_by_prefix: 返回 self.prefix_routed ✓
  - ExecutionDecision.triggered_by_prefix: 返回 self.prefix_routed ✓
  - AmpersandPrefixInfo.triggered_by_prefix: 返回 self.prefix_routed ✓
  - AmpersandPrefixStatus.triggered_by_prefix: 返回 self.is_routed ✓

core/config.py:
  - resolve_orchestrator_settings: 新增 prefix_routed 参数（推荐用法），
    triggered_by_prefix 作为兼容别名保留 ✓
  - build_unified_overrides: 新增 prefix_routed 参数（推荐用法），
    triggered_by_prefix 作为兼容别名保留 ✓
  - UnifiedOptions.triggered_by_prefix: 字段值来自 prefix_routed ✓

================================================================================
统一字段 Schema (Unified Field Schema)
================================================================================

本 Schema 是 core/execution_policy.py、core/config.py 和
tests/test_iterate_execution_matrix_consistency.py 三个模块的共同契约。
所有涉及执行模式决策的字段名、语义、来源应遵循此 Schema。

+---------------------------+----------+----------------------------------------------+
| 字段名                    | 类型     | 语义定义                                     |
+===========================+==========+==============================================+
| effective_mode            | str      | 有效执行模式：经过路由决策后实际使用的模式   |
|                           |          | 值域: cli/cloud/auto/plan/ask                |
|                           |          | 来源: build_execution_decision()             |
+---------------------------+----------+----------------------------------------------+
| requested_mode            | str|None | 请求执行模式：用户通过 CLI/config.yaml 请求  |
|                           |          | 的模式（可能因条件不满足而与 effective 不同）|
|                           |          | None 表示未显式指定                          |
|                           |          | 来源: CLI --execution-mode / config.yaml     |
+---------------------------+----------+----------------------------------------------+
| requested_mode_for_decision| str|None| 【测试专用】用于决策的请求模式               |
|                           |          | 语义等同于 requested_mode，但强调这是传给    |
|                           |          | build_execution_decision 的参数值            |
|                           |          | 来源: resolve_requested_mode_for_decision()  |
|                           |          | 注意: 此字段主要用于测试快照，区别于 CLI     |
|                           |          | 原始参数 cli_execution_mode                  |
+---------------------------+----------+----------------------------------------------+
| cli_execution_mode        | str|None | 【测试专用】CLI 原始 --execution-mode 参数   |
|                           |          | 与 requested_mode_for_decision 的区别:       |
|                           |          | - cli_execution_mode: 用户 CLI 输入的原始值  |
|                           |          | - requested_mode_for_decision: 经过           |
|                           |          |   resolve_requested_mode_for_decision 解析    |
|                           |          |   后传给 build_execution_decision 的值        |
+---------------------------+----------+----------------------------------------------+
| orchestrator              | str      | 编排器类型: mp（多进程）或 basic（协程）     |
|                           |          | 选择规则见下方优先级                         |
|                           |          | 来源: build_execution_decision()             |
+---------------------------+----------+----------------------------------------------+
| has_ampersand_prefix      | bool     | 【语法检测层面】原始 prompt 是否以 '&' 开头  |
|                           |          | 来源: is_cloud_request(prompt) /             |
|                           |          |       detect_ampersand_prefix()              |
|                           |          | 用途: 消息构建、日志记录、UI 显示            |
+---------------------------+----------+----------------------------------------------+
| prefix_routed             | bool     | 【策略决策层面】& 前缀是否成功触发 Cloud     |
|                           |          | 条件: has_ampersand_prefix=True 且满足:      |
|                           |          |   a. cloud_enabled=True                      |
|                           |          |   b. has_api_key=True                        |
|                           |          |   c. auto_detect_cloud_prefix=True           |
|                           |          |   d. 未显式指定 execution_mode=cli           |
|                           |          | 来源: detect_ampersand_prefix() /            |
|                           |          |       build_execution_decision()             |
|                           |          | 用途: 执行模式决策、编排器选择条件分支       |
+---------------------------+----------+----------------------------------------------+
| triggered_by_prefix       | bool     | 【兼容别名】等同于 prefix_routed             |
|                           |          | 状态: DEPRECATED - 新代码请使用 prefix_routed|
+---------------------------+----------+----------------------------------------------+
| sanitized_prompt          | str      | 清理后的 prompt（移除 & 前缀）               |
|                           |          | 来源: sanitize_prompt_for_cli_fallback()     |
+---------------------------+----------+----------------------------------------------+
| mode_reason               | str      | 执行模式决策原因（用于调试/日志）            |
+---------------------------+----------+----------------------------------------------+
| orchestrator_reason       | str      | 编排器选择原因（用于调试/日志）              |
+---------------------------+----------+----------------------------------------------+
| user_message              | str|None | 用户友好消息（仅构建不打印，由入口脚本决定） |
+---------------------------+----------+----------------------------------------------+
| auto_detect_cloud_prefix  | bool     | 是否启用 & 前缀自动检测 Cloud 路由           |
|                           |          | 优先级: CLI（如未来提供）> config.yaml       |
|                           |          |         `auto_detect_cloud_prefix` >         |
|                           |          |         `auto_detect_prefix`（兼容别名）>    |
|                           |          |         默认值 True                          |
|                           |          | 权威来源: config.yaml cloud_agent 配置块     |
|                           |          | 兼容别名: auto_detect_prefix（已废弃）       |
+---------------------------+----------+----------------------------------------------+

【字段来源映射】

| 字段名                      | 权威来源                                           |
|-----------------------------|----------------------------------------------------|
| effective_mode              | build_execution_decision() / resolve_effective_*() |
| requested_mode              | CLI --execution-mode / config.yaml                 |
| requested_mode_for_decision | resolve_requested_mode_for_decision()              |
| cli_execution_mode          | CLI --execution-mode 原始参数                      |
| orchestrator                | build_execution_decision() / resolve_orchestrator* |
| has_ampersand_prefix        | is_cloud_request(prompt) / detect_ampersand_prefix |
| prefix_routed               | detect_ampersand_prefix() / build_execution_*()    |
| triggered_by_prefix         | prefix_routed 的兼容别名（语义相同）               |
| auto_detect_cloud_prefix    | config.yaml cloud_agent.auto_detect_cloud_prefix   |
|                             | 兼容别名: auto_detect_prefix（已废弃）             |

【使用规范】

1. 内部条件分支（执行行为决策）应使用 **prefix_routed**:
       if decision.prefix_routed:
           # & 前缀成功触发 Cloud 模式
           effective_mode = "cloud"
           orchestrator = "basic"

2. 消息构建/日志记录应使用 **has_ampersand_prefix**:
       if decision.has_ampersand_prefix:
           message += "（由 & 前缀触发）"

3. to_dict() 输出应同时包含新旧字段以兼容下游:
       {
           "prefix_routed": True,           # 新字段
           "triggered_by_prefix": True,     # 兼容字段
           "has_ampersand_prefix": True,    # 新字段
       }

4. 测试快照字段应遵循此 Schema:
       DecisionSnapshot(
           effective_mode="cloud",
           requested_mode="auto",
           orchestrator="basic",
           prefix_routed=True,              # 使用新字段名
       )

【编排器选择优先级】

1. Cloud/Auto 模式或 & 前缀成功触发 -> 强制 basic（最高优先级）
2. 用户显式指定 --orchestrator basic 或 --no-mp -> basic
3. 用户显式指定 --orchestrator mp（仅在非 Cloud/Auto 模式下生效）
4. 默认 mp

================================================================================
user_message / message_level / cooldown_info 现状对照表
================================================================================

本对照表记录各模块中用户消息输出的字段、级别、去重机制，确保输出一致性和可维护性。

【1. ExecutionDecision 生成 user_message/message_level 场景矩阵】

来源：build_execution_decision() 函数

+--------------------------------------------+---------------+------------------------------------------+
| 场景                                       | message_level | user_message 内容                         |
+============================================+===============+==========================================+
| & 前缀 + cloud_enabled=False               | "info"        | "ℹ 检测到 '&' 前缀但 cloud_enabled=False" |
+--------------------------------------------+---------------+------------------------------------------+
| & 前缀 + 无 API Key                        | "warning"     | "⚠ 检测到 '&' 前缀但未配置 API Key"       |
+--------------------------------------------+---------------+------------------------------------------+
| & 前缀 + 显式 --execution-mode cli         | "info"        | "ℹ 检测到 '&' 前缀但显式指定 cli"         |
+--------------------------------------------+---------------+------------------------------------------+
| & 前缀 + auto_detect_cloud_prefix=False    | "info"        | "ℹ 检测到 '&' 前缀但 auto_detect=False"   |
+--------------------------------------------+---------------+------------------------------------------+
| CLI 显式 auto/cloud + 无 API Key           | "warning"     | "⚠ 请求 auto/cloud 模式但未配置 API Key"  |
+--------------------------------------------+---------------+------------------------------------------+
| config.yaml auto + 无 API Key              | "info"        | "ℹ 请求 auto 模式但未配置 API Key"        |
+--------------------------------------------+---------------+------------------------------------------+

规则总结:
- mode_source="cli" (用户显式) → 使用 "warning"
- mode_source="config"/None    → 使用 "info"（避免每次都警告）
- & 前缀表示用户显式意图 → 无 key 时使用 "warning"

【2. cooldown_info 构建位置与字段一致性】

构建函数：build_cooldown_info() / build_cooldown_info_from_metadata()

使用位置:
- cursor/client.py: _execute_via_cloud() 失败时调用 build_cooldown_info()
- cursor/executor.py: AutoAgentExecutor._build_cooldown_info_dict() 调用 build_cooldown_info()

统一字段结构 (cooldown_info dict):
+------------------+----------+------------------------------------------------+
| 字段             | 类型     | 说明                                           |
+==================+==========+================================================+
| kind             | str      | CloudFailureKind.value (no_key, auth, rate_limit 等) |
+------------------+----------+------------------------------------------------+
| user_message     | str      | 用户友好消息（非空，由入口脚本输出）            |
+------------------+----------+------------------------------------------------+
| retryable        | bool     | 是否可重试                                     |
+------------------+----------+------------------------------------------------+
| retry_after      | int|None | 建议重试等待秒数                               |
+------------------+----------+------------------------------------------------+
| reason           | str      | 回退原因                                       |
+------------------+----------+------------------------------------------------+
| fallback_reason  | str      | 回退原因（兼容字段，与 reason 相同）            |
+------------------+----------+------------------------------------------------+
| error_type       | str      | 错误类型（兼容字段，与 kind 相同）              |
+------------------+----------+------------------------------------------------+
| failure_kind     | str      | 失败类型（兼容字段，与 kind 相同）              |
+------------------+----------+------------------------------------------------+
| in_cooldown      | bool     | 是否处于冷却期                                 |
+------------------+----------+------------------------------------------------+
| remaining_seconds| float    | 冷却剩余秒数                                   |
+------------------+----------+------------------------------------------------+
| failure_count    | int      | 连续失败次数                                   |
+------------------+----------+------------------------------------------------+

【3. 入口脚本去重机制】

去重 key: compute_message_dedup_key(message) - 使用稳定 SHA256 哈希作为去重标识

+------------------------+---------------------------+-----------------------------+
| 模块/位置              | 去重集合                  | 使用场景                    |
+========================+===========================+=============================+
| run.py                 | TaskAnalyzer._shown_messages | user_message / cooldown_info |
+------------------------+---------------------------+-----------------------------+
| scripts/run_iterate.py | SelfIterator._shown_messages | user_message / cooldown_info |
+------------------------+---------------------------+-----------------------------+
| cursor/client.py       | _cloud_api_key_warning_shown | API Key 缺失警告（首次仅输出）|
+------------------------+---------------------------+-----------------------------+

【4. 输出位置与职责划分】

+---------------------------+----------------------------------------------------+
| 模块                      | 输出职责                                           |
+===========================+====================================================+
| core/execution_policy.py  | 只构建 user_message，不打印                        |
+---------------------------+----------------------------------------------------+
| cursor/executor.py        | 构建 cooldown_info，仅 logger.info/debug 技术日志   |
+---------------------------+----------------------------------------------------+
| cursor/client.py          | 构建 cooldown_info，仅 logger.debug 技术日志        |
+---------------------------+----------------------------------------------------+
| run.py                    | 打印 decision.user_message 和 cooldown_info.user_message |
+---------------------------+----------------------------------------------------+
| scripts/run_iterate.py    | 打印 decision.user_message 和 cooldown_info.user_message |
+---------------------------+----------------------------------------------------+

【5. run.py 输出去重位置】

1. TaskAnalyzer.analyze() - 决策消息输出:
   位置: run.py 行约 1117-1148
   去重: compute_message_dedup_key(decision.user_message) 加入 TaskAnalyzer._shown_messages
   输出: print_warning 或 print_info（根据 message_level）

2. Cloud 执行后 cooldown_info 输出:
   位置: run.py 行约 2752-2758
   去重: compute_message_dedup_key(cooldown_msg) 加入 TaskAnalyzer._shown_messages
   输出: print_warning(cooldown_msg)

【6. scripts/run_iterate.py 输出去重位置】

1. SelfIterator.__init__ 决策消息输出:
   位置: scripts/run_iterate.py 行约 4644-4651
   去重: compute_message_dedup_key(decision.user_message) 加入 SelfIterator._shown_messages
   输出: print_warning 或 print_info（根据 message_level）

2. _print_execution_result 输出:
   位置: scripts/run_iterate.py 行约 6038-6043
   去重: compute_message_dedup_key(cooldown_info[CooldownInfoFields.USER_MESSAGE]) 加入 SelfIterator._shown_messages
   输出: print_warning(cooldown_info[CooldownInfoFields.USER_MESSAGE])

【7. 防回归测试列表】

- tests/test_run.py::TestPrintWarningMockVerification::test_library_layer_no_user_message_print
- tests/test_run.py::TestCloudFallbackUserMessageDedup
- tests/test_self_iterate.py::TestSelfIterateCloudFallbackUserMessageDedup
- tests/test_cursor_client.py (验证 cooldown_info 结构一致性)
- tests/test_iterate_execution_matrix_consistency.py::TestMessageLevelSemanticMatrix

================================================================================
三类失败的精确定义
================================================================================

根据 CloudFailureKind 枚举，Cloud 失败分为三类：

【类型 1】配置/认证失败 - 需要用户行动修复，不自动重试
    - NO_KEY:        未配置 API Key
    - CLOUD_DISABLED: cloud_enabled=False
    - AUTH:          认证失败（API Key 无效、过期、403）

    冷却策略: auth_cooldown_seconds=600 (10分钟)
    恢复条件: 需要配置变化 (auth_require_config_change=True)
    示例:
        >>> info = classify_cloud_failure(AuthError("401 Unauthorized"))
        >>> info.kind == CloudFailureKind.AUTH
        >>> info.retryable == False

【类型 2】可重试失败 - 可自动冷却后重试
    - RATE_LIMIT:    速率限制（429），使用 retry_after
    - TIMEOUT:       请求超时
    - NETWORK:       网络连接错误
    - SERVICE:       服务端错误（5xx）

    冷却策略:
        - RATE_LIMIT: retry_after 夹逼到 [30s, 300s]，默认 60s
        - TIMEOUT:    60s
        - NETWORK:    120s
        - SERVICE:    30s
    示例:
        >>> info = classify_cloud_failure(RateLimitError("429", retry_after=45))
        >>> info.kind == CloudFailureKind.RATE_LIMIT
        >>> info.retry_after == 45
        >>> info.retryable == True

【类型 3】资源/配额失败 - 需要用户介入，不自动重试
    - QUOTA:         配额耗尽
    - UNKNOWN:       未知错误

    冷却策略: unknown_cooldown_seconds=300 (5分钟)
    恢复条件: 需要用户检查账户或联系支持
    示例:
        >>> info = classify_cloud_failure("quota exceeded")
        >>> info.kind == CloudFailureKind.QUOTA
        >>> info.retryable == False

================================================================================
核心概念
================================================================================

- cloud_enabled: config.yaml 中的总开关，控制 & 前缀的自动路由行为
- execution_mode: 显式指定的执行模式（cli/cloud/auto/plan/ask）

优先级规则:
1. 显式 execution_mode 参数（最高优先级）
   - execution_mode=cloud: 强制使用 Cloud，即使 cloud_enabled=False
   - execution_mode=cli: 强制使用 CLI，不受 & 前缀影响
   - execution_mode=auto: Cloud 优先，失败回退 CLI

2. & 前缀触发（当 execution_mode 未显式指定或为默认值时）
   - cloud_enabled=True: & 前缀路由到 Cloud
   - cloud_enabled=False: & 前缀被忽略，使用 CLI

3. 默认行为（函数级默认）
   - 当 requested_mode=None 且无 & 前缀时: effective_mode=cli, orchestrator=mp
   - 这是 build_execution_decision / resolve_effective_execution_mode 的函数内部默认
   - **注意**: 入口脚本通常传入 config.yaml 默认值（auto），所以函数级默认很少直接生效
   - 系统整体默认表现为 auto（来自 config.yaml），无 API Key 时回退 cli 且编排器强制 basic

================================================================================
使用场景
================================================================================

    from core.execution_policy import (
        resolve_effective_execution_mode,
        should_route_ampersand_to_cloud,
        classify_cloud_failure,
        build_user_facing_fallback_message,
        sanitize_prompt_for_cli_fallback,
    )

    # 解析实际执行模式
    mode, reason = resolve_effective_execution_mode(
        requested_mode="auto",
        has_ampersand_prefix=True,
        cloud_enabled=True,
        has_api_key=True,
    )

    # 检查是否应该路由 & 前缀到 Cloud
    should_cloud = should_route_ampersand_to_cloud(
        cloud_enabled=True,
        auto_detect_cloud_prefix=True,
        has_api_key=True,
    )

    # 分类 Cloud 错误
    failure_info = classify_cloud_failure(exception)

    # 构建用户友好的回退消息（只返回字符串，不打印）
    # has_ampersand_prefix 表示语法层面是否有 & 前缀，用于消息构建
    message = build_user_facing_fallback_message(
        kind="rate_limit",
        retry_after=60,
        requested_mode="cloud",
        has_ampersand_prefix=False,
    )

    # 清理 prompt 以避免回退时再次触发 Cloud
    clean_prompt = sanitize_prompt_for_cli_fallback(prompt)
"""

import contextlib
import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from core.cloud_utils import is_cloud_request, strip_cloud_prefix
from core.config import DEFAULT_EXECUTION_MODE
from core.contract_fields import CooldownInfoFields

# 模块级 logger - 仅用于 DEBUG 级别的内部调试日志
# 本模块遵循"只返回结构化元数据，不打印用户可见消息"的设计原则
logger = logging.getLogger(__name__)


# ============================================================
# 消息去重 - 稳定哈希工具
# ============================================================


def compute_message_dedup_key(text: str) -> str:
    """计算消息去重 key（稳定哈希）

    使用 SHA256 生成稳定的哈希值，用于消息去重。
    与 Python 内置 hash() 不同，此函数的返回值在不同进程/运行之间保持一致。

    ================================================================================
    为什么不使用 hash()
    ================================================================================

    Python 的内置 hash() 函数从 3.3 版本开始默认启用哈希随机化（PYTHONHASHSEED）：
    - 每次进程启动时，字符串的 hash 值可能不同
    - 这导致使用 hash() 作为去重 key 时，跨进程去重失效

    本函数使用 SHA256 算法，确保相同文本在任何环境下产生相同的 key。

    ================================================================================
    使用场景
    ================================================================================

    1. run.py - TaskAnalyzer._shown_messages 去重
    2. scripts/run_iterate.py - SelfIterator._shown_messages 去重

    注意：dedup_key 不属于 cooldown_info 契约，由入口脚本在需要时调用此函数计算。
    入口脚本通过 compute_message_dedup_key(cooldown_info[CooldownInfoFields.USER_MESSAGE]) 生成去重标识。

    ================================================================================
    示例
    ================================================================================

        >>> key1 = compute_message_dedup_key("⚠ 未设置 CURSOR_API_KEY")
        >>> key2 = compute_message_dedup_key("⚠ 未设置 CURSOR_API_KEY")
        >>> key1 == key2
        True

        >>> key3 = compute_message_dedup_key("不同的消息")
        >>> key1 == key3
        False

    Args:
        text: 要计算 dedup key 的消息文本

    Returns:
        SHA256 哈希值的十六进制字符串（前 16 位，足够去重使用）
    """
    if not text:
        return ""
    # 使用 SHA256 计算稳定哈希，取前 16 位作为 dedup key
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 执行模式枚举（兼容 cursor/executor.py 中的 ExecutionMode）
# ============================================================


class EffectiveExecutionMode(str, Enum):
    """有效执行模式

    与 cursor.executor.ExecutionMode 兼容，用于策略模块内部使用。
    """

    CLI = "cli"
    CLOUD = "cloud"
    AUTO = "auto"
    PLAN = "plan"
    ASK = "ask"


# ============================================================
# & 前缀状态枚举与数据类
# ============================================================


class AmpersandPrefixStatus(str, Enum):
    """& 前缀路由状态

    表示 & 前缀的检测和路由结果。

    Attributes:
        NOT_PRESENT: 原始 prompt 不包含 & 前缀
        DETECTED_ROUTED: 检测到 & 前缀，且成功路由到 Cloud
        DETECTED_IGNORED_CLI_MODE: 检测到 & 前缀，但因显式 CLI 模式而忽略
        DETECTED_IGNORED_EXPLICIT_MODE: 检测到 & 前缀，但因显式 cloud/auto/plan/ask 模式而不视为"触发"
        DETECTED_IGNORED_NO_KEY: 检测到 & 前缀，但因缺少 API Key 而忽略
        DETECTED_IGNORED_DISABLED: 检测到 & 前缀，但因 cloud_enabled=False 而忽略
        DETECTED_IGNORED_AUTO_DETECT_OFF: 检测到 & 前缀，但因 auto_detect_cloud_prefix=False 而忽略

    Examples:
        >>> # & 前缀成功路由到 Cloud
        >>> status = AmpersandPrefixStatus.DETECTED_ROUTED
        >>> status.has_prefix
        True
        >>> status.is_routed
        True

        >>> # & 前缀被忽略（无 API Key）
        >>> status = AmpersandPrefixStatus.DETECTED_IGNORED_NO_KEY
        >>> status.has_prefix
        True
        >>> status.is_routed
        False
    """

    NOT_PRESENT = "not_present"
    DETECTED_ROUTED = "detected_routed"
    DETECTED_IGNORED_CLI_MODE = "detected_ignored_cli_mode"
    DETECTED_IGNORED_EXPLICIT_MODE = "detected_ignored_explicit_mode"  # 显式 cloud/auto/plan/ask 模式
    DETECTED_IGNORED_NO_KEY = "detected_ignored_no_key"
    DETECTED_IGNORED_DISABLED = "detected_ignored_disabled"
    DETECTED_IGNORED_AUTO_DETECT_OFF = "detected_ignored_auto_detect_off"

    @property
    def has_prefix(self) -> bool:
        """原始 prompt 是否包含 & 前缀（语法层面）

        等价于旧代码中的 has_ampersand_prefix / is_cloud_request(prompt)。
        """
        return self != AmpersandPrefixStatus.NOT_PRESENT

    @property
    def is_routed(self) -> bool:
        """& 前缀是否成功路由到 Cloud（策略层面）

        等价于旧代码中的 prefix_routed / triggered_by_prefix（在表示成功触发时）。
        """
        return self == AmpersandPrefixStatus.DETECTED_ROUTED

    # ========== 别名属性（向后兼容） ==========

    @property
    def has_ampersand_prefix(self) -> bool:
        """别名: has_prefix 的语义明确版本

        用于明确表示"原始文本是否有 & 前缀"（语法检测层面）。
        """
        return self.has_prefix

    @property
    def prefix_routed(self) -> bool:
        """别名: is_routed 的语义明确版本

        用于明确表示"& 前缀是否成功触发 Cloud 模式"（策略决策层面）。
        等价于 triggered_by_prefix（当其表示成功触发时）。
        """
        return self.is_routed

    # ========== 兼容旧代码的属性 ==========

    @property
    def triggered_by_prefix(self) -> bool:
        """[DEPRECATED] 请使用 prefix_routed 或 is_routed

        此属性保留以兼容旧代码，语义等同于 prefix_routed。
        表示"& 前缀是否成功触发 Cloud 模式"。
        """
        return self.is_routed


@dataclass
class AmpersandPrefixInfo:
    """& 前缀检测与路由信息

    封装 & 前缀的完整状态信息，避免 triggered_by_prefix 语义歧义。

    设计原则：
    - has_ampersand_prefix: 纯语法层面的检测结果
    - prefix_routed: 策略层面的路由决策结果
    - status: 详细状态枚举，便于调试和日志

    Attributes:
        has_ampersand_prefix: 原始 prompt 是否以 '&' 开头（语法检测）
        prefix_routed: & 前缀是否成功触发 Cloud 模式（策略决策）
        status: 详细状态枚举
        ignore_reason: 当 prefix_routed=False 但 has_ampersand_prefix=True 时的原因

    Examples:
        >>> # & 前缀成功路由
        >>> info = AmpersandPrefixInfo(
        ...     has_ampersand_prefix=True,
        ...     prefix_routed=True,
        ...     status=AmpersandPrefixStatus.DETECTED_ROUTED,
        ... )
        >>> info.triggered_by_prefix  # 兼容旧代码
        True

        >>> # & 前缀被忽略
        >>> info = AmpersandPrefixInfo(
        ...     has_ampersand_prefix=True,
        ...     prefix_routed=False,
        ...     status=AmpersandPrefixStatus.DETECTED_IGNORED_NO_KEY,
        ...     ignore_reason="未配置 API Key",
        ... )
        >>> info.triggered_by_prefix  # 兼容旧代码
        False
    """

    # 核心字段（语义明确）
    has_ampersand_prefix: bool  # 语法检测：原始文本是否有 & 前缀
    prefix_routed: bool  # 策略决策：& 是否成功触发 Cloud

    # 详细状态
    status: AmpersandPrefixStatus = AmpersandPrefixStatus.NOT_PRESENT
    ignore_reason: Optional[str] = None  # 当 has_prefix 但未 routed 时的原因

    # ========== 兼容旧代码的属性 ==========

    @property
    def triggered_by_prefix(self) -> bool:
        """[DEPRECATED] 请使用 prefix_routed

        此属性保留以兼容旧代码，语义等同于 prefix_routed。
        表示"& 前缀是否成功触发 Cloud 模式"（非"是否存在 & 前缀"）。

        迁移说明：
        - 如需检测"是否存在 & 前缀"，请使用 has_ampersand_prefix
        - 如需检测"是否成功触发"，请使用 prefix_routed
        """
        return self.prefix_routed

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "has_ampersand_prefix": self.has_ampersand_prefix,
            "prefix_routed": self.prefix_routed,
            "triggered_by_prefix": self.triggered_by_prefix,  # 兼容字段
            "status": self.status.value,
            "ignore_reason": self.ignore_reason,
        }


def detect_ampersand_prefix(
    prompt: Optional[str],
    requested_mode: Optional[str],
    cloud_enabled: bool,
    has_api_key: bool,
    auto_detect_cloud_prefix: bool = True,
) -> AmpersandPrefixInfo:
    """检测 & 前缀并判断路由结果

    这是 & 前缀检测与路由的权威函数，统一封装检测逻辑。

    Args:
        prompt: 原始 prompt 文本
        requested_mode: 请求的执行模式（cli/cloud/auto/plan/ask）
        cloud_enabled: config.yaml 中的 cloud_agent.enabled 设置
        has_api_key: 是否配置了有效的 API Key
        auto_detect_cloud_prefix: 是否启用 & 前缀自动检测

    Returns:
        AmpersandPrefixInfo 包含完整的前缀状态信息

    Examples:
        >>> # 无 & 前缀
        >>> info = detect_ampersand_prefix("分析代码", None, True, True)
        >>> info.has_ampersand_prefix
        False
        >>> info.prefix_routed
        False

        >>> # & 前缀成功路由
        >>> info = detect_ampersand_prefix("& 分析代码", None, True, True)
        >>> info.has_ampersand_prefix
        True
        >>> info.prefix_routed
        True

        >>> # & 前缀被忽略（CLI 模式）
        >>> info = detect_ampersand_prefix("& 分析代码", "cli", True, True)
        >>> info.has_ampersand_prefix
        True
        >>> info.prefix_routed
        False
        >>> info.status
        AmpersandPrefixStatus.DETECTED_IGNORED_CLI_MODE
    """
    # Step 1: 语法检测
    has_prefix = is_cloud_request(prompt)

    if not has_prefix:
        return AmpersandPrefixInfo(
            has_ampersand_prefix=False,
            prefix_routed=False,
            status=AmpersandPrefixStatus.NOT_PRESENT,
        )

    # Step 2: 策略决策
    mode_lower = requested_mode.lower() if requested_mode else None

    # 显式 CLI 模式忽略 & 前缀
    if mode_lower == "cli":
        return AmpersandPrefixInfo(
            has_ampersand_prefix=True,
            prefix_routed=False,
            status=AmpersandPrefixStatus.DETECTED_IGNORED_CLI_MODE,
            ignore_reason="显式指定 execution_mode=cli",
        )

    # 显式 Cloud/Auto 模式：& 前缀不是"触发"执行模式的原因
    # 执行模式由用户显式设置决定，所以 prefix_routed=False
    if mode_lower in ("cloud", "auto"):
        return AmpersandPrefixInfo(
            has_ampersand_prefix=True,
            prefix_routed=False,  # 执行模式由显式设置决定，非 & 前缀触发
            status=AmpersandPrefixStatus.DETECTED_IGNORED_EXPLICIT_MODE,
            ignore_reason=f"显式指定 execution_mode={mode_lower}",
        )

    # 显式 Plan/Ask 模式：只读模式不参与 Cloud 路由
    # & 前缀被忽略，不触发 R-2 的 force_basic 逻辑
    if mode_lower in ("plan", "ask"):
        return AmpersandPrefixInfo(
            has_ampersand_prefix=True,
            prefix_routed=False,  # 只读模式不参与 Cloud 路由
            status=AmpersandPrefixStatus.DETECTED_IGNORED_EXPLICIT_MODE,
            ignore_reason=f"显式指定 execution_mode={mode_lower}（只读模式）",
        )

    # 自动检测被禁用
    if not auto_detect_cloud_prefix:
        return AmpersandPrefixInfo(
            has_ampersand_prefix=True,
            prefix_routed=False,
            status=AmpersandPrefixStatus.DETECTED_IGNORED_AUTO_DETECT_OFF,
            ignore_reason="auto_detect_cloud_prefix=False",
        )

    # Cloud 功能未启用
    if not cloud_enabled:
        return AmpersandPrefixInfo(
            has_ampersand_prefix=True,
            prefix_routed=False,
            status=AmpersandPrefixStatus.DETECTED_IGNORED_DISABLED,
            ignore_reason="cloud_enabled=False",
        )

    # 无 API Key
    if not has_api_key:
        return AmpersandPrefixInfo(
            has_ampersand_prefix=True,
            prefix_routed=False,
            status=AmpersandPrefixStatus.DETECTED_IGNORED_NO_KEY,
            ignore_reason="未配置 API Key",
        )

    # 成功路由到 Cloud
    return AmpersandPrefixInfo(
        has_ampersand_prefix=True,
        prefix_routed=True,
        status=AmpersandPrefixStatus.DETECTED_ROUTED,
    )


# ============================================================
# Cloud 错误类型
# ============================================================


class CloudFailureKind(str, Enum):
    """Cloud 错误类型枚举"""

    NO_KEY = "no_key"  # 未配置 API Key
    CLOUD_DISABLED = "cloud_disabled"  # cloud_enabled=False
    AUTH = "auth"  # 认证失败（API Key 无效、过期等）
    RATE_LIMIT = "rate_limit"  # 速率限制
    TIMEOUT = "timeout"  # 请求超时
    NETWORK = "network"  # 网络连接错误
    QUOTA = "quota"  # 配额耗尽
    SERVICE = "service"  # 服务端错误（5xx）
    UNKNOWN = "unknown"  # 未知错误


@dataclass
class CloudFailureInfo:
    """Cloud 错误分类结果"""

    kind: CloudFailureKind
    message: str
    retry_after: Optional[int] = None  # 建议重试等待时间（秒）
    retryable: bool = False  # 是否可重试
    original_error: Optional[Exception] = None

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "kind": self.kind.value,
            "message": self.message,
            "retry_after": self.retry_after,
            "retryable": self.retryable,
        }


# ============================================================
# 执行模式解析结果
# ============================================================


@dataclass
class ExecutionModeResolution:
    """执行模式解析结果"""

    mode: EffectiveExecutionMode
    reason: str
    fallback_mode: Optional[EffectiveExecutionMode] = None  # 当 mode 失败时的回退模式
    warnings: list[str] = field(default_factory=list)

    def to_tuple(self) -> tuple[str, str]:
        """返回 (mode, reason) 元组"""
        return (self.mode.value, self.reason)


# ============================================================
# 核心策略函数
# ============================================================


def resolve_effective_execution_mode(
    requested_mode: Optional[str],
    has_ampersand_prefix: Optional[bool] = None,
    cloud_enabled: bool = False,
    has_api_key: bool = False,
    *,
    triggered_by_prefix: Optional[bool] = None,  # DEPRECATED: 使用 has_ampersand_prefix
) -> tuple[str, str]:
    """解析实际执行模式

    根据请求参数、配置和上下文，确定最终应该使用的执行模式。

    优先级规则:
    1. 显式 requested_mode 参数（最高优先级）
       - cloud/auto: 需要 API Key，否则回退到 CLI
       - cli/plan/ask: 直接使用，不受 cloud_enabled 影响

    2. & 前缀触发（当 requested_mode 为 None 或 "cli" 时）
       - cloud_enabled=True + has_api_key: 使用 Cloud
       - cloud_enabled=True + 无 API Key: 警告并使用 CLI
       - cloud_enabled=False: 忽略 & 前缀，使用 CLI

    3. 默认行为
       - 无 & 前缀且无显式 requested_mode: 使用 CLI

    Args:
        requested_mode: 请求的执行模式 (cli/cloud/auto/plan/ask)，None 表示未指定
        has_ampersand_prefix: 原始 prompt 是否包含 '&' 前缀（语法检测层面）
            **语义**: 此参数表示语法层面的检测结果，即 is_cloud_request(prompt)。

            当 has_ampersand_prefix=True 时：
            - 函数基于 cloud_enabled、has_api_key 等条件决定是否路由到 Cloud
            - 若满足路由条件，返回 effective_mode="cloud"
            - 若不满足路由条件，返回 effective_mode="cli" 并附带原因

            当 has_ampersand_prefix=False 时：
            - 函数基于 requested_mode 等其他参数决策

            **推荐调用方式**: 使用 is_cloud_request(prompt) 获取语法检测结果，
            或使用 detect_ampersand_prefix() 获取完整状态后传入
            info.has_ampersand_prefix。

        cloud_enabled: config.yaml 中的 cloud_agent.enabled 设置
        has_api_key: 是否配置了有效的 API Key
        triggered_by_prefix: [DEPRECATED - 将在下一破坏性版本中移除]
            **旧语义**: has_ampersand_prefix 的别名，仅为兼容旧调用点。
            **严禁**与 prefix_routed 混用。

            迁移指南:
            - 此参数表示语法检测层面（原始文本是否有 & 前缀）
            - 若需策略决策层面（是否成功路由到 Cloud），请使用
              detect_ampersand_prefix().prefix_routed 或 build_execution_decision()
            - 此参数保留以兼容旧代码。若同时提供，has_ampersand_prefix 优先
            - 调用时会触发 DeprecationWarning（当前版本）

            仓库内调用点状态: 应为 0（通过 test_prefix_routed_migration.py 静态检查）

    Returns:
        元组 (mode, reason):
        - mode: 实际执行模式 (cli/cloud/auto/plan/ask)
        - reason: 决策原因描述

    Examples:
        >>> # 显式指定 cloud 模式
        >>> resolve_effective_execution_mode("cloud", False, False, True)
        ('cloud', '显式指定 execution_mode=cloud')

        >>> # & 前缀触发且 cloud_enabled（has_ampersand_prefix=True）
        >>> resolve_effective_execution_mode(None, has_ampersand_prefix=True, cloud_enabled=True, has_api_key=True)
        ('cloud', '& 前缀触发 Cloud 模式（cloud_enabled=True）')

        >>> # & 前缀存在但 cloud_enabled=False
        >>> resolve_effective_execution_mode(None, has_ampersand_prefix=True, cloud_enabled=False, has_api_key=True)
        ('cli', '& 前缀被忽略（cloud_enabled=False），使用 CLI')

    See Also:
        - detect_ampersand_prefix: 获取完整的前缀状态信息
        - build_execution_decision: 构建完整的执行决策（推荐使用）
    """
    # === 处理兼容别名 triggered_by_prefix ===
    # 优先使用 has_ampersand_prefix，回退到 triggered_by_prefix（兼容旧代码）
    effective_has_ampersand_prefix: bool
    if has_ampersand_prefix is not None:
        effective_has_ampersand_prefix = has_ampersand_prefix
    elif triggered_by_prefix is not None:
        # 兼容旧调用方式，发出 DeprecationWarning（当前版本）
        import warnings

        warnings.warn(
            "resolve_effective_execution_mode() 的 triggered_by_prefix 参数已弃用，"
            "将在下一破坏性版本中移除。请使用 has_ampersand_prefix 参数。"
            "注意：此参数为 has_ampersand_prefix 的别名，严禁与 prefix_routed 混用。",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.debug(
            "[resolve_effective_execution_mode] 使用了 deprecated 参数 "
            "triggered_by_prefix=%s，请迁移到 has_ampersand_prefix",
            triggered_by_prefix,
        )
        effective_has_ampersand_prefix = triggered_by_prefix
    else:
        effective_has_ampersand_prefix = False

    # 规范化 requested_mode
    mode_lower = requested_mode.lower() if requested_mode else None

    # === 规则 1: 显式 requested_mode 参数 ===
    if mode_lower in ("cloud", "auto"):
        if not has_api_key:
            return ("cli", f"请求 {mode_lower} 模式但未配置 API Key，回退到 CLI")
        return (mode_lower, f"显式指定 execution_mode={mode_lower}")

    if mode_lower == "plan":
        return ("plan", "显式指定 execution_mode=plan（只读模式）")

    if mode_lower == "ask":
        return ("ask", "显式指定 execution_mode=ask（只读模式）")

    if mode_lower == "cli":
        # 显式指定 CLI，忽略 & 前缀
        if effective_has_ampersand_prefix:
            return ("cli", "显式指定 execution_mode=cli，忽略 & 前缀")
        return ("cli", "显式指定 execution_mode=cli")

    # === 规则 2: & 前缀触发 ===
    if effective_has_ampersand_prefix:
        if cloud_enabled:
            if has_api_key:
                return ("cloud", "& 前缀触发 Cloud 模式（cloud_enabled=True）")
            else:
                return ("cli", "& 前缀触发但未配置 API Key，回退到 CLI")
        else:
            return ("cli", "& 前缀被忽略（cloud_enabled=False），使用 CLI")

    # === 规则 3: 默认行为（函数级默认） ===
    # 当 requested_mode=None 且无 & 前缀时，返回 cli 模式
    # **注意**: 这是函数内部的硬编码默认值，但入口脚本通常会传入
    # config.yaml 的默认值（auto），所以此分支很少直接触发。
    # 系统整体默认表现为 auto（来自 config.yaml），无 key 时回退 cli 且编排器强制 basic。
    return ("cli", "函数级默认: requested_mode=None 且无 & 前缀 → CLI 模式")


def resolve_effective_execution_mode_full(
    requested_mode: Optional[str],
    has_ampersand_prefix: Optional[bool] = None,
    cloud_enabled: bool = False,
    has_api_key: bool = False,
    *,
    triggered_by_prefix: Optional[bool] = None,  # DEPRECATED: 使用 has_ampersand_prefix
) -> ExecutionModeResolution:
    """解析实际执行模式（完整版）

    与 resolve_effective_execution_mode 相同的逻辑，但返回更详细的解析结果。

    Args:
        requested_mode: 请求的执行模式
        has_ampersand_prefix: 原始 prompt 是否包含 '&' 前缀（语法检测层面）
        cloud_enabled: 是否启用 Cloud
        has_api_key: 是否有 API Key
        triggered_by_prefix: [DEPRECATED - 将在下一破坏性版本中移除]
            **旧语义**: has_ampersand_prefix 的别名，仅为兼容旧调用点。
            **严禁**与 prefix_routed 混用。
            此参数保留以兼容旧代码。若同时提供，has_ampersand_prefix 优先。

    Returns:
        ExecutionModeResolution 包含模式、原因、回退模式和警告信息
    """
    # 处理兼容别名
    effective_has_ampersand_prefix: bool
    if has_ampersand_prefix is not None:
        effective_has_ampersand_prefix = has_ampersand_prefix
    elif triggered_by_prefix is not None:
        # 兼容旧调用方式，发出 DeprecationWarning
        import warnings

        warnings.warn(
            "resolve_effective_execution_mode_full() 的 triggered_by_prefix 参数已弃用，"
            "将在下一破坏性版本中移除。请使用 has_ampersand_prefix 参数。"
            "注意：此参数为 has_ampersand_prefix 的别名，严禁与 prefix_routed 混用。",
            DeprecationWarning,
            stacklevel=2,
        )
        effective_has_ampersand_prefix = triggered_by_prefix
    else:
        effective_has_ampersand_prefix = False

    mode_str, reason = resolve_effective_execution_mode(
        requested_mode,
        has_ampersand_prefix=effective_has_ampersand_prefix,
        cloud_enabled=cloud_enabled,
        has_api_key=has_api_key,
    )

    mode = EffectiveExecutionMode(mode_str)
    warning_messages: list[str] = []
    fallback_mode = None

    # 确定回退模式
    if mode in (EffectiveExecutionMode.CLOUD, EffectiveExecutionMode.AUTO):
        fallback_mode = EffectiveExecutionMode.CLI

    # 添加警告信息
    if effective_has_ampersand_prefix and not cloud_enabled:
        warning_messages.append(
            "检测到 & 前缀但 cloud_enabled=False，如需使用 Cloud 请在 config.yaml 中设置 cloud_agent.enabled=true"
        )

    if mode in (EffectiveExecutionMode.CLOUD, EffectiveExecutionMode.AUTO) and not has_api_key:
        warning_messages.append("Cloud 模式需要 API Key，请设置 CURSOR_API_KEY 环境变量或在 config.yaml 中配置")

    return ExecutionModeResolution(
        mode=mode,
        reason=reason,
        fallback_mode=fallback_mode,
        warnings=warning_messages,
    )


def should_route_ampersand_to_cloud(
    cloud_enabled: bool,
    auto_detect_cloud_prefix: bool,
    has_api_key: bool,
) -> bool:
    """判断 & 前缀是否应该路由到 Cloud

    用于在检测到 & 前缀后，判断是否应该使用 Cloud 执行。

    决策逻辑:
    1. auto_detect_cloud_prefix=False: 不路由（禁用了自动检测）
    2. cloud_enabled=False: 不路由（Cloud 功能未启用）
    3. has_api_key=False: 不路由（缺少认证信息）
    4. 以上条件都满足: 路由到 Cloud

    Args:
        cloud_enabled: config.yaml 中的 cloud_agent.enabled 设置
        auto_detect_cloud_prefix: 是否启用 & 前缀自动检测
        has_api_key: 是否配置了有效的 API Key

    Returns:
        是否应该路由到 Cloud

    Examples:
        >>> should_route_ampersand_to_cloud(True, True, True)
        True
        >>> should_route_ampersand_to_cloud(False, True, True)
        False
        >>> should_route_ampersand_to_cloud(True, True, False)
        False
    """
    if not auto_detect_cloud_prefix:
        return False
    if not cloud_enabled:
        return False
    return has_api_key


def classify_cloud_failure(
    error: Union[Exception, str, dict, None],
) -> CloudFailureInfo:
    """分类 Cloud 执行错误

    根据异常类型和错误信息，将 Cloud 错误分类为特定类型。
    用于决定回退策略和构建用户友好的错误消息。

    优先级:
    1. 如果 error 是 dict 且包含 error_type 字段，直接使用结构化字段
    2. 如果 error 是 CloudAgentResult 对象且有 error_type 属性，直接使用
    3. 否则从错误消息字符串中解析

    分类逻辑:
    - NO_KEY: 未配置 API Key
    - AUTH: 认证失败（401/403，API Key 无效）
    - RATE_LIMIT: 速率限制（429）
    - TIMEOUT: 请求超时
    - NETWORK: 网络连接错误
    - QUOTA: 配额耗尽
    - SERVICE: 服务端错误（5xx）
    - UNKNOWN: 其他错误

    Args:
        error: 异常对象、错误信息字符串、CloudAgentResult.to_dict() 字典或 None

    Returns:
        CloudFailureInfo 包含错误类型、消息、重试建议等

    Examples:
        >>> info = classify_cloud_failure(TimeoutError("Connection timed out"))
        >>> info.kind
        CloudFailureKind.TIMEOUT

        >>> info = classify_cloud_failure("Rate limit exceeded")
        >>> info.kind
        CloudFailureKind.RATE_LIMIT

        >>> # 优先使用结构化字段
        >>> info = classify_cloud_failure({"error": "...", "error_type": "rate_limit", "retry_after": 60})
        >>> info.kind
        CloudFailureKind.RATE_LIMIT
        >>> info.retry_after
        60
    """
    if error is None:
        return CloudFailureInfo(
            kind=CloudFailureKind.UNKNOWN,
            message="未知错误（无错误信息）",
            retryable=False,
        )

    # === 优先处理结构化输入 ===
    # 1. 如果是 dict（可能来自 CloudAgentResult.to_dict()）
    if isinstance(error, dict):
        structured_type = error.get("error_type")
        if structured_type:
            return _classify_from_structured_type(
                structured_type=structured_type,
                message=error.get("error") or error.get("message") or str(error),
                retry_after=error.get("retry_after"),
            )
        # 如果 dict 没有 error_type，从 error 字段继续解析
        error = error.get("error") or error.get("message") or str(error)

    # 2. 如果是对象且有 error_type 属性（可能是 CloudAgentResult）
    #    注意：如果 error_type 为 "unknown"，跳过此检查，继续使用字符串解析
    #    这样可以让 NetworkError 等异常类型正确分类
    error_type_attr = getattr(error, "error_type", None) if hasattr(error, "error_type") else None
    if error_type_attr and error_type_attr != "unknown":
        return _classify_from_structured_type(
            structured_type=error_type_attr,
            message=getattr(error, "error", None) or str(error),
            retry_after=getattr(error, "retry_after", None),
            original_error=error if isinstance(error, Exception) else None,
        )

    # === 回退到字符串解析 ===
    # 获取错误信息字符串
    error_str = str(error).lower() if error else ""
    error_type = type(error).__name__ if isinstance(error, Exception) else ""
    original_error = error if isinstance(error, Exception) else None

    # 尝试从异常对象中提取 retry_after 属性（如 RateLimitError）
    exception_retry_after: Optional[int] = None
    if isinstance(error, Exception):
        raw_retry_after = getattr(error, "retry_after", None)
        if raw_retry_after is not None:
            with contextlib.suppress(ValueError, TypeError):
                exception_retry_after = int(raw_retry_after)

    # === NO_KEY: 未配置 API Key ===
    no_key_patterns = [
        "no api key",
        "api key not",
        "missing api key",
        "api_key is required",
        "未配置 api key",
        "缺少 api key",
    ]
    if any(p in error_str for p in no_key_patterns):
        return CloudFailureInfo(
            kind=CloudFailureKind.NO_KEY,
            message="未配置 Cloud API Key",
            retryable=False,
            original_error=original_error,
        )

    # === AUTH: 认证失败 ===
    auth_patterns = [
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid token",
        "authentication failed",
        "认证失败",
        "api key 无效",
        "token 过期",
    ]
    # 检查异常类型是否为 AuthError
    is_auth_exception = error_type in ("AuthError",)
    if is_auth_exception or any(p in error_str for p in auth_patterns):
        return CloudFailureInfo(
            kind=CloudFailureKind.AUTH,
            message="Cloud API 认证失败，请检查 API Key 是否有效",
            retryable=False,
            original_error=original_error,
        )

    # === RATE_LIMIT: 速率限制 ===
    rate_limit_patterns = [
        "429",
        "rate limit",
        "too many requests",
        "throttl",
        "retry after",
        "retry-after",
        "速率限制",
        "请求过多",
    ]
    # 检查异常类型是否为 RateLimitError
    is_rate_limit_exception = error_type in ("RateLimitError",)
    if is_rate_limit_exception or any(p in error_str for p in rate_limit_patterns):
        # 优先使用异常对象的 retry_after，其次从字符串提取
        retry_after = exception_retry_after
        if retry_after is None:
            retry_after = _extract_retry_after(error_str)
        if retry_after is None:
            retry_after = 60  # 默认 60 秒
        return CloudFailureInfo(
            kind=CloudFailureKind.RATE_LIMIT,
            message="Cloud API 速率限制",
            retry_after=retry_after,
            retryable=True,
            original_error=original_error,
        )

    # === TIMEOUT: 超时 ===
    timeout_patterns = [
        "timeout",
        "timed out",
        "deadline exceeded",
        "超时",
    ]
    timeout_types = ["TimeoutError", "asyncio.TimeoutError"]
    if any(p in error_str for p in timeout_patterns) or error_type in timeout_types:
        return CloudFailureInfo(
            kind=CloudFailureKind.TIMEOUT,
            message="Cloud API 请求超时",
            retryable=True,
            original_error=original_error,
        )

    # === NETWORK: 网络错误 ===
    network_patterns = [
        "connection",
        "network",
        "dns",
        "socket",
        "unreachable",
        "refused",
        "reset",
        "网络",
        "连接",
    ]
    network_types = ["ConnectionError", "ConnectionRefusedError", "OSError", "aiohttp.ClientError", "NetworkError"]
    if any(p in error_str for p in network_patterns) or error_type in network_types:
        return CloudFailureInfo(
            kind=CloudFailureKind.NETWORK,
            message="Cloud API 网络连接失败",
            retryable=True,
            original_error=original_error,
        )

    # === QUOTA: 配额耗尽 ===
    quota_patterns = [
        "quota",
        "limit exceeded",
        "usage limit",
        "billing",
        "配额",
        "额度",
    ]
    if any(p in error_str for p in quota_patterns):
        return CloudFailureInfo(
            kind=CloudFailureKind.QUOTA,
            message="Cloud API 配额已耗尽",
            retryable=False,
            original_error=original_error,
        )

    # === SERVICE: 服务端错误 ===
    service_patterns = [
        "500",
        "502",
        "503",
        "504",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "服务器错误",
        "服务不可用",
    ]
    if any(p in error_str for p in service_patterns):
        return CloudFailureInfo(
            kind=CloudFailureKind.SERVICE,
            message="Cloud API 服务暂时不可用",
            retry_after=30,  # 服务端错误建议 30 秒后重试
            retryable=True,
            original_error=original_error,
        )

    # === UNKNOWN: 未知错误 ===
    return CloudFailureInfo(
        kind=CloudFailureKind.UNKNOWN,
        message=f"Cloud API 执行失败: {str(error)[:100]}",
        retryable=False,
        original_error=original_error,
    )


def _classify_from_structured_type(
    structured_type: str,
    message: str,
    retry_after: Optional[float] = None,
    original_error: Optional[Exception] = None,
) -> CloudFailureInfo:
    """从结构化错误类型创建 CloudFailureInfo

    Args:
        structured_type: 结构化错误类型字符串 (auth, rate_limit, timeout, etc.)
        message: 错误消息
        retry_after: 建议重试等待时间
        original_error: 原始异常

    Returns:
        CloudFailureInfo 实例
    """
    type_lower = structured_type.lower()

    # 错误类型映射
    type_mapping = {
        "auth": (CloudFailureKind.AUTH, "Cloud API 认证失败", False),
        "no_key": (CloudFailureKind.NO_KEY, "未配置 Cloud API Key", False),
        "cloud_disabled": (CloudFailureKind.CLOUD_DISABLED, "Cloud 功能未启用", False),
        "rate_limit": (CloudFailureKind.RATE_LIMIT, "Cloud API 速率限制", True),
        "timeout": (CloudFailureKind.TIMEOUT, "Cloud API 请求超时", True),
        "network": (CloudFailureKind.NETWORK, "Cloud API 网络连接失败", True),
        "connection": (CloudFailureKind.NETWORK, "Cloud API 网络连接失败", True),
        "quota": (CloudFailureKind.QUOTA, "Cloud API 配额已耗尽", False),
        "service": (CloudFailureKind.SERVICE, "Cloud API 服务暂时不可用", True),
        "not_found": (CloudFailureKind.UNKNOWN, "资源不存在", False),
        "unknown": (CloudFailureKind.UNKNOWN, "Cloud API 执行失败", False),
    }

    kind, default_message, retryable = type_mapping.get(
        type_lower, (CloudFailureKind.UNKNOWN, "Cloud API 执行失败", False)
    )

    # 确定 retry_after
    final_retry_after = None
    if retry_after is not None:
        final_retry_after = int(retry_after)
    elif kind == CloudFailureKind.RATE_LIMIT:
        final_retry_after = 60  # 默认 60 秒
    elif kind == CloudFailureKind.SERVICE:
        final_retry_after = 30  # 服务端错误建议 30 秒

    return CloudFailureInfo(
        kind=kind,
        message=message or default_message,
        retry_after=final_retry_after,
        retryable=retryable,
        original_error=original_error,
    )


def _extract_retry_after(error_str: str) -> Optional[int]:
    """从错误信息中提取 retry-after 值"""
    # 尝试匹配常见格式: "retry after 60s", "retry-after: 60", "wait 60 seconds"
    patterns = [
        r"retry[- ]?after[:\s]+(\d+)",
        r"wait\s+(\d+)\s*(?:second|sec|s)",
        r"(\d+)\s*(?:second|sec|s)\s*(?:later|wait)",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_str, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return None


def build_user_facing_fallback_message(
    kind: Union[CloudFailureKind, str],
    retry_after: Optional[int],
    requested_mode: Optional[str],
    has_ampersand_prefix: bool,
) -> str:
    """构建用户友好的回退消息

    当 Cloud 执行失败回退到 CLI 时，生成用户可读的说明消息。

    消息内容:
    - 说明失败原因
    - 告知回退行为
    - 提供建议操作

    Args:
        kind: 错误类型（CloudFailureKind 或字符串）
        retry_after: 建议重试等待时间（秒）
        requested_mode: 原请求的执行模式
        has_ampersand_prefix: 原始 prompt 是否包含 '&' 前缀（语法检测层面）
            **语义说明**: 此参数表示"原始文本是否有 & 前缀"，用于决定消息中
            是否提及 '&' 前缀触发行为。

            **与 prefix_routed/triggered_by_prefix 的区别**:
            - has_ampersand_prefix: 语法层面检测，等价于 is_cloud_request(prompt)
            - prefix_routed/triggered_by_prefix: 策略层面决策，表示 & 是否成功触发 Cloud

            推荐传入: AmpersandPrefixInfo.has_ampersand_prefix 或
                      is_cloud_request(prompt)

    Returns:
        用户友好的消息字符串

    Examples:
        >>> build_user_facing_fallback_message("rate_limit", 60, "cloud", False)
        '⚠ Cloud API 速率限制，60 秒后可重试\\n→ 正在回退到本地 CLI 执行...'

        >>> # 当原始 prompt 有 & 前缀时，消息会包含前缀说明
        >>> build_user_facing_fallback_message("timeout", None, None, has_ampersand_prefix=True)
        '...(由 & 前缀触发)'
    """
    # 规范化 kind
    if isinstance(kind, str):
        try:
            kind = CloudFailureKind(kind)
        except ValueError:
            kind = CloudFailureKind.UNKNOWN

    # 构建基础消息
    messages = {
        CloudFailureKind.NO_KEY: ("⚠ 未配置 Cloud API Key\n→ 请设置 CURSOR_API_KEY 环境变量或在 config.yaml 中配置"),
        CloudFailureKind.CLOUD_DISABLED: (
            "ℹ 检测到 '&' 前缀但 cloud_enabled=False（disabled）\n"
            "→ 使用 CLI 模式。如需启用，请设置 cloud_agent.enabled=true"
        ),
        CloudFailureKind.AUTH: ("⚠ Cloud API 认证失败\n→ 请检查 API Key 是否正确且未过期"),
        CloudFailureKind.RATE_LIMIT: (
            "⚠ Cloud API 速率限制" + (f"，{retry_after} 秒后可重试" if retry_after else "") + "\n"
            "→ 正在回退到本地 CLI 执行..."
        ),
        CloudFailureKind.TIMEOUT: ("⚠ Cloud API 请求超时\n→ 正在回退到本地 CLI 执行..."),
        CloudFailureKind.NETWORK: ("⚠ Cloud API 网络连接失败\n→ 请检查网络连接，正在回退到本地 CLI 执行..."),
        CloudFailureKind.QUOTA: ("⚠ Cloud API 配额已耗尽\n→ 请检查账户配额，正在回退到本地 CLI 执行..."),
        CloudFailureKind.SERVICE: ("⚠ Cloud API 服务暂时不可用\n→ 正在回退到本地 CLI 执行..."),
        CloudFailureKind.UNKNOWN: ("⚠ Cloud API 执行失败\n→ 正在回退到本地 CLI 执行..."),
    }

    base_message = messages.get(kind, messages[CloudFailureKind.UNKNOWN])

    # 添加触发方式信息
    # 使用 has_ampersand_prefix（语法检测层面）决定是否提及 & 前缀
    trigger_info = ""
    if has_ampersand_prefix:
        trigger_info = "\n(由 & 前缀触发)"
    elif requested_mode in ("cloud", "auto"):
        trigger_info = f"\n(执行模式: {requested_mode})"

    return base_message + trigger_info


def sanitize_prompt_for_cli_fallback(prompt: Optional[str]) -> str:
    """清理 prompt 以避免 CLI 回退时再次触发 Cloud 路由

    当 Cloud 执行失败回退到 CLI 时，需要确保 prompt 不会再次触发 Cloud 路由。
    主要操作是移除 & 前缀。

    处理规则:
    1. 移除开头的 & 前缀及其后的空白
    2. 处理多个 & 的情况（只移除第一个作为前缀的 &）
    3. 保留 prompt 中间的 & 符号（非前缀用途）
    4. 处理 None 和空字符串

    Args:
        prompt: 可能带 & 前缀的原始 prompt

    Returns:
        清理后的 prompt，确保不会再次触发 Cloud 路由

    Examples:
        >>> sanitize_prompt_for_cli_fallback("& 分析代码")
        '分析代码'

        >>> sanitize_prompt_for_cli_fallback("& 包含 & 符号的任务")
        '包含 & 符号的任务'

        >>> sanitize_prompt_for_cli_fallback("普通任务")
        '普通任务'
    """
    if not prompt:
        return ""

    # 使用 cloud_utils 中的权威实现
    if is_cloud_request(prompt):
        return strip_cloud_prefix(prompt)

    return prompt


# ============================================================
# cooldown_info 构建函数 - 统一回退信息结构
# ============================================================


@dataclass
class CooldownInfo:
    """冷却信息结构

    统一 cooldown_info 字典的结构，供 Cloud->CLI 回退时使用。
    确保 cursor/executor.py 和 cursor/client.py 输出一致的结构。

    ================================================================================
    字段说明
    ================================================================================

    核心字段（必须存在）:
    - kind: CloudFailureKind 的值（如 "no_key", "rate_limit"），统一输出
    - user_message: 用户友好的提示消息
    - retryable: 是否可重试
    - retry_after: 建议重试等待时间（秒），可为 None

    原因/上下文字段:
    - reason: 回退原因（技术性描述）
    - fallback_reason: 别名，与 reason 相同值

    兼容字段（向后兼容，新代码应使用 kind）:
    - error_type: 旧版错误类型字符串（与 kind 映射）
    - failure_kind: 别名，与 kind 相同值

    冷却状态字段（可选，由 executor 层填充）:
    - in_cooldown: 当前是否处于冷却期
    - remaining_seconds: 冷却剩余秒数
    - failure_count: 连续失败次数

    ================================================================================
    使用场景
    ================================================================================

    1. cursor/client.py _execute_via_cloud:
       使用 build_cooldown_info() 构建回退信息

    2. cursor/executor.py AutoAgentExecutor:
       使用 build_cooldown_info() 或 build_cooldown_info_from_metadata() 构建

    3. 入口脚本 (run.py, scripts/run_iterate.py):
       读取 result.cooldown_info[CooldownInfoFields.USER_MESSAGE] 输出给用户
       根据 result.cooldown_info[CooldownInfoFields.MESSAGE_LEVEL] 选择 print_warning 或 print_info

    消息级别 (message_level) 策略:
    - mode_source="cli": 用户显式 --execution-mode auto/cloud → "warning"
    - mode_source="config" 或 None: config.yaml 默认值 → "info"（避免每次都警告）
    - has_ampersand_prefix=True 且未成功路由: 用户显式使用 & 前缀 → "warning"

    Attributes:
        kind: CloudFailureKind 值 (no_key, auth, rate_limit, etc.)
        user_message: 用户友好消息
        retryable: 是否可重试
        retry_after: 建议重试等待时间（秒）
        reason: 回退原因
        in_cooldown: 是否处于冷却期
        remaining_seconds: 冷却剩余秒数
        failure_count: 连续失败次数
        message_level: 消息级别 ("warning" 或 "info")
    """

    # 核心字段
    kind: str  # CloudFailureKind.value
    user_message: str
    retryable: bool
    retry_after: Optional[int] = None

    # 原因字段
    reason: str = ""

    # 冷却状态字段（可选）
    in_cooldown: bool = False
    remaining_seconds: Optional[float] = None
    failure_count: int = 0

    # 消息级别（控制入口脚本使用 print_warning 还是 print_info）
    message_level: str = "info"

    # 扩展字段
    skip_reason: Optional[str] = None  # 跳过 Cloud 的原因

    # ========== 兼容属性 ==========

    @property
    def error_type(self) -> str:
        """[兼容] 映射 kind 到旧版 error_type

        映射规则：
        - no_key -> auth
        - cloud_disabled -> auth
        - quota -> auth
        - service -> network
        - 其他保持不变
        """
        kind_to_error_type = {
            "no_key": "auth",
            "cloud_disabled": "auth",
            "quota": "auth",
            "service": "network",
        }
        return kind_to_error_type.get(self.kind, self.kind)

    @property
    def failure_kind(self) -> str:
        """[兼容] 别名，等同于 kind"""
        return self.kind

    @property
    def fallback_reason(self) -> str:
        """[兼容] 别名，等同于 reason"""
        return self.reason

    def to_dict(self) -> dict:
        """转换为字典格式

        输出包含所有字段，确保兼容性。
        使用 CooldownInfoFields 常量确保字段名与契约一致。
        """
        return {
            # 核心字段
            CooldownInfoFields.KIND: self.kind,
            CooldownInfoFields.USER_MESSAGE: self.user_message,
            CooldownInfoFields.RETRYABLE: self.retryable,
            CooldownInfoFields.RETRY_AFTER: self.retry_after,
            # 原因字段
            CooldownInfoFields.REASON: self.reason,
            CooldownInfoFields.FALLBACK_REASON: self.fallback_reason,  # 兼容
            # 兼容字段
            CooldownInfoFields.ERROR_TYPE: self.error_type,
            CooldownInfoFields.FAILURE_KIND: self.failure_kind,
            # 冷却状态字段
            CooldownInfoFields.IN_COOLDOWN: self.in_cooldown,
            CooldownInfoFields.REMAINING_SECONDS: self.remaining_seconds,
            CooldownInfoFields.FAILURE_COUNT: self.failure_count,
            # 消息级别（控制入口脚本打印方式）
            CooldownInfoFields.MESSAGE_LEVEL: self.message_level,
            # 扩展字段（始终输出，值可为 None）
            CooldownInfoFields.SKIP_REASON: self.skip_reason,
        }


def build_cooldown_info(
    failure_info: CloudFailureInfo,
    fallback_reason: Optional[str] = None,
    requested_mode: Optional[str] = None,
    has_ampersand_prefix: bool = False,
    in_cooldown: bool = False,
    remaining_seconds: Optional[float] = None,
    failure_count: int = 0,
    mode_source: Optional[str] = None,
) -> dict:
    """构建统一的 cooldown_info 字典

    这是 cooldown_info 构建的**权威函数**，供 cursor/executor.py 和
    cursor/client.py 统一使用，确保回退信息结构一致。

    ================================================================================
    输出字段契约
    ================================================================================

    必须存在的字段:
    - kind (str): CloudFailureKind 值，如 "no_key", "rate_limit"
    - user_message (str): 用户友好消息，非空
    - retryable (bool): 是否可重试
    - retry_after (int|None): 建议重试等待秒数
    - message_level (str): 消息级别，"warning" 或 "info"

    必须存在的原因字段:
    - reason (str): 回退原因
    - fallback_reason (str): 兼容别名

    兼容字段（向后兼容）:
    - error_type (str): 旧版错误类型
    - failure_kind (str): 与 kind 相同

    冷却状态字段:
    - in_cooldown (bool): 是否处于冷却期
    - remaining_seconds (float|None): 冷却剩余秒数
    - failure_count (int): 连续失败次数

    ================================================================================
    消息级别 (message_level) 策略
    ================================================================================

    | mode_source | has_ampersand_prefix | message_level | 说明 |
    |-------------|---------------------|---------------|------|
    | "cli"       | 任意                | "warning"     | 用户显式 --execution-mode auto/cloud |
    | "config"    | False               | "info"        | config.yaml 默认值，避免每次都警告 |
    | "config"    | True                | "warning"     | 用户显式使用 & 前缀表示意图 |
    | None        | False               | "info"        | 默认情况 |
    | None        | True                | "warning"     | 用户显式使用 & 前缀表示意图 |

    ================================================================================
    使用示例
    ================================================================================

    cursor/client.py:
        failure_info = classify_cloud_failure(error)
        cooldown_info = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason=failure_info.message,
            requested_mode=self.config.execution_mode,
            has_ampersand_prefix=self._is_cloud_request(instruction),
            mode_source="config",  # 或 "cli"（根据来源）
        )
        result.cooldown_info = cooldown_info

    cursor/executor.py:
        cooldown_info = build_cooldown_info(
            failure_info=failure_info,
            fallback_reason=str(e),
            requested_mode="auto",
            has_ampersand_prefix=has_ampersand_prefix,
            in_cooldown=cooldown_meta.in_cooldown,
            remaining_seconds=cooldown_meta.remaining_seconds,
            failure_count=cooldown_meta.failure_count,
            mode_source=mode_source,
        )

    Args:
        failure_info: CloudFailureInfo 实例（来自 classify_cloud_failure）
        fallback_reason: 回退原因描述（可选，默认使用 failure_info.message）
        requested_mode: 请求的执行模式（用于构建 user_message）
        has_ampersand_prefix: 原始 prompt 是否有 & 前缀（语法检测层面）
        in_cooldown: 是否处于冷却期
        remaining_seconds: 冷却剩余秒数
        failure_count: 连续失败次数
        mode_source: execution_mode 的来源，用于决定消息级别
            - "cli": 来自 CLI 显式参数（--execution-mode）→ "warning"
            - "config": 来自 config.yaml 配置 → "info"（无 & 前缀时）
            - None: 未指定（使用默认值）→ "info"（无 & 前缀时）

    Returns:
        符合契约的 cooldown_info 字典
    """
    # 构建 user_message
    user_message = build_user_facing_fallback_message(
        kind=failure_info.kind,
        retry_after=failure_info.retry_after,
        requested_mode=requested_mode,
        has_ampersand_prefix=has_ampersand_prefix,
    )

    # 确定 reason
    reason = fallback_reason or failure_info.message

    # 计算 message_level
    # 策略：
    # - mode_source="cli": 用户显式请求，使用 "warning"
    # - mode_source="config" 或 None: 配置默认值，使用 "info"（避免每次都警告）
    # - has_ampersand_prefix=True: 用户显式使用 & 前缀表示意图，使用 "warning"
    message_level = "info"  # 默认使用 info 级别
    if mode_source == "cli":
        # 用户显式 --execution-mode auto/cloud
        message_level = "warning"
    elif has_ampersand_prefix:
        # 用户显式使用 & 前缀表示 Cloud 意图，应该警告
        message_level = "warning"
    # else: mode_source="config" 或 None，保持 "info"

    # 创建 CooldownInfo 实例并转换为字典
    info = CooldownInfo(
        kind=failure_info.kind.value,
        user_message=user_message,
        retryable=failure_info.retryable,
        retry_after=failure_info.retry_after,
        reason=reason,
        in_cooldown=in_cooldown,
        remaining_seconds=remaining_seconds,
        failure_count=failure_count,
        message_level=message_level,
    )

    return info.to_dict()


def build_cooldown_info_from_metadata(
    failure_kind: Optional[CloudFailureKind],
    failure_message: Optional[str],
    retry_after: Optional[int],
    retryable: bool,
    fallback_reason: Optional[str] = None,
    requested_mode: Optional[str] = None,
    has_ampersand_prefix: bool = False,
    in_cooldown: bool = False,
    remaining_seconds: Optional[float] = None,
    failure_count: int = 0,
    mode_source: Optional[str] = None,
) -> dict:
    """从元数据构建 cooldown_info（不需要 CloudFailureInfo 实例）

    当已有分散的元数据（如 CooldownMetadata）时使用此函数。

    Args:
        failure_kind: CloudFailureKind 枚举值
        failure_message: 错误消息
        retry_after: 建议重试等待时间
        retryable: 是否可重试
        fallback_reason: 回退原因
        requested_mode: 请求的执行模式
        has_ampersand_prefix: 原始 prompt 是否有 & 前缀
        in_cooldown: 是否处于冷却期
        remaining_seconds: 冷却剩余秒数
        failure_count: 连续失败次数
        mode_source: execution_mode 的来源（"cli"/"config"/None）

    Returns:
        符合契约的 cooldown_info 字典
    """
    # 处理 None 的情况
    kind = failure_kind or CloudFailureKind.UNKNOWN
    message = failure_message or "Unknown error"

    # 构建 CloudFailureInfo 以复用 build_cooldown_info 逻辑
    failure_info = CloudFailureInfo(
        kind=kind,
        message=message,
        retry_after=retry_after,
        retryable=retryable,
    )

    return build_cooldown_info(
        failure_info=failure_info,
        fallback_reason=fallback_reason,
        requested_mode=requested_mode,
        has_ampersand_prefix=has_ampersand_prefix,
        in_cooldown=in_cooldown,
        remaining_seconds=remaining_seconds,
        failure_count=failure_count,
        mode_source=mode_source,
    )


# ============================================================
# 便捷函数
# ============================================================


def resolve_requested_mode_for_decision(
    cli_execution_mode: Optional[str],
    has_ampersand_prefix: bool,
    config_execution_mode: Optional[str],
) -> Optional[str]:
    """确定用于执行决策的 requested_mode

    这是 run.py 和 scripts/run_iterate.py 共享的 requested_mode 判定逻辑的统一实现。
    确保两个入口对相同输入产生一致的 requested_mode。

    优先级规则（从高到低）：
    1. CLI 显式设置（--execution-mode 参数）
    2. & 前缀触发时传 None，让 build_execution_decision 内部处理路由
    3. config.yaml 的 execution_mode 设置

    Args:
        cli_execution_mode: CLI 显式设置的执行模式（--execution-mode 参数）
            - None: 用户未通过 CLI 指定
            - "cli"/"cloud"/"auto"/"plan"/"ask": 用户显式指定
        has_ampersand_prefix: 原始 prompt 是否以 '&' 开头（语法检测）
            - True: 存在 & 前缀，返回 None 让 build_execution_decision 处理
            - False: 无 & 前缀，使用 config.yaml 的值
        config_execution_mode: config.yaml 中的 cloud_agent.execution_mode 设置
            - 通常为 "cli"/"cloud"/"auto"，默认为 "auto"（config.yaml 默认值）

    Returns:
        requested_mode: 传递给 build_execution_decision 的模式参数
        - str: CLI 显式设置或 config.yaml 的值
        - None: 存在 & 前缀且无 CLI 显式设置，由 build_execution_decision 决策

    Examples:
        >>> # CLI 显式设置，优先级最高
        >>> resolve_requested_mode_for_decision("cloud", False, "cli")
        'cloud'
        >>> resolve_requested_mode_for_decision("cloud", True, "cli")
        'cloud'

        >>> # 无 CLI 设置，有 & 前缀，返回 None
        >>> resolve_requested_mode_for_decision(None, True, "cli")
        None

        >>> # 无 CLI 设置，无 & 前缀，使用 config.yaml
        >>> resolve_requested_mode_for_decision(None, False, "auto")
        'auto'

        >>> # 边界情况：config_execution_mode 为 None，返回 DEFAULT_EXECUTION_MODE
        >>> resolve_requested_mode_for_decision(None, False, None)
        'auto'  # DEFAULT_EXECUTION_MODE

    Note:
        此函数与 build_execution_decision 配合使用。调用顺序：
        1. 检测 & 前缀: has_ampersand_prefix = is_cloud_request(prompt)
        2. 确定 requested_mode: requested_mode = resolve_requested_mode_for_decision(...)
        3. 构建决策: decision = build_execution_decision(prompt, requested_mode, ...)

    Invariant:
        当 has_ampersand_prefix=False 且 cli_execution_mode=None 时，
        返回值**必须非 None**（为 config_execution_mode 或 DEFAULT_EXECUTION_MODE）。
        此不变式确保后续决策逻辑的一致性。
    """
    if cli_execution_mode is not None:
        # CLI 显式设置，优先级最高
        logger.debug(
            "[resolve_requested_mode_for_decision] CLI 显式设置: %s",
            cli_execution_mode,
        )
        return cli_execution_mode
    elif has_ampersand_prefix:
        # 有 & 前缀，传 None 让 build_execution_decision 处理路由
        logger.debug("[resolve_requested_mode_for_decision] 有 & 前缀，返回 None 让 build_execution_decision 处理")
        return None
    else:
        # 没有 CLI 设置也没有 & 前缀，使用 config.yaml 的值或 DEFAULT_EXECUTION_MODE
        # 当 has_ampersand_prefix=False 且 cli_execution_mode=None 时，
        # 必须返回非 None 值，以保证后续决策逻辑的一致性
        effective_mode = config_execution_mode or DEFAULT_EXECUTION_MODE
        if config_execution_mode is None:
            logger.debug(
                "[resolve_requested_mode_for_decision] 无 & 前缀且无 CLI 设置，"
                "config_execution_mode=None，使用 DEFAULT_EXECUTION_MODE=%s",
                DEFAULT_EXECUTION_MODE,
            )
        else:
            logger.debug(
                "[resolve_requested_mode_for_decision] 无 & 前缀，使用 config.yaml: %s",
                config_execution_mode,
            )
        return effective_mode


def resolve_mode_source(
    cli_execution_mode: Optional[str],
    has_ampersand_prefix: bool,
    requested_mode_for_decision: Optional[str],
) -> Optional[str]:
    """确定 mode_source（execution_mode 的来源）

    这是 run.py 和 scripts/run_iterate.py 共享的 mode_source 判定逻辑的统一实现。
    mode_source 用于决定消息级别（CLI 显式设置时使用 warning，config 来源使用 info）。

    Args:
        cli_execution_mode: CLI 显式设置的执行模式（--execution-mode 参数）
            - None: 用户未通过 CLI 指定
            - "cli"/"cloud"/"auto"/"plan"/"ask": 用户显式指定
        has_ampersand_prefix: 原始 prompt 是否以 '&' 开头（语法检测）
        requested_mode_for_decision: resolve_requested_mode_for_decision 的返回值
            - 传给 build_execution_decision 的 requested_mode 参数

    Returns:
        mode_source: execution_mode 的来源
        - "cli": 用户通过 --execution-mode 显式指定
        - "config": 来自 config.yaml 配置（无 CLI 显式设置，无 & 前缀）
        - None: & 前缀触发或默认值

    Examples:
        >>> # CLI 显式设置
        >>> resolve_mode_source("cloud", False, "cloud")
        'cli'
        >>> resolve_mode_source("auto", True, "auto")
        'cli'

        >>> # 无 CLI 设置，无 & 前缀，来自 config.yaml
        >>> resolve_mode_source(None, False, "auto")
        'config'

        >>> # 有 & 前缀，无 CLI 设置
        >>> resolve_mode_source(None, True, None)
        None
    """
    if cli_execution_mode is not None:
        return "cli"
    elif requested_mode_for_decision is not None and not has_ampersand_prefix:
        # 无 CLI 显式设置，无 & 前缀，requested_mode 来自 config.yaml
        return "config"
    else:
        return None  # & 前缀触发或默认值


def validate_requested_mode_invariant(
    has_ampersand_prefix: bool,
    cli_execution_mode: Optional[str],
    requested_mode_for_decision: Optional[str],
    config_execution_mode: Optional[str],
    caller_name: str = "unknown",
    *,
    raise_on_violation: bool = False,
) -> None:
    """验证 requested_mode_for_decision 的不变式

    当无 & 前缀且 CLI 未指定时，requested_mode_for_decision 不应为 None。
    这是一个防御性断言，确保 should_use_mp_orchestrator 接收的是经过正确解析的 requested_mode。

    此函数集中化了 run.py 和 scripts/run_iterate.py 中的断言/日志策略，
    避免入口分叉导致的不一致。

    Args:
        has_ampersand_prefix: 原始 prompt 是否以 '&' 开头（语法检测）
        cli_execution_mode: CLI 显式设置的执行模式
        requested_mode_for_decision: resolve_requested_mode_for_decision 的返回值
        config_execution_mode: config.yaml 中的 cloud_agent.execution_mode 设置
        caller_name: 调用方标识（用于日志，如 "run.py" 或 "SelfIterator.__init__"）
        raise_on_violation: 是否在违反不变式时抛出异常（测试环境建议为 True）

    Raises:
        ValueError: 当 raise_on_violation=True 且不变式被违反时

    Note:
        不变式规则：
        - has_ampersand_prefix=False 且 cli_execution_mode=None 时
        - requested_mode_for_decision 不应为 None（应为 config 值或 DEFAULT_EXECUTION_MODE）
        - resolve_requested_mode_for_decision 已保证此条件，此函数作为双重防御

    Warning:
        此不变式在 resolve_requested_mode_for_decision 修改后应始终满足。
        如果触发警告，说明存在逻辑错误或函数调用顺序问题。
    """
    if not has_ampersand_prefix and cli_execution_mode is None and requested_mode_for_decision is None:
        error_msg = (
            f"[{caller_name}] 不变式违反: 无 & 前缀且 CLI 未指定，"
            f"但 requested_mode=None（config.cloud_agent.execution_mode={config_execution_mode}，"
            f"DEFAULT_EXECUTION_MODE={DEFAULT_EXECUTION_MODE}）。"
            f"这表明 resolve_requested_mode_for_decision 未被正确调用或存在逻辑错误。"
        )
        logger.warning(error_msg)
        if raise_on_violation:
            raise ValueError(error_msg)


def is_cloud_mode(mode: Optional[str]) -> bool:
    """判断是否为 Cloud 相关模式

    Args:
        mode: 执行模式字符串

    Returns:
        是否为 cloud 或 auto 模式
    """
    if not mode:
        return False
    mode_lower = mode.lower()
    return mode_lower in ("cloud", "auto")


def is_readonly_mode(mode: Optional[str]) -> bool:
    """判断是否为只读模式

    Args:
        mode: 执行模式字符串

    Returns:
        是否为 plan 或 ask 模式
    """
    if not mode:
        return False
    mode_lower = mode.lower()
    return mode_lower in ("plan", "ask")


def get_fallback_mode(mode: Optional[str]) -> str:
    """获取回退模式

    Args:
        mode: 当前执行模式

    Returns:
        回退模式（cloud/auto 回退到 cli，其他保持不变）
    """
    if is_cloud_mode(mode):
        return "cli"
    return mode or "cli"


def should_use_mp_orchestrator(requested_mode: Optional[str]) -> bool:
    """判断是否可以使用 MP 编排器

    基于 **requested_mode**（用户请求的执行模式）判断，不受 API Key/cloud_enabled 影响。

    规则：
    - requested_mode=auto/cloud: 强制使用 basic 编排器（不支持 MP）
    - requested_mode=cli/plan/ask/None: 允许使用 MP 编排器

    这与 CLI help 对齐：用户请求 auto/cloud 即强制 basic，不管最终 effective_mode
    是否因为缺少 API Key 等原因回退到 CLI。

    ================================================================================
    重要：输入参数约定
    ================================================================================

    **requested_mode 必须是经过 resolve_requested_mode_for_decision() 解析后的值**

    调用方应遵循以下流程：
    1. 检测 & 前缀: has_ampersand_prefix = is_cloud_request(prompt)
    2. 解析 requested_mode: requested_mode = resolve_requested_mode_for_decision(...)
    3. 调用本函数: can_use_mp = should_use_mp_orchestrator(requested_mode)

    **不得直接传入未经解析的 None**，除非确实是 & 前缀场景：
    - 当 has_ampersand_prefix=True 且无 CLI 显式设置时，resolve_requested_mode_for_decision
      返回 None（让 build_execution_decision 处理 & 前缀路由），此时 None 是**预期值**
    - 当 has_ampersand_prefix=False 且无 CLI 显式设置时，resolve_requested_mode_for_decision
      返回 config.yaml 的值（如 "auto"），**不应为 None**

    错误使用示例（会导致编排器选择不一致）：
        # ✗ 错误：直接传入 None
        should_use_mp_orchestrator(None)  # 返回 True，但实际应使用 config.yaml 默认 auto

    正确使用示例：
        # ✓ 正确：先解析 requested_mode
        requested_mode = resolve_requested_mode_for_decision(cli_mode, has_prefix, config_mode)
        can_use_mp = should_use_mp_orchestrator(requested_mode)

    Args:
        requested_mode: 用户请求的执行模式（cli/auto/cloud/plan/ask 或 None）
            注意：这是 requested_mode，不是 effective_mode
            **必须是 resolve_requested_mode_for_decision() 的返回值**

    Returns:
        是否可以使用 MP 编排器

    See Also:
        - resolve_requested_mode_for_decision: 解析 requested_mode 的统一入口
        - build_execution_decision: 构建完整的执行决策
    """
    # Cloud/Auto 模式强制使用 basic 编排器，不管最终是否回退到 CLI
    return not is_cloud_mode(requested_mode)


# ============================================================
# 策略上下文 - 用于组合多个决策
# ============================================================


@dataclass
class ExecutionPolicyContext:
    """执行策略上下文

    封装执行决策所需的所有上下文信息，提供统一的决策接口。

    ================================================================================
    & 前缀属性语义说明（重要：内部条件分支规范）
    ================================================================================

    本类提供两个语义明确的属性用于 & 前缀相关的判断：

    1. has_ampersand_prefix (语法检测层面)
       - 定义：原始 prompt 文本是否以 '&' 开头
       - 来源：is_cloud_request(self.prompt) 的返回值
       - 用途：构建用户消息、日志记录、UI 显示
       - 示例：
           >>> ctx = ExecutionPolicyContext(prompt="& 分析代码")
           >>> ctx.has_ampersand_prefix
           True

    2. prefix_routed (策略决策层面) - **内部条件分支优先使用此字段**
       - 定义：& 前缀是否成功触发 Cloud 模式
       - 条件：满足以下全部条件时为 True:
           a. has_ampersand_prefix=True
           b. cloud_enabled=True
           c. has_api_key=True
           d. auto_detect_cloud_prefix=True
           e. 未显式指定 execution_mode=cli
       - 用途：决定执行模式、编排器选择等条件分支
       - 示例：
           >>> ctx = ExecutionPolicyContext(
           ...     prompt="& 分析代码",
           ...     cloud_enabled=True,
           ...     has_api_key=True
           ... )
           >>> ctx.prefix_routed
           True

    3. triggered_by_prefix (兼容别名) - **仅用于兼容输出，避免新代码引用**
       - 定义：prefix_routed 的别名（历史语义为 has_ampersand_prefix）
       - 注意：旧代码可能混用两种语义，新代码应避免使用此属性
       - 迁移：
           * 如需语法检测，请使用 has_ampersand_prefix
           * 如需策略决策，请使用 prefix_routed

    ================================================================================
    使用规范
    ================================================================================

    内部条件分支（决定执行行为）应使用 prefix_routed：
        if ctx.prefix_routed:
            # & 前缀成功触发 Cloud 模式
            effective_mode = "cloud"
            orchestrator = "basic"

    消息构建/日志记录应使用 has_ampersand_prefix：
        if ctx.has_ampersand_prefix:
            message += "（由 & 前缀触发）"
    """

    # 配置信息
    cloud_enabled: bool = False
    has_api_key: bool = False
    auto_detect_cloud_prefix: bool = True

    # 请求信息
    requested_mode: Optional[str] = None
    prompt: Optional[str] = None

    # 缓存的前缀信息
    _prefix_info: Optional[AmpersandPrefixInfo] = field(default=None, repr=False)

    # ========== 语义明确的属性（优先使用这些属性） ==========

    @property
    def has_ampersand_prefix(self) -> bool:
        """原始 prompt 是否包含 '&' 前缀（语法检测层面）

        用途：用于消息构建、日志记录、UI 显示等需要知道原始文本格式的场景。
        等价于 is_cloud_request(self.prompt)。

        注意：此属性仅表示语法层面是否存在 & 前缀，不表示是否成功触发 Cloud。
        如需判断是否成功触发 Cloud，请使用 prefix_routed 属性。
        """
        return is_cloud_request(self.prompt)

    @property
    def prefix_routed(self) -> bool:
        """& 前缀是否成功触发 Cloud 模式（策略决策层面）

        **内部条件分支应优先使用此属性**

        用途：用于决定执行模式、编排器选择等需要进行条件分支的场景。

        仅当满足以下全部条件时为 True：
        - has_ampersand_prefix=True（语法层面存在 & 前缀）
        - cloud_enabled=True（Cloud 功能已启用）
        - has_api_key=True（有有效 API Key）
        - auto_detect_cloud_prefix=True（未禁用自动检测）
        - 未显式指定 execution_mode=cli（CLI 模式忽略 & 前缀）

        示例：
            >>> ctx = ExecutionPolicyContext(
            ...     prompt="& 分析代码",
            ...     cloud_enabled=True,
            ...     has_api_key=False  # 无 API Key
            ... )
            >>> ctx.has_ampersand_prefix  # 语法层面存在
            True
            >>> ctx.prefix_routed  # 策略层面未成功触发
            False
        """
        return self._get_prefix_info().prefix_routed

    def _get_prefix_info(self) -> AmpersandPrefixInfo:
        """获取前缀信息（懒加载缓存）"""
        if self._prefix_info is None:
            self._prefix_info = detect_ampersand_prefix(
                prompt=self.prompt,
                requested_mode=self.requested_mode,
                cloud_enabled=self.cloud_enabled,
                has_api_key=self.has_api_key,
                auto_detect_cloud_prefix=self.auto_detect_cloud_prefix,
            )
        return self._prefix_info

    # ========== 兼容旧代码的属性（仅用于兼容输出，避免新代码引用） ==========

    @property
    def triggered_by_prefix(self) -> bool:
        """[DEPRECATED] 兼容别名 - 新代码请使用 prefix_routed

        ================================================================================
        语义统一说明（迁移完成）
        ================================================================================

        此属性现在返回 prefix_routed（策略决策层面，& 前缀是否成功触发 Cloud）。

        **语义变更历史**：
        - 旧实现返回 has_ampersand_prefix（语法检测），导致与其他模块的
          triggered_by_prefix=prefix_routed 语义冲突
        - 现已统一为 prefix_routed 语义，确保全仓 triggered_by_prefix 语义一致

        **新代码规范**：
        - 需要语法检测（原始文本是否有 & 前缀）→ has_ampersand_prefix
        - 需要策略决策（是否成功触发 Cloud）→ prefix_routed
        - triggered_by_prefix 仅作为 prefix_routed 的兼容别名，避免新代码引用

        ================================================================================
        注意：此属性返回 prefix_routed，用于兼容输出（如 to_dict）
        ================================================================================
        """
        return self.prefix_routed

    # ========== 决策方法 ==========

    def resolve_mode(self) -> tuple[str, str]:
        """解析实际执行模式"""
        return resolve_effective_execution_mode(
            requested_mode=self.requested_mode,
            has_ampersand_prefix=self.has_ampersand_prefix,  # 语法检测层面
            cloud_enabled=self.cloud_enabled,
            has_api_key=self.has_api_key,
        )

    def should_route_to_cloud(self) -> bool:
        """判断是否应该路由到 Cloud

        等价于 prefix_routed（策略决策层面）。
        """
        return self.prefix_routed

    def get_sanitized_prompt(self) -> str:
        """获取清理后的 prompt"""
        return sanitize_prompt_for_cli_fallback(self.prompt)

    def build_fallback_message(
        self,
        failure_info: CloudFailureInfo,
    ) -> str:
        """构建回退消息"""
        return build_user_facing_fallback_message(
            kind=failure_info.kind,
            retry_after=failure_info.retry_after,
            requested_mode=self.requested_mode,
            has_ampersand_prefix=self.has_ampersand_prefix,  # 语法检测层面
        )


# ============================================================
# 执行决策结果 - 统一封装所有执行决策信息
# ============================================================


@dataclass
class ExecutionDecision:
    """执行决策结果

    封装 build_execution_decision 的完整输出，包含：
    - 有效执行模式 (effective_mode)
    - 编排器类型 (orchestrator)
    - & 前缀状态：has_ampersand_prefix（语法检测）和 prefix_routed（策略决策）
    - 清理后的 prompt (sanitized_prompt)
    - 用户提示消息 (user_message) - 仅构建，不打印
    - 消息级别 (message_level) - 控制打印时使用 warning 还是 info

    这是执行决策的"快照"结构，用于确保 run.py 和 scripts/run_iterate.py
    两个入口对相同输入产生一致的决策结果。

    ================================================================================
    消息级别策略 (message_level)
    ================================================================================

    用于控制入口脚本打印 user_message 时的级别选择：

    | mode_source | 场景 | message_level | 说明 |
    |-------------|------|---------------|------|
    | "cli" | 用户显式 --execution-mode auto/cloud | "warning" | 用户显式请求，明确提示回退 |
    | "config" | config.yaml 设置 execution_mode=auto | "info" | 避免"每次都警告"的问题 |
    | None | 使用默认值 | "info" | 信息提示而非警告 |

    入口脚本根据 message_level 决定使用 print_warning 还是 print_info。

    ================================================================================
    & 前缀字段语义说明（重要：内部条件分支规范）
    ================================================================================

    1. has_ampersand_prefix (语法检测层面)
       - 定义：原始 prompt 文本是否以 '&' 开头
       - 用途：消息构建、日志记录、UI 显示

    2. prefix_routed (策略决策层面) - **内部条件分支优先使用此字段**
       - 定义：& 前缀是否成功触发 Cloud 模式
       - 用途：决定执行模式、编排器选择等条件分支

    3. triggered_by_prefix (兼容别名) - **仅用于兼容输出，避免新代码引用**
       - 定义：prefix_routed 的别名
       - 注意：仅用于 to_dict() 等兼容输出场景

    ================================================================================
    示例场景对比
    ================================================================================

    关键区分：& 前缀 prefix_routed=False 时有两种子状态：
    - **未成功路由**：auto_detect 开启但缺少条件（无 API Key 或 cloud_disabled），
      & 前缀仍表达 Cloud 意图 → **强制 basic**
    - **被忽略**：显式 cli 模式或 auto_detect=false，& 前缀被忽略 → **允许 mp**

    | 场景 | has_ampersand_prefix | prefix_routed | effective_mode | orchestrator | 说明 |
    |------|----------------------|---------------|----------------|--------------|------|
    | 无 & 前缀 | False | False | cli | mp | 函数级默认 |
    | & + 成功路由 | True | True | cloud | basic | R-2 |
    | & + 无 API Key | True | False | cli | **basic** | R-2: 未成功路由但仍表达意图 |
    | & + cloud_disabled | True | False | cli | **basic** | R-2: 未成功路由但仍表达意图 |
    | & + CLI 模式 | True | False | cli | mp | R-3: 显式 cli 忽略前缀 |
    | & + auto_detect=false | True | False | cli | mp | R-3: 禁用检测忽略前缀 |
    """

    # === 核心决策 ===
    effective_mode: str  # 有效执行模式: cli/cloud/auto/plan/ask
    orchestrator: str  # 编排器类型: mp/basic

    # === & 前缀状态（语义明确的字段） ===
    has_ampersand_prefix: bool = False  # 语法检测：原始 prompt 是否有 & 前缀
    prefix_routed: bool = False  # 策略决策：& 是否成功触发 Cloud 模式

    # === 请求信息 ===
    requested_mode: Optional[str] = None  # 原始请求模式
    original_prompt: Optional[str] = None  # 原始 prompt
    sanitized_prompt: str = ""  # 清理后的 prompt（移除 & 前缀）

    # === 决策原因 ===
    mode_reason: str = ""  # 执行模式决策原因
    orchestrator_reason: str = ""  # 编排器选择原因

    # === 用户提示消息 ===
    user_message: Optional[str] = None  # 用户友好消息（仅构建，不打印）
    message_level: str = "info"  # 消息级别: "warning" 或 "info"

    # === 配置来源 ===
    mode_source: Optional[str] = None  # 执行模式来源: "cli"/"config"/None

    # === & 前缀详细状态（可选） ===
    ampersand_prefix_info: Optional[AmpersandPrefixInfo] = None

    # ========== 兼容旧代码的属性（仅用于兼容输出，避免新代码引用） ==========

    @property
    def triggered_by_prefix(self) -> bool:
        """[DEPRECATED] 兼容别名 - 新代码请使用 prefix_routed

        ================================================================================
        迁移指南
        ================================================================================

        此属性保留以兼容旧代码输出（如 to_dict），语义等同于 prefix_routed。
        表示"& 前缀是否成功触发 Cloud 模式"。

        新代码应明确使用语义明确的属性：
        - 需要语法检测（原始文本是否有 & 前缀）→ has_ampersand_prefix
        - 需要策略决策（是否成功触发 Cloud）→ prefix_routed

        ================================================================================
        注意：仅作为兼容别名用于输出，内部条件分支请使用 prefix_routed
        ================================================================================
        """
        return self.prefix_routed

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "effective_mode": self.effective_mode,
            "orchestrator": self.orchestrator,
            # 新字段（语义明确）
            "has_ampersand_prefix": self.has_ampersand_prefix,
            "prefix_routed": self.prefix_routed,
            # 兼容字段
            "triggered_by_prefix": self.triggered_by_prefix,
            "requested_mode": self.requested_mode,
            "sanitized_prompt": self.sanitized_prompt,
            "mode_reason": self.mode_reason,
            "orchestrator_reason": self.orchestrator_reason,
            "user_message": self.user_message,
            "message_level": self.message_level,
            "mode_source": self.mode_source,
        }


# ============================================================
# 执行决策矩阵测试用例 - 驱动一致性测试
# ============================================================


@dataclass
class DecisionMatrixCase:
    """执行决策矩阵测试用例

    封装 resolve_requested_mode_for_decision 和 build_execution_decision 的
    输入参数与期望输出，用于驱动一致性测试。

    本结构与 tests/test_iterate_execution_matrix_consistency.py 中的
    DecisionSnapshot/SnapshotTestCase 对齐，遵循"统一字段 Schema"。

    ================================================================================
    字段说明（与统一字段 Schema 对齐）
    ================================================================================

    【推导输入】传给 resolve_requested_mode_for_decision 的参数：
    - cli_execution_mode: CLI --execution-mode 原始参数值
    - has_ampersand_prefix: 语法检测层面，原始 prompt 是否有 & 前缀
    - config_execution_mode: config.yaml 中的 cloud_agent.execution_mode

    【build_execution_decision 输入】直接传给 build_execution_decision 的参数：
    - has_api_key: 是否配置了有效的 API Key
    - cloud_enabled: config.yaml 中的 cloud_agent.enabled 设置
    - auto_detect_cloud_prefix: 是否启用 & 前缀自动检测

    【期望输出】断言维度：
    - expected_requested_mode_for_decision: resolve_requested_mode_for_decision 的输出
    - expected_effective_mode: 有效执行模式（经过路由决策后）
    - expected_orchestrator: 编排器类型 (mp/basic)
    - expected_prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud

    【规则标识】
    - applicable_rules: 适用的规则（R-1/R-2/R-3/plan&ask 忽略等）

    ================================================================================
    适用规则快速索引（与 build_execution_decision docstring 对齐）
    ================================================================================

    | 规则 ID | 简述                                                           |
    |---------|----------------------------------------------------------------|
    | R-1     | requested=auto/cloud → 强制 basic                              |
    | R-2     | & 前缀表达 Cloud 意图（未忽略时）→ 强制 basic（即使未成功路由）    |
    | R-3     | auto_detect=false 或显式 cli/plan/ask → 忽略 & 前缀，允许 mp    |
    | R-4     | user_message 仅构建不打印，由调用方决定输出                      |

    Attributes:
        case_id: 用例唯一标识符（用于测试参数化 ids）
        description: 用例描述（用于测试失败时的诊断信息）
        cli_execution_mode: CLI --execution-mode 原始参数值
        has_ampersand_prefix: 语法检测层面，原始 prompt 是否有 & 前缀
        config_execution_mode: config.yaml 中的 cloud_agent.execution_mode
        has_api_key: 是否配置了有效的 API Key
        cloud_enabled: config.yaml 中的 cloud_agent.enabled 设置
        auto_detect_cloud_prefix: 是否启用 & 前缀自动检测
        expected_requested_mode_for_decision: 期望的 requested_mode_for_decision 值
        expected_effective_mode: 期望的有效执行模式
        expected_orchestrator: 期望的编排器类型
        expected_prefix_routed: 期望的 & 前缀路由结果
        applicable_rules: 适用的规则标识（用于文档和调试）
    """

    # === 用例标识 ===
    case_id: str
    description: str

    # === 推导输入（传给 resolve_requested_mode_for_decision）===
    cli_execution_mode: Optional[str]  # CLI --execution-mode 参数
    has_ampersand_prefix: bool  # 语法检测：原始 prompt 是否有 & 前缀
    config_execution_mode: Optional[str]  # config.yaml 中的 cloud_agent.execution_mode

    # === build_execution_decision 的其他输入 ===
    has_api_key: bool
    cloud_enabled: bool
    auto_detect_cloud_prefix: bool = True

    # === 期望输出 ===
    expected_requested_mode_for_decision: Optional[str] = None  # resolve_requested_mode_for_decision 的输出
    expected_effective_mode: str = "cli"  # 有效执行模式
    expected_orchestrator: str = "mp"  # 编排器类型
    expected_prefix_routed: bool = False  # 策略决策：& 前缀是否成功触发 Cloud

    # === 规则标识 ===
    applicable_rules: str = ""  # R-1/R-2/R-3 等规则标识


# ============================================================
# 执行决策矩阵 - 覆盖 R-1/R-2/R-3/plan&ask 忽略/auto_detect=false 等关键边界
# ============================================================

EXECUTION_DECISION_MATRIX_CASES: list[DecisionMatrixCase] = [
    # ===================================================================
    # R-1: requested=auto/cloud → 强制 basic（即使无 API Key 回退到 CLI）
    # ===================================================================
    # --- CLI 模式（允许 mp）---
    DecisionMatrixCase(
        case_id="cli_explicit_allows_mp",
        description="显式 --execution-mode cli，允许 MP 编排器",
        cli_execution_mode="cli",
        has_ampersand_prefix=False,
        config_execution_mode="auto",  # config 默认 auto，但 CLI 显式覆盖
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cli",
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        applicable_rules="R-1（显式 cli 不受影响）",
    ),
    DecisionMatrixCase(
        case_id="cli_no_key_allows_mp",
        description="显式 --execution-mode cli 无 API Key，允许 MP",
        cli_execution_mode="cli",
        has_ampersand_prefix=False,
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cli",
        expected_effective_mode="cli",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        applicable_rules="R-1",
    ),
    # --- AUTO 模式（强制 basic）---
    DecisionMatrixCase(
        case_id="auto_with_key_forces_basic",
        description="AUTO 模式有 API Key，强制 basic 编排器",
        cli_execution_mode="auto",
        has_ampersand_prefix=False,
        config_execution_mode="cli",  # config 是 cli，但 CLI 显式 auto
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",
        expected_effective_mode="auto",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        applicable_rules="R-1",
    ),
    DecisionMatrixCase(
        case_id="auto_no_key_forces_basic_key_scenario",
        description="关键场景：AUTO 模式无 API Key，回退 CLI 但仍强制 basic",
        cli_execution_mode="auto",
        has_ampersand_prefix=False,
        config_execution_mode="cli",
        has_api_key=False,  # 无 API Key 导致回退
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 但仍强制 basic（基于 requested_mode）
        expected_prefix_routed=False,
        applicable_rules="R-1（关键场景：回退不影响编排器）",
    ),
    # 注意：resolve_effective_execution_mode 在 requested_mode="auto" 时
    # 只检查 has_api_key，不检查 cloud_enabled。cloud_enabled 仅影响 & 前缀路由。
    # 因此 auto + has_api_key=True + cloud_enabled=False → effective_mode="auto"
    DecisionMatrixCase(
        case_id="auto_cloud_disabled_still_auto",
        description="AUTO 模式 cloud_enabled=False，有 API Key 仍使用 auto",
        cli_execution_mode="auto",
        has_ampersand_prefix=False,
        config_execution_mode="cli",
        has_api_key=True,
        cloud_enabled=False,  # cloud_enabled=False 不影响显式 auto 模式
        expected_requested_mode_for_decision="auto",
        expected_effective_mode="auto",  # 有 API Key，使用 auto
        expected_orchestrator="basic",  # auto 强制 basic
        expected_prefix_routed=False,
        applicable_rules="R-1（cloud_enabled 不影响显式 auto 模式）",
    ),
    # --- CLOUD 模式（强制 basic）---
    DecisionMatrixCase(
        case_id="cloud_with_key_forces_basic",
        description="CLOUD 模式有 API Key，强制 basic 编排器",
        cli_execution_mode="cloud",
        has_ampersand_prefix=False,
        config_execution_mode="cli",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cloud",
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=False,
        applicable_rules="R-1",
    ),
    DecisionMatrixCase(
        case_id="cloud_no_key_forces_basic_key_scenario",
        description="关键场景：CLOUD 模式无 API Key，回退 CLI 但仍强制 basic",
        cli_execution_mode="cloud",
        has_ampersand_prefix=False,
        config_execution_mode="cli",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision="cloud",
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 但仍强制 basic
        expected_prefix_routed=False,
        applicable_rules="R-1（关键场景：回退不影响编排器）",
    ),
    # ===================================================================
    # R-2: & 前缀表达 Cloud 意图时强制 basic（即使未成功路由）
    # ===================================================================
    DecisionMatrixCase(
        case_id="ampersand_routed_success",
        description="& 前缀成功路由到 Cloud",
        cli_execution_mode=None,  # 无显式 CLI 参数
        has_ampersand_prefix=True,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        auto_detect_cloud_prefix=True,
        expected_requested_mode_for_decision=None,  # & 前缀场景返回 None
        expected_effective_mode="cloud",
        expected_orchestrator="basic",
        expected_prefix_routed=True,
        applicable_rules="R-2",
    ),
    DecisionMatrixCase(
        case_id="ampersand_no_key_still_basic",
        description="关键场景：& 前缀无 API Key，未成功路由但仍强制 basic",
        cli_execution_mode=None,
        has_ampersand_prefix=True,
        config_execution_mode="auto",
        has_api_key=False,  # 无 API Key 导致未成功路由
        cloud_enabled=True,
        auto_detect_cloud_prefix=True,
        expected_requested_mode_for_decision=None,
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 但仍强制 basic（& 前缀表达了 Cloud 意图）
        expected_prefix_routed=False,
        applicable_rules="R-2（关键场景：未成功路由但仍表达意图）",
    ),
    DecisionMatrixCase(
        case_id="ampersand_cloud_disabled_still_basic",
        description="关键场景：& 前缀 cloud_disabled，未成功路由但仍强制 basic",
        cli_execution_mode=None,
        has_ampersand_prefix=True,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=False,  # cloud_enabled=False 导致未成功路由
        auto_detect_cloud_prefix=True,
        expected_requested_mode_for_decision=None,
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 但仍强制 basic
        expected_prefix_routed=False,
        applicable_rules="R-2（关键场景：未成功路由但仍表达意图）",
    ),
    # ===================================================================
    # R-3: auto_detect=false 或显式 cli/plan/ask → 忽略 & 前缀，允许 mp
    # ===================================================================
    DecisionMatrixCase(
        case_id="ampersand_cli_explicit_ignores_prefix",
        description="R-3: 显式 --execution-mode cli 忽略 & 前缀，允许 MP",
        cli_execution_mode="cli",
        has_ampersand_prefix=True,  # 有 & 前缀
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        auto_detect_cloud_prefix=True,
        expected_requested_mode_for_decision="cli",  # CLI 显式覆盖
        expected_effective_mode="cli",
        expected_orchestrator="mp",  # 允许 MP（& 前缀被忽略）
        expected_prefix_routed=False,
        applicable_rules="R-3（显式 cli 忽略 & 前缀）",
    ),
    # 注意：当 auto_detect_cloud_prefix=False 时，& 前缀被忽略，
    # 但 requested_mode 仍由 config_execution_mode 决定。
    # 如果 config_execution_mode="auto"，R-1 仍生效（auto 强制 basic）。
    # 为了验证 R-3（& 前缀被忽略 → 允许 mp），需要 config_execution_mode="cli"。
    DecisionMatrixCase(
        case_id="ampersand_auto_detect_false_ignores_prefix",
        description="R-3: auto_detect_cloud_prefix=False 忽略 & 前缀，config=cli 允许 MP",
        cli_execution_mode=None,
        has_ampersand_prefix=True,
        config_execution_mode="cli",  # config=cli 以验证 R-3
        has_api_key=True,
        cloud_enabled=True,
        auto_detect_cloud_prefix=False,  # 禁用自动检测
        expected_requested_mode_for_decision=None,  # & 前缀场景返回 None
        expected_effective_mode="cli",  # 使用 config 默认的 cli
        expected_orchestrator="mp",  # 允许 MP（& 前缀被忽略，config=cli）
        expected_prefix_routed=False,
        applicable_rules="R-3（auto_detect=false 忽略 & 前缀）",
    ),
    # ===================================================================
    # PLAN/ASK 只读模式：& 前缀被忽略，允许 mp
    # ===================================================================
    DecisionMatrixCase(
        case_id="plan_mode_allows_mp",
        description="PLAN 模式（只读）允许 MP",
        cli_execution_mode="plan",
        has_ampersand_prefix=False,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="plan",
        expected_effective_mode="plan",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        applicable_rules="plan/ask 只读模式",
    ),
    DecisionMatrixCase(
        case_id="ask_mode_allows_mp",
        description="ASK 模式（只读）允许 MP",
        cli_execution_mode="ask",
        has_ampersand_prefix=False,
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="ask",
        expected_effective_mode="ask",
        expected_orchestrator="mp",
        expected_prefix_routed=False,
        applicable_rules="plan/ask 只读模式",
    ),
    DecisionMatrixCase(
        case_id="plan_mode_ignores_ampersand",
        description="R-3: PLAN 模式忽略 & 前缀，允许 MP",
        cli_execution_mode="plan",
        has_ampersand_prefix=True,  # 有 & 前缀
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="plan",
        expected_effective_mode="plan",
        expected_orchestrator="mp",  # 允许 MP（& 前缀被忽略）
        expected_prefix_routed=False,
        applicable_rules="R-3（plan 只读模式忽略 & 前缀）",
    ),
    DecisionMatrixCase(
        case_id="ask_mode_ignores_ampersand",
        description="R-3: ASK 模式忽略 & 前缀，允许 MP",
        cli_execution_mode="ask",
        has_ampersand_prefix=True,  # 有 & 前缀
        config_execution_mode="auto",
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="ask",
        expected_effective_mode="ask",
        expected_orchestrator="mp",  # 允许 MP（& 前缀被忽略）
        expected_prefix_routed=False,
        applicable_rules="R-3（ask 只读模式忽略 & 前缀）",
    ),
    # ===================================================================
    # 函数级默认：无显式参数 + 无 & 前缀 + 无 config → 使用 DEFAULT_EXECUTION_MODE
    # ===================================================================
    # 注意：当 config_execution_mode=None 时，resolve_requested_mode_for_decision
    # 返回 DEFAULT_EXECUTION_MODE（即 "auto"），而非 None。
    # 这是因为 has_ampersand_prefix=False 且 cli_execution_mode=None 时，
    # 必须返回非 None 值以保证后续决策逻辑的一致性。
    DecisionMatrixCase(
        case_id="function_default_uses_default_mode",
        description="函数级默认：无显式参数，使用 DEFAULT_EXECUTION_MODE (auto)",
        cli_execution_mode=None,
        has_ampersand_prefix=False,
        config_execution_mode=None,  # 无 config 默认值
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",  # DEFAULT_EXECUTION_MODE
        expected_effective_mode="auto",  # auto 模式有 API Key
        expected_orchestrator="basic",  # auto 强制 basic
        expected_prefix_routed=False,
        applicable_rules="函数级默认（使用 DEFAULT_EXECUTION_MODE）",
    ),
    # ===================================================================
    # config.yaml 默认 auto 场景（系统级默认）
    # ===================================================================
    DecisionMatrixCase(
        case_id="config_default_auto_with_key",
        description="config.yaml 默认 auto + 有 API Key，使用 Cloud + basic",
        cli_execution_mode=None,  # 无 CLI 显式参数
        has_ampersand_prefix=False,
        config_execution_mode="auto",  # config 默认 auto
        has_api_key=True,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",
        expected_effective_mode="auto",
        expected_orchestrator="basic",  # auto 强制 basic
        expected_prefix_routed=False,
        applicable_rules="R-1（系统级默认 auto）",
    ),
    DecisionMatrixCase(
        case_id="config_default_auto_no_key",
        description="config.yaml 默认 auto + 无 API Key，回退 CLI 但仍强制 basic",
        cli_execution_mode=None,
        has_ampersand_prefix=False,
        config_execution_mode="auto",
        has_api_key=False,
        cloud_enabled=True,
        expected_requested_mode_for_decision="auto",
        expected_effective_mode="cli",  # 回退到 CLI
        expected_orchestrator="basic",  # 但仍强制 basic
        expected_prefix_routed=False,
        applicable_rules="R-1（系统级默认 auto，回退不影响编排器）",
    ),
]


def build_execution_decision(
    prompt: Optional[str],
    requested_mode: Optional[str],
    cloud_enabled: bool,
    has_api_key: bool,
    auto_detect_cloud_prefix: bool = True,
    user_requested_orchestrator: Optional[str] = None,
    mode_source: Optional[str] = None,
) -> ExecutionDecision:
    """构建执行决策

    这是执行模式和编排器选择的**统一决策入口**。所有入口脚本（run.py、
    scripts/run_iterate.py）应使用此函数获取一致的决策结果。

    本函数内部复用以下策略函数：
    - resolve_effective_execution_mode: 解析有效执行模式
    - should_use_mp_orchestrator: 判断是否可使用 MP 编排器
    - build_user_facing_fallback_message: 构建用户提示消息
    - sanitize_prompt_for_cli_fallback: 清理 prompt 移除 & 前缀

    ============================================================
    核心规则（与 CLI help 和测试矩阵一致）
    ============================================================

    【规则 R-1】requested=auto/cloud 强制 basic 编排器
        - 即使因为没有 API Key 导致 effective_mode 回退到 CLI
        - 只要 requested_mode 是 auto/cloud，编排器就应该是 basic
        - 这是与 CLI help 对齐的设计："请求 Cloud 即强制 basic"

    【规则 R-2】& 前缀表达 Cloud 意图时强制 basic（即使未成功路由）
        - 仅当满足以下**全部**条件时，& 前缀才成功触发 Cloud（prefix_routed=True）：
          1. auto_detect_cloud_prefix=True（未禁用自动检测）
          2. cloud_enabled=True（Cloud 功能已启用）
          3. has_api_key=True（有有效 API Key）
          4. 未显式指定 execution_mode=cli（CLI 模式忽略 & 前缀）
        - **重要**：当 auto_detect_cloud_prefix=True 时，& 前缀即表达 Cloud 意图：
          - 若成功路由（prefix_routed=True）→ effective_mode=cloud, orchestrator=basic
          - 若未成功路由（prefix_routed=False，如无 API Key 或 cloud_disabled）
            → effective_mode=cli, 但 **orchestrator 仍强制 basic**（意图未改变）
        - **对比**：当 auto_detect_cloud_prefix=False 时，& 前缀被完全忽略（见 R-3）

    【规则 R-3】& 前缀被忽略的条件（此时允许 mp）
        - 显式 --execution-mode cli（用户明确选择本地 CLI）
        - 显式 --execution-mode plan/ask（只读模式不参与 Cloud 路由）
        - auto_detect_cloud_prefix=False（用户禁用了 & 前缀自动检测）
        - 在上述情况下，& 前缀被忽略，不视为 Cloud 意图，编排器可使用 mp

    【规则 R-4】user_message 仅构建不打印
        - 当需要向用户解释决策时，构建 user_message 字符串
        - 打印职责由调用方（入口脚本）决定

    ============================================================
    可引用规则快速索引
    ============================================================

    | 规则 ID | 简述                                                           |
    |---------|----------------------------------------------------------------|
    | R-1     | requested=auto/cloud → 强制 basic                              |
    | R-2     | & 前缀表达 Cloud 意图（未忽略时）→ 强制 basic（即使未成功路由）    |
    | R-3     | auto_detect=false 或显式 cli/plan/ask → 忽略 & 前缀，允许 mp    |
    | R-4     | user_message 仅构建不打印，由调用方决定输出                      |

    ============================================================
    系统级默认 vs 函数级默认
    ============================================================

    【系统级默认】入口脚本无显式 --execution-mode 时读取 config.yaml 默认 auto，
    因此整体默认表现为 auto（无 key 时回退 cli）且编排器强制 basic。

    【函数级默认】requested_mode=None 且无 & 前缀 → effective_mode=cli, orchestrator=mp。
    但入口脚本通常传入 config.yaml 默认值（auto），所以函数级默认很少直接生效。

    ============================================================
    决策矩阵（与 tests/test_iterate_execution_matrix_consistency.py 一致）
    ============================================================

    列说明：
    - requested_mode: 传给本函数的 requested_mode 参数
      （测试文件中对应 requested_mode_for_decision，表示经过 resolve_requested_mode_for_decision 解析后的值）
    - has_ampersand_prefix: 语法检测层面，原始 prompt 是否有 & 前缀（隐含在 & 前缀列）
    - prefix_routed: 策略决策层面，& 前缀是否成功触发 Cloud
    - orchestrator 与 prefix_routed 标注的关系：
      - `False (未成功路由)`: auto_detect=true 但缺少条件（无 API Key 或 cloud_enabled=False），
        & 前缀**仍表达 Cloud 意图** → **强制 basic**（R-2）
      - `False (被忽略)`: 显式 cli 或 auto_detect=false，& 前缀**不视为 Cloud 意图**
        → **允许 mp**（R-3）
      - `False`: 无 & 前缀或不适用，orchestrator 由其他规则决定

    | requested_mode | has_api_key | cloud_enabled | & 前缀 | effective_mode | orchestrator | prefix_routed       | 适用规则 |
    |----------------|-------------|---------------|--------|----------------|--------------|---------------------|----------|
    | cli            | *           | *             | *      | cli            | mp           | False (被忽略)      | R-3      |
    | cli            | *           | *             | True   | cli            | mp           | False (被忽略)      | R-3      |
    | auto           | True        | True          | False  | auto           | basic        | False               | R-1      |
    | auto           | False       | *             | *      | cli (回退)     | basic        | False               | R-1      |
    | cloud          | True        | True          | False  | cloud          | basic        | False               | R-1      |
    | cloud          | False       | *             | *      | cli (回退)     | basic        | False               | R-1      |
    | None           | True        | True          | True   | cloud          | basic        | True                | R-2      |
    | None           | False       | True          | True   | cli            | basic        | False (未成功路由)  | R-2      |
    | None           | True        | False         | True   | cli            | basic        | False (未成功路由)  | R-2      |
    | None           | False       | False         | True   | cli            | basic        | False (未成功路由)  | R-2      |
    | None           | *           | *             | False  | cli            | mp           | False (函数级默认)  | -        |
    | None           | *           | *             | True   | cli            | mp           | False (被忽略)      | R-3 (*)  |
    | plan/ask       | *           | *             | False  | plan/ask       | mp           | False               | -        |
    | plan/ask       | *           | *             | True   | plan/ask       | mp           | False (被忽略)      | R-3 (**) |

    (*) R-3 忽略条件: 显式 --execution-mode cli（第1-2行）或 auto_detect_cloud_prefix=False（此行）
    (**) R-3 忽略条件: 显式 --execution-mode plan/ask（只读模式不参与 Cloud 路由）

    断言维度（测试矩阵应验证的字段）：
    - effective_mode: 验证经过决策后的有效执行模式
    - orchestrator: 验证编排器类型（mp/basic）
    - prefix_routed: 验证 & 前缀是否成功触发 Cloud
    - requested_mode_for_decision: 验证传给 build_execution_decision 的 requested_mode
    - has_ampersand_prefix: 验证语法层面的 & 前缀检测结果（可选，用于追溯）

    Args:
        prompt: 用户输入的任务 prompt（可能带 & 前缀）
        requested_mode: CLI 参数或 config.yaml 中的 execution_mode
            - None: 未显式指定
            - "cli": 强制 CLI 模式
            - "cloud": 强制 Cloud 模式
            - "auto": Cloud 优先，失败回退 CLI
            - "plan"/"ask": 只读模式
        cloud_enabled: config.yaml 中的 cloud_agent.enabled 设置
        has_api_key: 是否配置了有效的 API Key
        auto_detect_cloud_prefix: 是否启用 & 前缀自动检测（默认 True）
        user_requested_orchestrator: 用户显式指定的编排器类型（"mp"/"basic"）
            如果为 None，则按规则自动选择
        mode_source: execution_mode 的来源，用于决定消息级别
            - "cli": 来自 CLI 显式参数（--execution-mode）
            - "config": 来自 config.yaml 配置
            - None: 未指定（使用默认值）
            当 mode_source="cli" 且需要回退时，使用 warning 级别消息；
            否则使用 info 级别，避免"每次都警告"的问题。

    Returns:
        ExecutionDecision 包含完整的决策信息

    Examples:
        >>> # CLI 模式
        >>> decision = build_execution_decision(
        ...     prompt="实现功能",
        ...     requested_mode="cli",
        ...     cloud_enabled=True,
        ...     has_api_key=True,
        ... )
        >>> decision.effective_mode
        'cli'
        >>> decision.orchestrator
        'mp'
        >>> decision.triggered_by_prefix
        False

        >>> # & 前缀成功触发
        >>> decision = build_execution_decision(
        ...     prompt="& 分析代码",
        ...     requested_mode=None,
        ...     cloud_enabled=True,
        ...     has_api_key=True,
        ... )
        >>> decision.effective_mode
        'cloud'
        >>> decision.orchestrator
        'basic'
        >>> decision.triggered_by_prefix
        True
        >>> decision.sanitized_prompt
        '分析代码'

        >>> # AUTO 模式无 API Key（关键场景）
        >>> decision = build_execution_decision(
        ...     prompt="任务",
        ...     requested_mode="auto",
        ...     cloud_enabled=True,
        ...     has_api_key=False,
        ... )
        >>> decision.effective_mode
        'cli'  # 回退到 CLI
        >>> decision.orchestrator
        'basic'  # 但仍强制 basic（基于 requested_mode）

        >>> # & 前缀存在但未成功路由（R-2 关键场景：仍强制 basic）
        >>> decision = build_execution_decision(
        ...     prompt="& 分析代码",
        ...     requested_mode=None,
        ...     cloud_enabled=True,
        ...     has_api_key=False,  # 无 API Key 导致未成功路由
        ... )
        >>> decision.effective_mode
        'cli'  # 回退到 CLI
        >>> decision.orchestrator
        'basic'  # 仍强制 basic（& 前缀表达了 Cloud 意图，即使未成功路由）
        >>> decision.triggered_by_prefix
        False  # 未成功路由
    """
    # === Step 1: 检测 & 前缀并判断路由结果 ===
    # 使用统一的 detect_ampersand_prefix 函数，返回语义明确的状态
    prefix_info = detect_ampersand_prefix(
        prompt=prompt,
        requested_mode=requested_mode,
        cloud_enabled=cloud_enabled,
        has_api_key=has_api_key,
        auto_detect_cloud_prefix=auto_detect_cloud_prefix,
    )

    # 提取语义明确的字段
    has_ampersand_prefix = prefix_info.has_ampersand_prefix  # 语法检测
    prefix_routed = prefix_info.prefix_routed  # 策略决策

    # === Step 2: 解析有效执行模式 ===
    # 传入 has_ampersand_prefix（语法检测结果）给 resolve_effective_execution_mode
    # 该函数根据 cloud_enabled、has_api_key 决定是否路由到 Cloud
    # 注意：该函数不知道 auto_detect_cloud_prefix，所以需要后处理修正
    mode_lower = requested_mode.lower() if requested_mode else None
    effective_mode, mode_reason = resolve_effective_execution_mode(
        requested_mode=requested_mode,
        has_ampersand_prefix=has_ampersand_prefix,  # 使用语法检测结果
        cloud_enabled=cloud_enabled,
        has_api_key=has_api_key,
    )

    # === Step 2.1: 修正 - 当 & 前缀未成功路由时 ===
    # resolve_effective_execution_mode 不知道 auto_detect_cloud_prefix 等条件
    # 当 detect_ampersand_prefix 判定 prefix_routed=False 但函数返回 cloud 时需修正
    #
    # 重要例外：当 status == DETECTED_IGNORED_EXPLICIT_MODE 时，用户显式指定了
    # --execution-mode cloud/auto，此时 effective_mode 由用户显式设置决定，
    # 不应该因为 & 前缀的存在而修正为 CLI。
    if (
        has_ampersand_prefix
        and not prefix_routed
        and effective_mode == "cloud"
        and prefix_info.status != AmpersandPrefixStatus.DETECTED_IGNORED_EXPLICIT_MODE
    ):
        effective_mode = "cli"
        # 使用 detect_ampersand_prefix 提供的详细原因
        if prefix_info.status == AmpersandPrefixStatus.DETECTED_IGNORED_AUTO_DETECT_OFF:
            mode_reason = "& 前缀被忽略（auto_detect_cloud_prefix=False），使用 CLI"
        elif prefix_info.status == AmpersandPrefixStatus.DETECTED_IGNORED_DISABLED:
            mode_reason = "& 前缀被忽略（cloud_enabled=False），使用 CLI"
        elif prefix_info.status == AmpersandPrefixStatus.DETECTED_IGNORED_CLI_MODE:
            mode_reason = "显式指定 execution_mode=cli，忽略 & 前缀"
        else:
            mode_reason = f"& 前缀被忽略（{prefix_info.ignore_reason or '条件不满足'}），使用 CLI"
    elif (
        has_ampersand_prefix
        and not prefix_routed
        and prefix_info.status == AmpersandPrefixStatus.DETECTED_IGNORED_EXPLICIT_MODE
    ):
        # 显式模式：& 前缀存在但不影响执行模式（由用户显式设置决定）
        # effective_mode 保持不变，只更新 mode_reason
        mode_reason = f"显式指定 execution_mode={mode_lower}，& 前缀不触发路由"

    # === Step 3: 确定编排器类型 ===
    #
    # 编排器选择优先级（与 AGENTS.md 和 docstring 规则快速索引对齐）:
    #
    # 优先级 1 (R-1): Cloud/Auto 模式 → 强制 basic（即使用户显式指定 mp 也覆盖）
    # 优先级 2 (R-2): & 前缀表达 Cloud 意图（未忽略时）→ 强制 basic（即使未成功路由）
    # 优先级 3      : 用户显式指定 --orchestrator basic 或 --no-mp → basic
    # 优先级 4      : 用户显式指定 --orchestrator mp（仅在非 Cloud/Auto 模式下生效）
    # 优先级 5      : 默认 mp
    #
    # 例外 (R-3): 以下情况 & 前缀被忽略，不触发 R-2：
    #   - 显式 --execution-mode cli（用户明确选择本地 CLI）
    #   - auto_detect_cloud_prefix=False（用户禁用了 & 前缀自动检测）

    # R-1: Cloud/Auto 模式强制 basic
    # 这与 AGENTS.md 描述一致：
    # "当 --execution-mode 为 cloud 或 auto 时，系统会强制使用 basic 编排器，
    #  即使显式指定 --orchestrator mp 也会自动切换"
    force_basic_due_to_mode = not should_use_mp_orchestrator(requested_mode)

    # R-2 + R-3: & 前缀存在时强制 basic（除非被 R-3 忽略）
    # 根据 AGENTS.md：当 & 前缀存在但未成功路由时（prefix_routed=False），
    # 编排器选择仍基于用户意图（使用 Cloud），所以应该是 basic。
    # 但 R-3 定义的情况 & 前缀应该被忽略，不影响编排器选择：
    #   - 用户显式指定 --execution-mode cli（用户意图是使用本地 CLI）
    #   - 用户显式指定 --execution-mode plan/ask（只读模式不参与 Cloud 路由）
    #   - auto_detect_cloud_prefix=False（用户禁用了 & 前缀自动检测）
    mode_lower = (requested_mode or "").lower()
    user_explicitly_chose_cli = mode_lower == "cli"
    user_chose_readonly_mode = mode_lower in ("plan", "ask")
    auto_detect_disabled = prefix_info.status == AmpersandPrefixStatus.DETECTED_IGNORED_AUTO_DETECT_OFF
    should_ignore_prefix = user_explicitly_chose_cli or user_chose_readonly_mode or auto_detect_disabled
    force_basic_due_to_prefix = has_ampersand_prefix and not should_ignore_prefix

    if force_basic_due_to_mode or force_basic_due_to_prefix:
        orchestrator = "basic"
        if prefix_routed:
            orchestrator_reason = "& 前缀成功触发 Cloud 模式，强制使用 basic 编排器"
        elif has_ampersand_prefix:
            orchestrator_reason = "& 前缀表达 Cloud 意图（未成功路由），仍使用 basic 编排器"
        else:
            orchestrator_reason = f"requested_mode={mode_lower} 强制使用 basic 编排器"
    elif user_requested_orchestrator:
        # 规则 3: 非 Cloud/Auto 模式下，尊重用户显式指定
        orchestrator = user_requested_orchestrator
        orchestrator_reason = f"用户显式指定 orchestrator={orchestrator}"
    else:
        # 规则 4: 默认使用 mp
        orchestrator = "mp"
        # 如果 & 前缀存在但被忽略，需要在 reason 中体现
        if has_ampersand_prefix and should_ignore_prefix:
            if auto_detect_disabled:
                orchestrator_reason = "& 前缀被忽略（auto_detect_cloud_prefix=False），允许使用 MP 编排器"
            elif user_explicitly_chose_cli:
                orchestrator_reason = "& 前缀被忽略（显式 --execution-mode cli），允许使用 MP 编排器"
            elif user_chose_readonly_mode:
                orchestrator_reason = (
                    f"& 前缀被忽略（显式 --execution-mode {mode_lower}，只读模式），允许使用 MP 编排器"
                )
            else:
                orchestrator_reason = "& 前缀被忽略，允许使用 MP 编排器"
        else:
            orchestrator_reason = f"requested_mode={requested_mode or 'None'} 允许使用 MP 编排器"

    # === Step 4: 清理 prompt ===
    sanitized_prompt = sanitize_prompt_for_cli_fallback(prompt)

    # === Step 5: 构建用户提示消息（仅在需要时构建）===
    # 统一文案模板：
    # 行1: [图标] 原因
    # 行2: → 实际执行方式 [+ 编排器信息]
    # 行3: → 下一步操作（env/flag/config）
    #
    # 注意：mode_reason 用于内部调试日志，user_message 用于显示给用户
    # 两者信息有重叠但用途不同，不构成"重复提示"
    #
    # 消息级别策略（由 mode_source 决定）：
    # - mode_source="cli": 用户显式请求，使用 "warning" 级别
    # - mode_source="config" 或 None: 配置默认值，使用 "info" 级别避免每次都警告
    # - & 前缀相关消息：用户显式使用 & 前缀表示意图，根据情况决定级别
    user_message = None
    message_level = "info"  # 默认使用 info 级别

    # 场景 1: & 前缀存在但未成功触发，给出解释
    # 使用 prefix_info 中的 ignore_reason 提供更精确的消息
    # 注意：DETECTED_IGNORED_EXPLICIT_MODE 不在此处理，由场景 2 统一处理
    if has_ampersand_prefix and not prefix_routed:
        if prefix_info.status == AmpersandPrefixStatus.DETECTED_IGNORED_DISABLED:
            user_message = (
                "ℹ 检测到 '&' 前缀但 cloud_enabled=False\n"
                f"→ 实际执行: CLI 模式 + {orchestrator} 编排器\n"
                "→ 如需启用 Cloud: 设置 config.yaml 中 cloud_agent.enabled=true"
            )
            message_level = "info"  # 配置问题，信息提示
        elif prefix_info.status == AmpersandPrefixStatus.DETECTED_IGNORED_NO_KEY:
            user_message = (
                "⚠ 检测到 '&' 前缀但未配置 API Key\n"
                f"→ 实际执行: CLI 模式 + {orchestrator} 编排器\n"
                "→ 如需启用 Cloud: export CURSOR_API_KEY=xxx 或 --cloud-api-key"
            )
            message_level = "warning"  # 用户显式使用 & 前缀，应该警告
        elif prefix_info.status == AmpersandPrefixStatus.DETECTED_IGNORED_CLI_MODE:
            user_message = (
                "ℹ 检测到 '&' 前缀但显式指定 --execution-mode cli\n"
                f"→ 实际执行: CLI 模式 + {orchestrator} 编排器（忽略 & 前缀）"
            )
            message_level = "info"  # 用户显式选择 CLI，仅信息提示
        elif prefix_info.status == AmpersandPrefixStatus.DETECTED_IGNORED_AUTO_DETECT_OFF:
            user_message = (
                f"ℹ 检测到 '&' 前缀但 auto_detect_cloud_prefix=False\n→ 实际执行: CLI 模式 + {orchestrator} 编排器"
            )
            message_level = "info"
        # DETECTED_IGNORED_EXPLICIT_MODE 由场景 2 处理（统一显式模式回退消息）

    # 场景 2: 请求 Cloud/Auto 但因缺少 API Key 回退
    # 此场景优先级高于场景 1 的 DETECTED_IGNORED_EXPLICIT_MODE，避免重复提示
    # 消息级别由 mode_source 决定：
    # - mode_source="cli": 用户显式 --execution-mode auto/cloud，使用 warning
    # - mode_source="config" 或 None: 来自配置默认值，使用 info 避免每次都警告
    if mode_lower in ("cloud", "auto") and effective_mode == "cli":
        # 根据 mode_source 决定消息级别
        message_level = "warning" if mode_source == "cli" else "info"

        user_message = (
            f"{'⚠' if message_level == 'warning' else 'ℹ'} 请求 {mode_lower} 模式但未配置 API Key\n"
            f"→ 实际执行: CLI 模式 + {orchestrator} 编排器（强制 basic）\n"
            "→ 如需启用 Cloud: export CURSOR_API_KEY=xxx 或 --cloud-api-key"
        )

    # === DEBUG 日志：记录决策过程 ===
    logger.debug(
        "[build_execution_decision] "
        f"requested_mode={requested_mode}, "
        f"has_ampersand_prefix={has_ampersand_prefix}, "
        f"prefix_routed={prefix_routed}, "
        f"cloud_enabled={cloud_enabled}, "
        f"has_api_key={has_api_key} -> "
        f"effective_mode={effective_mode}, orchestrator={orchestrator}"
    )
    if force_basic_due_to_mode or force_basic_due_to_prefix:
        logger.debug(f"[build_execution_decision] 强制 basic 原因: {orchestrator_reason}")

    return ExecutionDecision(
        effective_mode=effective_mode,
        orchestrator=orchestrator,
        # 新字段（语义明确）
        has_ampersand_prefix=has_ampersand_prefix,
        prefix_routed=prefix_routed,
        # 请求信息
        requested_mode=requested_mode,
        original_prompt=prompt,
        sanitized_prompt=sanitized_prompt,
        # 决策原因
        mode_reason=mode_reason,
        orchestrator_reason=orchestrator_reason,
        # 用户提示
        user_message=user_message,
        message_level=message_level,
        # 配置来源
        mode_source=mode_source,
        # 详细前缀状态
        ampersand_prefix_info=prefix_info,
    )


# ============================================================
# 决策输入构建辅助函数
# ============================================================


@dataclass
class DecisionInputs:
    """构建 ExecutionDecision 所需的输入参数

    封装从 argparse args、原始 prompt 和配置中提取的所有必要参数，
    用于统一 run.py、scripts/run_iterate.py 和 core/config.build_unified_overrides
    三个调用点的决策输入构建逻辑。

    Attributes:
        prompt: 用于决策的 prompt（可能是原始 prompt 或合成的虚拟 prompt）
        requested_mode: 请求的执行模式（经过 resolve_requested_mode_for_decision 计算）
        cloud_enabled: config.yaml 中的 cloud_agent.enabled 设置
        has_api_key: 是否配置了有效的 API Key
        auto_detect_cloud_prefix: 是否启用 & 前缀自动检测
        user_requested_orchestrator: 用户显式指定的编排器类型
        mode_source: 执行模式来源 ("cli"/"config"/None)
        has_ampersand_prefix: 预先检测的 & 前缀状态（语法层面）
        original_prompt: 原始 prompt（未经处理）
    """

    prompt: Optional[str]
    requested_mode: Optional[str]
    cloud_enabled: bool
    has_api_key: bool
    auto_detect_cloud_prefix: bool = True
    user_requested_orchestrator: Optional[str] = None
    mode_source: Optional[str] = None
    has_ampersand_prefix: bool = False
    original_prompt: Optional[str] = None

    def build_decision(self) -> "ExecutionDecision":
        """使用当前输入参数构建 ExecutionDecision

        Returns:
            ExecutionDecision 对象
        """
        return build_execution_decision(
            prompt=self.prompt,
            requested_mode=self.requested_mode,
            cloud_enabled=self.cloud_enabled,
            has_api_key=self.has_api_key,
            auto_detect_cloud_prefix=self.auto_detect_cloud_prefix,
            user_requested_orchestrator=self.user_requested_orchestrator,
            mode_source=self.mode_source,
        )


# 虚拟 prompt 常量，用于 & 前缀检测
# 当 original_prompt 不可用但 has_ampersand_prefix=True 时使用
VIRTUAL_PROMPT_FOR_PREFIX_DETECTION = "& _"


def compute_decision_inputs(
    args: Any,
    original_prompt: Optional[str] = None,
    nl_options: Optional[dict[str, Any]] = None,
    *,
    config: Optional[Any] = None,
) -> DecisionInputs:
    """统一计算 build_execution_decision 所需的输入参数

    从 argparse args、原始 prompt、nl_options 和配置中提取并计算所有必要参数。
    这是 run.py、scripts/run_iterate.py 和 core/config.build_unified_overrides
    共享的决策输入构建逻辑的统一实现。

    设计目标：
    - 消除三个调用点的重复逻辑
    - 减少对虚拟 prompt 的依赖（仅在必要时构造）
    - 提供清晰的参数提取和计算流程

    Args:
        args: argparse.Namespace 命令行参数，需包含：
            - execution_mode: CLI 显式设置的执行模式（可选）
            - orchestrator: 用户显式设置的编排器（可选）
            - no_mp: 是否禁用 MP 编排器（可选）
            - _orchestrator_user_set: 用户是否显式设置了 orchestrator（可选）
        original_prompt: 原始用户 prompt（用于 & 前缀检测）
            - 在 run.py 中为 task 参数
            - 在 scripts/run_iterate.py 中为 args.requirement
        nl_options: 自然语言解析结果字典（可选，run.py 场景）
            - 可包含 "_original_goal", "goal", "has_ampersand_prefix" 等字段
        config: Config 对象（可选，若未提供则调用 get_config()）

    Returns:
        DecisionInputs 对象，包含 build_execution_decision 所需的全部参数

    Examples:
        >>> # run.py 场景
        >>> inputs = compute_decision_inputs(args, original_prompt=task)
        >>> decision = inputs.build_decision()

        >>> # scripts/run_iterate.py 场景
        >>> inputs = compute_decision_inputs(args, original_prompt=args.requirement)
        >>> decision = inputs.build_decision()

        >>> # core/config.build_unified_overrides 场景（execution_decision 缺失重建）
        >>> inputs = compute_decision_inputs(args, nl_options=nl_options)
        >>> decision = inputs.build_decision()
    """
    # 延迟导入避免循环依赖
    from core.config import get_config as _get_config

    nl_options = nl_options or {}

    # === Step 1: 获取配置 ===
    if config is None:
        config = _get_config()
    cloud_enabled = config.cloud_agent.enabled
    config_execution_mode = config.cloud_agent.execution_mode
    config_auto_detect_cloud_prefix = config.cloud_agent.auto_detect_cloud_prefix

    # === Step 2: 获取 API Key ===
    # 延迟导入避免循环依赖
    from cursor.cloud_client import CloudClientFactory

    api_key = CloudClientFactory.resolve_api_key()
    has_api_key = bool(api_key)

    # === Step 3: 确定原始 prompt ===
    # 优先级: original_prompt 参数 > nl_options["_original_goal"] > nl_options["goal"]
    effective_original_prompt = original_prompt
    if effective_original_prompt is None:
        effective_original_prompt = nl_options.get("_original_goal") or nl_options.get("goal")

    # === Step 4: 检测 & 前缀（语法层面）===
    # 优先使用 nl_options 中已有的值，其次从 effective_original_prompt 检测
    has_ampersand_prefix_from_options = nl_options.get("has_ampersand_prefix")
    if has_ampersand_prefix_from_options is not None:
        has_ampersand_prefix = has_ampersand_prefix_from_options
    elif effective_original_prompt:
        has_ampersand_prefix = is_cloud_request(effective_original_prompt)
    else:
        has_ampersand_prefix = False

    # === Step 5: 确定用于决策的 prompt ===
    # 当 original_prompt 不可用但 has_ampersand_prefix=True 时，
    # 构造虚拟 prompt 以便 build_execution_decision 正确检测 & 前缀
    prompt_for_decision = effective_original_prompt
    if prompt_for_decision is None and has_ampersand_prefix:
        prompt_for_decision = VIRTUAL_PROMPT_FOR_PREFIX_DETECTION
        logger.debug(
            "[compute_decision_inputs] 原始 prompt 不可用，使用虚拟 prompt '%s' 以保留 & 前缀检测",
            VIRTUAL_PROMPT_FOR_PREFIX_DETECTION,
        )

    # === Step 6: 确定 CLI 参数 ===
    cli_execution_mode = getattr(args, "execution_mode", None)

    # === Step 6.1: 确定 auto_detect_cloud_prefix ===
    # 优先级: CLI 参数（tri-state: None/True/False）> 配置
    # args.auto_detect_cloud_prefix 可能为 None（未设置）、True 或 False
    cli_auto_detect = getattr(args, "auto_detect_cloud_prefix", None)
    auto_detect_cloud_prefix = cli_auto_detect if cli_auto_detect is not None else config_auto_detect_cloud_prefix

    # === Step 7: 确定 requested_mode（使用统一函数）===
    requested_mode = resolve_requested_mode_for_decision(
        cli_execution_mode=cli_execution_mode,
        has_ampersand_prefix=has_ampersand_prefix,
        config_execution_mode=config_execution_mode,
    )

    # === Step 8: 确定 mode_source ===
    mode_source = resolve_mode_source(
        cli_execution_mode=cli_execution_mode,
        has_ampersand_prefix=has_ampersand_prefix,
        requested_mode_for_decision=requested_mode,
    )

    # === Step 9: 确定 user_requested_orchestrator ===
    user_requested_orchestrator = None
    if getattr(args, "no_mp", False):
        # --no-mp 被设置，强制 orchestrator 为 basic
        user_requested_orchestrator = "basic"
    elif getattr(args, "_orchestrator_user_set", False):
        # 用户显式设置了 --orchestrator 参数
        user_requested_orchestrator = getattr(args, "orchestrator", None)

    # === Step 10: 构建并返回 DecisionInputs ===
    return DecisionInputs(
        prompt=prompt_for_decision,
        requested_mode=requested_mode,
        cloud_enabled=cloud_enabled,
        has_api_key=has_api_key,
        auto_detect_cloud_prefix=auto_detect_cloud_prefix,
        user_requested_orchestrator=user_requested_orchestrator,
        mode_source=mode_source,
        has_ampersand_prefix=has_ampersand_prefix,
        original_prompt=effective_original_prompt,
    )


# ============================================================
# 副作用控制策略（Side Effect Control Policy）
# ============================================================


@dataclass
class SideEffectPolicy:
    """副作用控制策略

    根据 skip_online, dry_run, minimal 计算各类副作用的允许状态。

    ================================================================================
    策略矩阵 (与模块头部文档保持一致)
    ================================================================================

    | 策略    | 网络请求 | 文件写入 | 缓存写入 | Git 操作 | 目录创建 |
    |---------|----------|----------|----------|----------|----------|
    | normal  | ✓        | ✓        | ✓        | ✓        | ✓        |
    | skip-on | ✗        | ✓        | ✗(*)     | ✓        | ✓        |
    | dry-run | ✓(分析)  | ✗        | ✗        | ✗        | ✗        |
    | minimal | ✗        | ✗        | ✗        | ✗        | ✗        |

    (*) skip_online 模式下，disable_cache_write 默认为 True，但可通过参数覆盖

    ================================================================================
    使用场景
    ================================================================================

    1. KnowledgeUpdater 构造时：
       policy = compute_side_effects(skip_online, dry_run, minimal)
       updater = KnowledgeUpdater(
           offline=not policy.allow_network_fetch,
           disable_cache_write=not policy.allow_cache_write,
           dry_run=not policy.allow_file_write,
       )

    2. SelfIterator.run() 流程控制：
       policy = compute_side_effects(skip_online, dry_run, minimal)
       if policy.allow_network_fetch:
           analysis = await changelog_analyzer.analyze()
       if policy.allow_directory_create:
           await knowledge_updater.initialize()

    3. 条件判断简化：
       # 旧代码
       if not skip_online and not dry_run:
           ...
       # 新代码
       if policy.allow_network_fetch:
           ...

    Attributes:
        allow_network_fetch: 是否允许网络请求（changelog/llms.txt/urls 抓取）
        allow_file_write: 是否允许文件写入（知识库文档写入）
        allow_cache_write: 是否允许缓存写入（llms.txt 缓存）
        allow_git_operations: 是否允许 Git 操作（commit/push）
        allow_directory_create: 是否允许目录创建（.cursor/knowledge/）
        skip_online: 原始 skip_online 参数（用于调试/日志）
        dry_run: 原始 dry_run 参数（用于调试/日志）
        minimal: 原始 minimal 参数（用于调试/日志）
    """

    # 网络请求控制
    allow_network_fetch: bool = True  # changelog/llms.txt/urls 抓取

    # 文件写入控制
    allow_file_write: bool = True  # 知识库文档写入
    allow_cache_write: bool = True  # llms.txt 缓存写入

    # Git 操作控制
    allow_git_operations: bool = True  # commit/push

    # 目录创建控制
    allow_directory_create: bool = True  # .cursor/knowledge/ 目录

    # 原始参数（用于调试/日志/传递给下游）
    skip_online: bool = False
    dry_run: bool = False
    minimal: bool = False

    @property
    def is_minimal(self) -> bool:
        """是否为最小副作用模式

        等效于 skip_online + dry_run 的组合效果。
        """
        return self.skip_online and self.dry_run

    @property
    def is_normal(self) -> bool:
        """是否为正常模式（允许所有副作用）"""
        return (
            self.allow_network_fetch
            and self.allow_file_write
            and self.allow_cache_write
            and self.allow_git_operations
            and self.allow_directory_create
        )

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "allow_network_fetch": self.allow_network_fetch,
            "allow_file_write": self.allow_file_write,
            "allow_cache_write": self.allow_cache_write,
            "allow_git_operations": self.allow_git_operations,
            "allow_directory_create": self.allow_directory_create,
            "skip_online": self.skip_online,
            "dry_run": self.dry_run,
            "minimal": self.minimal,
            "is_minimal": self.is_minimal,
            "is_normal": self.is_normal,
        }

    def __repr__(self) -> str:
        if self.is_minimal:
            return "SideEffectPolicy(minimal=True)"
        elif self.is_normal:
            return "SideEffectPolicy(normal)"
        else:
            flags = []
            if self.skip_online:
                flags.append("skip_online")
            if self.dry_run:
                flags.append("dry_run")
            return f"SideEffectPolicy({', '.join(flags) or 'custom'})"


def compute_side_effects(
    skip_online: bool = False,
    dry_run: bool = False,
    minimal: bool = False,
) -> SideEffectPolicy:
    """计算副作用控制策略

    根据输入参数计算各类副作用的允许状态。这是副作用控制的**统一入口**，
    确保 run.py 和 scripts/run_iterate.py 对相同参数组合产生一致的策略。

    ================================================================================
    参数语义
    ================================================================================

    | 参数        | 语义                           | 来源                        |
    |-------------|--------------------------------|-----------------------------|
    | skip_online | 跳过在线检查，使用本地缓存     | --skip-online CLI 参数      |
    | dry_run     | 仅分析不执行，不修改任何文件   | --dry-run CLI 参数          |
    | minimal     | 最小副作用模式（组合参数）     | --minimal CLI 参数          |

    minimal=True 等效于 skip_online=True + dry_run=True，但提供更清晰的语义。

    ================================================================================
    策略计算规则
    ================================================================================

    1. **minimal 模式优先**：
       - 当 minimal=True 时，自动设置 skip_online=True 和 dry_run=True
       - 所有副作用都被禁止

    2. **skip_online 模式**：
       - 禁止网络请求（changelog/llms.txt/urls 抓取）
       - 禁止缓存写入（因为无新数据可写）
       - 允许文件写入、Git 操作、目录创建

    3. **dry_run 模式**：
       - 允许网络请求（用于分析）
       - 禁止文件写入、缓存写入、Git 操作、目录创建

    4. **组合规则**：
       - skip_online + dry_run = minimal（最严格）
       - 各策略字段取交集（更严格者优先）

    Args:
        skip_online: 跳过在线检查，使用本地缓存（默认 False）
        dry_run: 仅分析不执行，不修改任何文件（默认 False）
        minimal: 最小副作用模式，等效于 skip_online + dry_run（默认 False）

    Returns:
        SideEffectPolicy 实例，包含各类副作用的允许状态

    Examples:
        >>> # 正常模式
        >>> policy = compute_side_effects()
        >>> policy.allow_network_fetch
        True
        >>> policy.allow_file_write
        True

        >>> # skip_online 模式
        >>> policy = compute_side_effects(skip_online=True)
        >>> policy.allow_network_fetch
        False
        >>> policy.allow_file_write
        True
        >>> policy.allow_cache_write
        False

        >>> # dry_run 模式
        >>> policy = compute_side_effects(dry_run=True)
        >>> policy.allow_network_fetch
        True
        >>> policy.allow_file_write
        False

        >>> # minimal 模式
        >>> policy = compute_side_effects(minimal=True)
        >>> policy.allow_network_fetch
        False
        >>> policy.allow_file_write
        False
        >>> policy.is_minimal
        True

        >>> # 显式组合
        >>> policy = compute_side_effects(skip_online=True, dry_run=True)
        >>> policy.is_minimal
        True
    """
    # Step 1: minimal 模式强制 skip_online + dry_run
    if minimal:
        skip_online = True
        dry_run = True

    # Step 2: 计算各策略字段

    # 网络请求：skip_online 禁止
    allow_network_fetch = not skip_online

    # 文件写入：dry_run 禁止
    allow_file_write = not dry_run

    # 缓存写入：skip_online 或 dry_run 禁止
    # - skip_online: 无新数据可写
    # - dry_run: 不允许任何写入
    allow_cache_write = not skip_online and not dry_run

    # Git 操作：dry_run 禁止
    allow_git_operations = not dry_run

    # 目录创建：dry_run 禁止（与文件写入一致）
    allow_directory_create = not dry_run

    return SideEffectPolicy(
        allow_network_fetch=allow_network_fetch,
        allow_file_write=allow_file_write,
        allow_cache_write=allow_cache_write,
        allow_git_operations=allow_git_operations,
        allow_directory_create=allow_directory_create,
        skip_online=skip_online,
        dry_run=dry_run,
        minimal=minimal,
    )
