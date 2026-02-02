"""验证模型配置脚本"""

import sys

sys.path.insert(0, ".")

from coordinator.orchestrator import Orchestrator, OrchestratorConfig


def verify_models():
    config = OrchestratorConfig()
    orchestrator = Orchestrator(config)

    results = []

    # 验证配置值
    results.append(("config.planner_model", config.planner_model, "gpt-5.2-high"))
    results.append(("config.worker_model", config.worker_model, "opus-4.5-thinking"))
    results.append(("config.reviewer_model", config.reviewer_model, "gpt-5.2-codex"))

    # 验证 Planner
    planner_config = getattr(orchestrator.planner, "_config", None)
    planner_model = planner_config.cursor_config.model if planner_config else "unknown"
    results.append(("planner.model", planner_model, "gpt-5.2-high"))

    # 验证 Reviewer
    reviewer_config = getattr(orchestrator.reviewer, "_config", None)
    reviewer_model = reviewer_config.cursor_config.model if reviewer_config else "unknown"
    results.append(("reviewer.model", reviewer_model, "gpt-5.2-codex"))

    # 验证 Workers
    for i, worker in enumerate(orchestrator.worker_pool.workers):
        worker_config = getattr(worker, "_config", None)
        worker_model = worker_config.cursor_config.model if worker_config else "unknown"
        results.append((f"worker[{i}].model", worker_model, "opus-4.5-thinking"))

    # 输出结果
    all_passed = True
    for name, actual, expected in results:
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            all_passed = False
        print(f"[{status}] {name}: {actual} (expected: {expected})")

    if all_passed:
        print("\n所有模型配置验证通过!")
        return 0
    else:
        print("\n验证失败!")
        return 1


if __name__ == "__main__":
    sys.exit(verify_models())
