"""验证模型配置脚本"""

import sys

sys.path.insert(0, ".")

from coordinator.orchestrator import Orchestrator, OrchestratorConfig
from core.config import get_config


def verify_models():
    yaml_config = get_config()
    expected_planner = yaml_config.models.planner
    expected_worker = yaml_config.models.worker
    expected_reviewer = yaml_config.models.reviewer

    config = OrchestratorConfig()
    orchestrator = Orchestrator(config)

    results = []

    # 验证解析后的模型配置（优先级: 显式传入 > config.yaml > 代码默认值）
    resolved = getattr(orchestrator, "_resolved_config", {}) or {}
    results.append(("resolved.planner_model", resolved.get("planner_model"), expected_planner))
    results.append(("resolved.worker_model", resolved.get("worker_model"), expected_worker))
    results.append(("resolved.reviewer_model", resolved.get("reviewer_model"), expected_reviewer))

    # 验证 Planner
    planner_config = getattr(orchestrator.planner, "_config", None)
    planner_model = planner_config.cursor_config.model if planner_config else "unknown"
    results.append(("planner.model", planner_model, expected_planner))

    # 验证 Reviewer
    reviewer_config = getattr(orchestrator.reviewer, "_config", None)
    reviewer_model = reviewer_config.cursor_config.model if reviewer_config else "unknown"
    results.append(("reviewer.model", reviewer_model, expected_reviewer))

    # 验证 Workers
    for i, worker in enumerate(orchestrator.worker_pool.workers):
        worker_config = getattr(worker, "_config", None)
        worker_model = worker_config.cursor_config.model if worker_config else "unknown"
        results.append((f"worker[{i}].model", worker_model, expected_worker))

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
