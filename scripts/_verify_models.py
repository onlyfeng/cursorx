"""验证模型配置脚本"""
import sys
sys.path.insert(0, '.')

from coordinator.orchestrator import Orchestrator, OrchestratorConfig

def verify_models():
    config = OrchestratorConfig()
    orchestrator = Orchestrator(config)
    
    results = []
    
    # 验证配置值
    results.append(("config.planner_model", config.planner_model, "gpt-5.2-high"))
    results.append(("config.worker_model", config.worker_model, "opus-4.5-thinking"))
    results.append(("config.reviewer_model", config.reviewer_model, "opus-4.5-thinking"))
    
    # 验证 Planner
    planner_model = orchestrator.planner._config.cursor_config.model
    results.append(("planner.model", planner_model, "gpt-5.2-high"))
    
    # 验证 Reviewer
    reviewer_model = orchestrator.reviewer._config.cursor_config.model
    results.append(("reviewer.model", reviewer_model, "opus-4.5-thinking"))
    
    # 验证 Workers
    for i, worker in enumerate(orchestrator.worker_pool.workers):
        worker_model = worker._config.cursor_config.model
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
