---
name: task-planning
description: 任务规划和分解。用于分析复杂需求、分解任务、制定实施计划。
---

# 任务规划技能 (Task Planning Skill)

将复杂需求分解为可执行的具体任务，制定清晰的实施计划。

## 使用时机

- 接收到复杂的开发需求
- 需要分解大型功能
- 制定实施路线图
- 评估任务依赖关系

## 规划流程

### 1. 需求分析
- 理解业务目标
- 明确功能范围
- 识别约束条件

### 2. 代码库分析
```bash
# 查看项目结构
tree -L 2

# 分析依赖
cat package.json
cat requirements.txt

# 理解现有代码
find . -name "*.py" -type f | head -20
```

### 3. 任务分解

将需求分解为：
- **原子任务**: 单一职责，可独立完成
- **优先级**: high/medium/low
- **依赖关系**: 明确前置任务
- **复杂度评估**: simple/medium/complex

### 4. 输出计划

```json
{
  "plan_id": "plan_xxx",
  "summary": "计划概要",
  "tasks": [
    {
      "id": "task_1",
      "title": "任务标题",
      "description": "详细描述",
      "priority": "high",
      "dependencies": [],
      "estimated_complexity": "medium",
      "files_to_modify": ["path/to/file.py"]
    }
  ],
  "risks": ["潜在风险"],
  "notes": "其他说明"
}
```

## 最佳实践

1. **具体可执行**: 每个任务应该是可直接执行的
2. **依赖明确**: 清楚标注任务间的依赖关系
3. **范围合理**: 单个任务不宜过大
4. **风险评估**: 识别并记录潜在风险

## 示例

规划一个新功能：
```
请分析需求并制定实施计划：实现用户登录功能，包括注册、登录、密码重置。
```
