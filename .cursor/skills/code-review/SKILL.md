---
name: code-review
description: 自动化代码审查。用于审查代码质量、检测潜在问题和安全漏洞。
---

# 代码审查技能 (Code Review Skill)

对代码变更进行全面审查，识别问题并提供改进建议。

## 使用时机

- 审查 PR 或提交的代码变更
- 评估代码质量
- 检查安全漏洞
- 识别性能问题

## 审查流程

### 1. 获取变更信息
```bash
# 查看未提交的变更
git diff

# 查看特定提交的变更
git show <commit-hash>

# 查看 PR 差异
git diff main..HEAD
```

### 2. 分析变更
- 理解变更的目的和范围
- 检查代码逻辑正确性
- 评估代码质量和可读性

### 3. 检查重点

#### 高严重性问题
- 空值/未定义解引用
- 资源泄漏（未关闭的文件或连接）
- SQL/XSS 注入漏洞
- 并发/竞态条件
- 缺少错误处理

#### 中等严重性问题
- 逻辑错误
- 性能反模式
- 代码重复

#### 低严重性问题
- 代码风格不一致
- 缺少注释

#### 代码集成验证

##### 导入验证
检查新增/修改的模块是否可正常导入：
```bash
# 验证单个模块导入
python -c "import module_name"

# 验证修改的文件可导入
python -c "import sys; sys.path.insert(0, '.'); import path.to.module"

# 批量验证所有变更文件
git diff --name-only --diff-filter=AM -- '*.py' | while read f; do
  module=$(echo "$f" | sed 's/\.py$//' | tr '/' '.')
  python -c "import $module" 2>&1 || echo "导入失败: $f"
done
```

##### 依赖检查
确认新增的 import 是否在 requirements.txt 中声明：
```bash
# 提取变更文件中的外部导入
git diff --name-only -- '*.py' | xargs grep -h "^import\|^from" | \
  grep -v "^\(import\|from\) \." | \
  awk '{print $2}' | cut -d'.' -f1 | sort -u

# 对比 requirements.txt
pip freeze | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]' > /tmp/installed.txt
# 手动检查新导入是否已声明

# 快速检查特定包
grep -q "package_name" requirements.txt && echo "已声明" || echo "未声明"
```

##### 接口一致性
检查修改的类/函数签名是否与所有调用处一致：
```bash
# 查找函数/方法的所有调用处
rg "function_name\(" --type py

# 检查类的所有实例化
rg "ClassName\(" --type py

# 查看函数签名定义
rg "def function_name\(" --type py -A 3

# 对比变更前后的签名
git diff HEAD~1 -- '*.py' | grep -E "^\+.*def |^\-.*def "
```

### 4. 运行验证

在完成代码审查后，执行自动化验证确保代码质量：

#### 预提交检查
```bash
# 执行完整的预提交检查（类型检查、代码风格、安全检查）
python scripts/pre_commit_check.py
```

#### 端到端测试
```bash
# 执行基础端到端测试
python tests/test_e2e_basic.py
```

#### 验证结果处理
- **验证通过**: 在报告中标注 "✓ 验证通过"
- **验证失败**: 在报告中明确标注 "✗ 验证失败"，并列出具体错误信息
- 验证失败时，应在审查报告的问题列表中添加相应的高优先级问题

### 5. 输出报告

提供结构化的审查报告：
- 问题列表（按严重程度排序）
- 改进建议
- 整体评价

## 示例

审查最近的变更：
```
请审查最近的代码变更，关注安全问题和代码质量。
```
