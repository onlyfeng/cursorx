#!/bin/bash
# check_all.sh - 一键检查脚本
# 执行全面的项目健康检查
#
# 支持选项:
#   --ci          CI 模式（非交互式、无颜色输出）
#   --json        JSON 输出格式（便于 CI 解析）
#   --fail-fast   遇到失败立即退出
#   --full, -f    执行完整检查
#   --help, -h    显示帮助信息
#
# 环境变量:
#   CI=true       自动启用 CI 模式
#   NO_COLOR=1    禁用颜色输出

set -e

# ============================================================
# 参数解析
# ============================================================

CI_MODE=false
JSON_OUTPUT=false
FAIL_FAST=false
FULL_CHECK=false

# 环境变量检测：CI=true 自动启用 CI 模式
if [ "${CI:-}" = "true" ] || [ "${CI:-}" = "1" ] || [ -n "${GITHUB_ACTIONS:-}" ] || [ -n "${GITLAB_CI:-}" ] || [ -n "${JENKINS_URL:-}" ]; then
    CI_MODE=true
fi

# NO_COLOR 环境变量支持
if [ -n "${NO_COLOR:-}" ]; then
    CI_MODE=true
fi

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ci)
            CI_MODE=true
            shift
            ;;
        --json)
            JSON_OUTPUT=true
            CI_MODE=true  # JSON 输出时自动禁用颜色
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --full|-f)
            FULL_CHECK=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --ci          CI 模式（非交互式、无颜色输出）"
            echo "  --json        JSON 输出格式（便于 CI 解析）"
            echo "  --fail-fast   遇到失败立即退出"
            echo "  -f, --full    执行完整检查（包括类型、风格和测试）"
            echo "  -h, --help    显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  CI=true       自动启用 CI 模式"
            echo "  NO_COLOR=1    禁用颜色输出"
            echo ""
            echo "示例:"
            echo "  $0                 快速检查"
            echo "  $0 --full          完整检查"
            echo "  $0 --ci --json     CI 环境 JSON 输出"
            echo "  $0 --fail-fast     遇到失败立即退出"
            echo "  CI=true $0         自动 CI 模式"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================
# 颜色定义 (CI 模式下禁用)
# ============================================================

if [ "$CI_MODE" = true ]; then
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
    BOLD=''
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
    BOLD='\033[1m'
fi

# ============================================================
# 计数器和结果收集
# ============================================================

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
SKIP_COUNT=0

# JSON 结果收集
declare -a JSON_CHECKS=()
CURRENT_SECTION=""

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============================================================
# 辅助函数
# ============================================================

# 添加 JSON 检查结果
add_json_check() {
    local status="$1"
    local name="$2"
    local message="$3"
    local section="${CURRENT_SECTION:-unknown}"

    # 转义 JSON 特殊字符
    message=$(echo "$message" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g')
    name=$(echo "$name" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g')

    JSON_CHECKS+=("{\"section\":\"$section\",\"status\":\"$status\",\"name\":\"$name\",\"message\":\"$message\"}")
}

print_header() {
    if [ "$JSON_OUTPUT" = true ]; then
        return
    fi
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
}

print_section() {
    CURRENT_SECTION="$1"
    if [ "$JSON_OUTPUT" = true ]; then
        return
    fi
    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
}

check_pass() {
    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "pass" "$1" ""
    else
        echo -e "  ${GREEN}✓${NC} $1"
    fi
    ((++PASS_COUNT)) || true
}

check_fail() {
    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "fail" "$1" ""
    else
        echo -e "  ${RED}✗${NC} $1"
    fi
    ((++FAIL_COUNT)) || true

    # --fail-fast 模式下立即退出
    if [ "$FAIL_FAST" = true ]; then
        if [ "$JSON_OUTPUT" = true ]; then
            output_json_result 1
        else
            echo ""
            echo -e "${RED}${BOLD}[fail-fast] 检查失败，立即退出${NC}"
        fi
        exit 1
    fi
}

check_warn() {
    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "warn" "$1" ""
    else
        echo -e "  ${YELLOW}⚠${NC} $1"
    fi
    ((++WARN_COUNT)) || true
}

check_skip() {
    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "skip" "$1" ""
    else
        echo -e "  ${BLUE}○${NC} $1 ${YELLOW}(跳过)${NC}"
    fi
    ((++SKIP_COUNT)) || true
}

check_info() {
    if [ "$JSON_OUTPUT" = true ]; then
        add_json_check "info" "$1" ""
    else
        echo -e "  ${BLUE}ℹ${NC} $1"
    fi
}

# 输出 JSON 结果
output_json_result() {
    local exit_code="${1:-0}"
    local success="true"
    [ "$FAIL_COUNT" -gt 0 ] && success="false"

    # 构建检查数组
    local checks_json=""
    for check in "${JSON_CHECKS[@]}"; do
        if [ -n "$checks_json" ]; then
            checks_json="$checks_json,$check"
        else
            checks_json="$check"
        fi
    done

    cat << EOF
{
  "success": $success,
  "exit_code": $exit_code,
  "summary": {
    "passed": $PASS_COUNT,
    "failed": $FAIL_COUNT,
    "warnings": $WARN_COUNT,
    "skipped": $SKIP_COUNT,
    "total": $((PASS_COUNT + FAIL_COUNT + WARN_COUNT + SKIP_COUNT))
  },
  "ci_mode": $CI_MODE,
  "fail_fast": $FAIL_FAST,
  "full_check": $FULL_CHECK,
  "project_root": "$PROJECT_ROOT",
  "timestamp": "$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')",
  "checks": [$checks_json]
}
EOF
}

# ============================================================
# 检查函数
# ============================================================

check_python_version() {
    print_section "Python 环境检查"

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
        PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            check_pass "Python 版本: $PYTHON_VERSION (>= 3.9)"
        else
            check_warn "Python 版本: $PYTHON_VERSION (推荐 >= 3.9)"
        fi
    else
        check_fail "Python3 未安装"
        return 1
    fi

    # 检查 pip
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version 2>/dev/null | cut -d' ' -f2)
        check_pass "pip 版本: $PIP_VERSION"
    else
        check_warn "pip3 命令不可用"
    fi
}

check_dependencies() {
    print_section "依赖检查"

    if [ ! -f "requirements.txt" ]; then
        check_fail "requirements.txt 不存在"
        return 1
    fi

    check_pass "requirements.txt 存在"

    # 使用 Python 直接导入检查（比 pip list 更快）
    DEPS_MISSING=0

    # 核心依赖 (包名:导入名)
    declare -A CORE_DEPS=(
        ["pydantic"]="pydantic"
        ["loguru"]="loguru"
        ["pyyaml"]="yaml"
        ["aiofiles"]="aiofiles"
        ["beautifulsoup4"]="bs4"
    )

    for pkg in "${!CORE_DEPS[@]}"; do
        import_name="${CORE_DEPS[$pkg]}"
        if python3 -c "import $import_name" 2>/dev/null; then
            check_pass "已安装: $pkg"
        else
            check_fail "未安装: $pkg"
            ((DEPS_MISSING++))
        fi
    done

    # 检查可选依赖
    declare -A OPTIONAL_DEPS=(
        ["sentence-transformers"]="sentence_transformers"
        ["chromadb"]="chromadb"
    )

    for pkg in "${!OPTIONAL_DEPS[@]}"; do
        import_name="${OPTIONAL_DEPS[$pkg]}"
        if python3 -c "import $import_name" 2>/dev/null; then
            check_pass "已安装 (可选): $pkg"
        else
            check_warn "未安装 (可选): $pkg"
        fi
    done

    if [ $DEPS_MISSING -gt 0 ]; then
        check_info "运行 'pip install -r requirements.txt' 安装依赖"
    fi
}

check_syntax() {
    print_section "Python 语法检查"

    SYNTAX_ERRORS=0
    PY_FILES=$(find . -name "*.py" -not -path "./.git/*" -not -path "./venv/*" -not -path "./__pycache__/*" 2>/dev/null)

    for file in $PY_FILES; do
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            check_fail "语法错误: $file"
            ((SYNTAX_ERRORS++))
        fi
    done

    if [ $SYNTAX_ERRORS -eq 0 ]; then
        FILE_COUNT=$(echo "$PY_FILES" | wc -w)
        check_pass "语法检查通过 ($FILE_COUNT 个文件)"
    else
        check_fail "发现 $SYNTAX_ERRORS 个语法错误"
    fi
}

check_imports() {
    print_section "模块结构检查"

    MODULES=("core" "agents" "coordinator" "tasks" "cursor" "indexing" "knowledge" "process")
    IMPORT_ERRORS=0

    for module in "${MODULES[@]}"; do
        if [ -d "$module" ]; then
            # 检查是否有 __init__.py
            if [ -f "$module/__init__.py" ]; then
                check_pass "模块结构正确: $module/"
            else
                check_warn "模块缺少 __init__.py: $module/"
                ((IMPORT_ERRORS++))
            fi
        else
            check_skip "模块不存在: $module"
        fi
    done

    return $IMPORT_ERRORS
}

check_type_hints() {
    print_section "类型检查 (mypy)"

    if command -v mypy &> /dev/null; then
        # 只检查核心模块
        if mypy core/ agents/ --ignore-missing-imports --no-error-summary 2>/dev/null; then
            check_pass "类型检查通过"
        else
            check_warn "类型检查有警告 (非致命)"
        fi
    else
        check_skip "mypy 未安装 (pip install mypy)"
    fi
}

check_code_style() {
    print_section "代码风格检查"

    # flake8 检查
    if command -v flake8 &> /dev/null; then
        FLAKE8_ERRORS=$(flake8 --count --select=E9,F63,F7,F82 --show-source --statistics . 2>&1 | tail -1)
        if [ "$FLAKE8_ERRORS" = "0" ] || [ -z "$FLAKE8_ERRORS" ]; then
            check_pass "flake8 严重错误检查通过"
        else
            check_fail "flake8 发现 $FLAKE8_ERRORS 个严重错误"
        fi

        # 代码风格警告（不计入失败）
        STYLE_WARNS=$(flake8 --count --exit-zero --max-complexity=10 --max-line-length=120 . 2>&1 | tail -1)
        if [ "$STYLE_WARNS" = "0" ] || [ -z "$STYLE_WARNS" ]; then
            check_pass "flake8 代码风格检查通过"
        else
            check_warn "flake8 代码风格警告: $STYLE_WARNS 个"
        fi
    else
        check_skip "flake8 未安装 (pip install flake8)"
    fi

    # ruff 检查（更快的替代方案）
    if command -v ruff &> /dev/null; then
        RUFF_ERRORS=$(ruff check . --quiet 2>&1 | wc -l)
        if [ "$RUFF_ERRORS" -eq 0 ]; then
            check_pass "ruff 检查通过"
        else
            check_warn "ruff 发现 $RUFF_ERRORS 个问题"
        fi
    else
        check_info "ruff 未安装 (pip install ruff) - 推荐使用"
    fi
}

check_tests() {
    print_section "测试检查"

    if [ -d "tests" ]; then
        TEST_COUNT=$(find tests -name "test_*.py" | wc -l)
        check_pass "测试目录存在 ($TEST_COUNT 个测试文件)"

        if command -v pytest &> /dev/null; then
            if [ "$JSON_OUTPUT" != true ]; then
                echo -e "  ${BLUE}ℹ${NC} 运行测试..."
            fi

            # 运行测试并捕获结果
            if pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_output.txt; then
                PASSED=$(grep -oP '\d+(?= passed)' /tmp/pytest_output.txt | tail -1)
                check_pass "测试通过: ${PASSED:-0} 个"
            else
                FAILED=$(grep -oP '\d+(?= failed)' /tmp/pytest_output.txt | tail -1)
                check_fail "测试失败: ${FAILED:-?} 个"
            fi
            rm -f /tmp/pytest_output.txt
        else
            check_skip "pytest 未安装 (pip install pytest)"
        fi
    else
        check_warn "tests 目录不存在"
    fi
}

check_coverage() {
    print_section "代码覆盖率检查"

    if ! command -v pytest &> /dev/null; then
        check_skip "pytest 未安装 (pip install pytest)"
        return 0
    fi

    # 检查 pytest-cov 是否安装
    if ! python3 -c "import pytest_cov" 2>/dev/null; then
        check_skip "pytest-cov 未安装 (pip install pytest-cov)"
        return 0
    fi

    if [ "$JSON_OUTPUT" != true ]; then
        echo -e "  ${BLUE}ℹ${NC} 运行覆盖率检查..."
    fi

    # 运行带覆盖率的测试
    # 目标: run.py 及核心模块
    COV_OUTPUT=$(pytest tests/ --cov=run --cov=core --cov=agents --cov=coordinator --cov=cursor --cov=tasks --cov=knowledge --cov=indexing --cov=process --cov-report=term-missing --cov-fail-under=80 2>&1) || true
    COV_EXIT=$?

    # 提取覆盖率百分比
    TOTAL_COV=$(echo "$COV_OUTPUT" | grep -E "^TOTAL" | awk '{print $NF}' | tr -d '%')

    if [ -n "$TOTAL_COV" ]; then
        if [ "$COV_EXIT" -eq 0 ]; then
            check_pass "代码覆盖率: ${TOTAL_COV}% (阈值: 80%)"
        else
            # 检查是否是覆盖率不足导致的失败
            if echo "$COV_OUTPUT" | grep -q "FAIL Required test coverage"; then
                check_fail "代码覆盖率不足: ${TOTAL_COV}% (阈值: 80%)"
            else
                # 测试失败但覆盖率可能已计算
                check_warn "测试有失败，覆盖率: ${TOTAL_COV}%"
            fi
        fi

        # 显示详细信息 (非 JSON 模式)
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}ℹ${NC} 覆盖率详情:"
            echo "$COV_OUTPUT" | grep -E "^(Name|TOTAL|run|core|agents|coordinator|cursor|tasks|knowledge|indexing|process)" | head -20 | while read line; do
                echo "      $line"
            done
        fi
    else
        check_warn "无法获取覆盖率数据"
        if [ "$JSON_OUTPUT" != true ]; then
            echo "$COV_OUTPUT" | tail -10
        fi
    fi

    # 生成 HTML 报告 (如果需要)
    if [ "$FULL_CHECK" = true ]; then
        pytest tests/ --cov=run --cov=core --cov=agents --cov=coordinator --cov=cursor --cov=tasks --cov=knowledge --cov=indexing --cov=process --cov-report=html --cov-report=xml -q 2>/dev/null || true
        if [ -d "htmlcov" ]; then
            check_info "HTML 覆盖率报告: htmlcov/index.html"
        fi
        if [ -f "coverage.xml" ]; then
            check_info "XML 覆盖率报告: coverage.xml"
        fi
    fi
}

check_config() {
    print_section "配置文件检查"

    # config.yaml
    if [ -f "config.yaml" ]; then
        if python3 -c "import yaml; yaml.safe_load(open('config.yaml'))" 2>/dev/null; then
            check_pass "config.yaml 格式正确"
        else
            check_fail "config.yaml 格式错误"
        fi
    else
        check_fail "config.yaml 不存在"
    fi

    # mcp.json
    if [ -f "mcp.json" ]; then
        if python3 -c "import json; json.load(open('mcp.json'))" 2>/dev/null; then
            check_pass "mcp.json 格式正确"
        else
            check_fail "mcp.json 格式错误"
        fi
    else
        check_warn "mcp.json 不存在"
    fi

    # .cursor 目录
    if [ -d ".cursor" ]; then
        check_pass ".cursor 目录存在"

        # 检查子配置
        [ -f ".cursor/cli.json" ] && check_pass ".cursor/cli.json 存在" || check_warn ".cursor/cli.json 不存在"
        [ -f ".cursor/hooks.json" ] && check_pass ".cursor/hooks.json 存在" || check_info ".cursor/hooks.json 不存在"
        [ -d ".cursor/agents" ] && check_pass ".cursor/agents/ 目录存在" || check_info ".cursor/agents/ 不存在"
        [ -d ".cursor/rules" ] && check_pass ".cursor/rules/ 目录存在" || check_info ".cursor/rules/ 不存在"
    else
        check_warn ".cursor 目录不存在"
    fi
}

check_git() {
    print_section "Git 状态检查"

    if [ -d ".git" ]; then
        check_pass "Git 仓库已初始化"

        # 当前分支
        BRANCH=$(git branch --show-current 2>/dev/null)
        check_info "当前分支: $BRANCH"

        # 未提交更改
        CHANGES=$(git status --porcelain 2>/dev/null | wc -l)
        if [ "$CHANGES" -eq 0 ]; then
            check_pass "工作区干净"
        else
            check_warn "有 $CHANGES 个未提交的更改"
        fi

        # 未跟踪文件
        UNTRACKED=$(git status --porcelain 2>/dev/null | grep "^??" | wc -l)
        if [ "$UNTRACKED" -gt 0 ]; then
            check_info "有 $UNTRACKED 个未跟踪的文件"
        fi

        # 检查与远程的差异（不做 fetch，使用缓存信息）
        BEHIND=$(git rev-list --count HEAD..@{upstream} 2>/dev/null || echo "")
        AHEAD=$(git rev-list --count @{upstream}..HEAD 2>/dev/null || echo "")

        if [ -n "$BEHIND" ] && [ "$BEHIND" -gt 0 ]; then
            check_info "落后远程 $BEHIND 个提交 (使用缓存)"
        fi
        if [ -n "$AHEAD" ] && [ "$AHEAD" -gt 0 ]; then
            check_info "领先远程 $AHEAD 个提交"
        fi
    else
        check_warn "非 Git 仓库"
    fi
}

check_agent_cli() {
    print_section "Agent CLI 检查"

    if command -v agent &> /dev/null; then
        # 只获取版本，不做网络调用
        AGENT_VERSION=$(timeout 5 agent --version 2>&1 || echo "未知")
        check_pass "agent CLI 已安装: $AGENT_VERSION"
        check_info "运行 'agent status' 检查认证状态"
    else
        check_warn "agent CLI 未安装"
        check_info "安装命令: curl https://cursor.com/install -fsS | bash"
    fi

    # 检查 API 密钥
    if [ -n "$CURSOR_API_KEY" ]; then
        check_pass "CURSOR_API_KEY 环境变量已设置"
    else
        check_info "CURSOR_API_KEY 未设置 (可选)"
    fi
}

check_knowledge_base() {
    print_section "知识库验证"

    # 检查知识库模块是否可导入
    if PYTHONPATH=. python3 -c "from knowledge import KnowledgeManager" 2>/dev/null; then
        check_pass "knowledge 模块可导入"
    else
        check_fail "knowledge 模块导入失败"
        return 1
    fi

    # 检查验证测试文件
    if [ -f "tests/test_knowledge_validation.py" ]; then
        check_pass "知识库验证测试文件存在"

        # 运行快速验证
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}ℹ${NC} 运行知识库快速验证..."
        fi
        if PYTHONPATH=. python3 tests/test_knowledge_validation.py 2>/dev/null | grep -q "所有验证通过"; then
            check_pass "知识库快速验证通过"
        else
            check_warn "知识库快速验证有问题"
        fi
    else
        check_warn "知识库验证测试文件不存在"
        check_info "运行 'bash scripts/validate_knowledge.sh' 进行验证"
    fi

    # 检查验证脚本
    if [ -f "scripts/validate_knowledge.sh" ]; then
        check_pass "知识库验证脚本存在"
    else
        check_info "知识库验证脚本不存在"
    fi
}

check_pre_commit() {
    print_section "预提交检查 (Python)"

    # 检查预提交检查脚本是否存在
    if [ -f "scripts/pre_commit_check.py" ]; then
        check_pass "预提交检查脚本存在"

        # 运行预提交检查
        if [ "$JSON_OUTPUT" != true ]; then
            echo -e "  ${BLUE}ℹ${NC} 运行预提交检查..."
        fi

        # 捕获输出并检查结果
        PRE_COMMIT_OUTPUT=$(PYTHONPATH=. python3 scripts/pre_commit_check.py --json 2>&1)
        PRE_COMMIT_EXIT=$?

        if [ $PRE_COMMIT_EXIT -eq 0 ]; then
            # 解析 JSON 输出获取通过数量
            PASSED_COUNT=$(echo "$PRE_COMMIT_OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('passed_count', 0))" 2>/dev/null || echo "?")
            check_pass "预提交检查通过 ($PASSED_COUNT 项)"
        else
            # 解析失败数量
            FAILED_COUNT=$(echo "$PRE_COMMIT_OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('failed_count', 0))" 2>/dev/null || echo "?")
            check_fail "预提交检查失败 ($FAILED_COUNT 项)"

            # 在非 JSON 模式下显示失败详情
            if [ "$JSON_OUTPUT" != true ]; then
                echo "$PRE_COMMIT_OUTPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for c in d.get('checks', []):
        if not c.get('passed'):
            print(f\"    - {c.get('name')}: {c.get('message')}\")
            if c.get('fix_suggestion'):
                print(f\"      修复: {c.get('fix_suggestion')}\")
except: pass
" 2>/dev/null
            fi
        fi
    else
        check_warn "预提交检查脚本不存在"
        check_info "运行 'python scripts/pre_commit_check.py' 进行检查"
    fi
}

check_directories() {
    print_section "目录结构检查"

    REQUIRED_DIRS=("core" "agents" "coordinator" "tasks" "cursor" "scripts" "logs")

    for dir in "${REQUIRED_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            FILE_COUNT=$(find "$dir" -name "*.py" 2>/dev/null | wc -l)
            check_pass "目录存在: $dir/ ($FILE_COUNT 个 .py 文件)"
        else
            check_fail "目录缺失: $dir/"
        fi
    done
}

check_main_entries() {
    print_section "入口文件检查"

    MAIN_FILES=("main.py" "main_mp.py" "main_with_knowledge.py")

    for file in "${MAIN_FILES[@]}"; do
        if [ -f "$file" ]; then
            if python3 -m py_compile "$file" 2>/dev/null; then
                check_pass "入口文件可用: $file"
            else
                check_fail "入口文件语法错误: $file"
            fi
        else
            check_warn "入口文件不存在: $file"
        fi
    done
}

# ============================================================
# 主程序
# ============================================================

main() {
    # JSON 模式不输出头部
    if [ "$JSON_OUTPUT" != true ]; then
        print_header "项目健康检查 - $(basename "$PROJECT_ROOT")"
        echo -e "  ${BLUE}时间:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
        echo -e "  ${BLUE}路径:${NC} $PROJECT_ROOT"
        if [ "$CI_MODE" = true ]; then
            echo -e "  ${BLUE}模式:${NC} CI"
        fi
        if [ "$FAIL_FAST" = true ]; then
            echo -e "  ${BLUE}选项:${NC} fail-fast"
        fi
    fi

    # 执行所有检查
    check_python_version
    check_dependencies
    check_syntax
    check_imports
    check_directories
    check_main_entries
    check_config
    check_git
    check_agent_cli
    check_knowledge_base
    check_pre_commit

    # 可选的深度检查
    if [ "$FULL_CHECK" = true ]; then
        check_type_hints
        check_code_style
        check_tests
        check_coverage
    else
        if [ "$JSON_OUTPUT" != true ]; then
            print_section "跳过的检查 (使用 --full 启用)"
            check_info "类型检查 (mypy)"
            check_info "代码风格检查 (flake8/ruff)"
            check_info "测试运行 (pytest)"
            check_info "代码覆盖率检查 (pytest-cov, 阈值 80%)"
        fi
    fi

    # 计算退出码
    local EXIT_CODE=0
    if [ $FAIL_COUNT -gt 0 ]; then
        EXIT_CODE=1
    fi

    # JSON 输出
    if [ "$JSON_OUTPUT" = true ]; then
        output_json_result $EXIT_CODE
        exit $EXIT_CODE
    fi

    # 汇总报告
    print_header "检查结果汇总"
    echo ""
    echo -e "  ${GREEN}✓ 通过:${NC} $PASS_COUNT"
    echo -e "  ${RED}✗ 失败:${NC} $FAIL_COUNT"
    echo -e "  ${YELLOW}⚠ 警告:${NC} $WARN_COUNT"
    echo -e "  ${BLUE}○ 跳过:${NC} $SKIP_COUNT"
    echo ""

    TOTAL=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT + SKIP_COUNT))

    if [ $FAIL_COUNT -eq 0 ]; then
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}${BOLD}  所有检查通过! 项目状态良好${NC}"
        echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        exit 0
    else
        echo -e "${RED}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}${BOLD}  发现 $FAIL_COUNT 个问题需要修复${NC}"
        echo -e "${RED}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        exit 1
    fi
}

main
