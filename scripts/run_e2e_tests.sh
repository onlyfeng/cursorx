#!/bin/bash
# =============================================================================
# E2E 测试运行脚本
# 支持本地运行和 CI 运行两种模式
# 
# 使用方法:
#   ./scripts/run_e2e_tests.sh              # 默认模式 (本地)
#   ./scripts/run_e2e_tests.sh --ci         # CI 模式
#   ./scripts/run_e2e_tests.sh --local      # 本地模式 (详细输出)
#   ./scripts/run_e2e_tests.sh --coverage   # 生成覆盖率报告
#   ./scripts/run_e2e_tests.sh --all        # 运行所有测试 (包括 slow)
#   ./scripts/run_e2e_tests.sh --core       # 仅运行核心测试集合 (快速验证)
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
MODE="local"
COVERAGE=false
RUN_ALL=false
CORE_ONLY=false
TIMEOUT=300
VERBOSE=true
EXIT_CODE=0

# 核心测试文件列表
CORE_TEST_FILES=(
    "tests/test_run.py"
    "tests/test_self_iterate.py"
    "tests/test_e2e_execution_modes.py"
    "tests/test_orchestrator_mp_commit.py"
)

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ci)
            MODE="ci"
            VERBOSE=false
            shift
            ;;
        --local)
            MODE="local"
            VERBOSE=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --core)
            CORE_ONLY=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            echo "E2E 测试运行脚本"
            echo ""
            echo "使用方法:"
            echo "  $0 [options]"
            echo ""
            echo "选项:"
            echo "  --ci          CI 模式 (简洁输出)"
            echo "  --local       本地模式 (详细输出，默认)"
            echo "  --coverage    生成覆盖率报告"
            echo "  --all         运行所有测试 (包括 slow 标记的测试)"
            echo "  --core        仅运行核心测试集合 (快速验证关键功能)"
            echo "  --timeout N   设置超时时间 (秒，默认 300)"
            echo "  -h, --help    显示帮助信息"
            echo ""
            echo "核心测试集合包含:"
            echo "  - tests/test_run.py (plan/ask/auto/iterate 模式)"
            echo "  - tests/test_self_iterate.py (changelog 解析稳健性与回退)"
            echo "  - tests/test_e2e_execution_modes.py (执行器模式)"
            echo "  - tests/test_orchestrator_mp_commit.py (提交策略与健康检查)"
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            exit 1
            ;;
    esac
done

# 打印函数
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    print_header "检查依赖"
    
    # 检查 pytest
    if ! python -c "import pytest" 2>/dev/null; then
        print_error "pytest 未安装"
        print_info "请运行: pip install pytest"
        exit 1
    fi
    print_success "pytest 已安装"
    
    # 检查 pytest-timeout
    if ! python -c "import pytest_timeout" 2>/dev/null; then
        print_warning "pytest-timeout 未安装，正在安装..."
        pip install pytest-timeout
    fi
    print_success "pytest-timeout 已安装"
    
    # 如果需要覆盖率，检查 pytest-cov
    if [ "$COVERAGE" = true ]; then
        if ! python -c "import pytest_cov" 2>/dev/null; then
            print_warning "pytest-cov 未安装，正在安装..."
            pip install pytest-cov
        fi
        print_success "pytest-cov 已安装"
    fi
}

# 构建 pytest 命令
build_pytest_command() {
    local cmd="pytest"
    
    # 测试目标
    if [ "$CORE_ONLY" = true ]; then
        # 核心测试集合
        cmd="$cmd ${CORE_TEST_FILES[*]}"
    else
        cmd="$cmd tests/"
    fi
    
    # 基本选项
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v --tb=long"
    else
        cmd="$cmd -v --tb=short"
    fi
    
    # 颜色输出
    cmd="$cmd --color=yes"
    
    # 测试标记
    if [ "$CORE_ONLY" = true ]; then
        # 核心测试：跳过 slow 和 e2e 标记的测试以保持快速
        cmd="$cmd -m 'not slow and not e2e'"
    elif [ "$RUN_ALL" = true ]; then
        cmd="$cmd -m 'e2e'"
    else
        cmd="$cmd -m 'e2e and not slow'"
    fi
    
    # 覆盖率
    if [ "$COVERAGE" = true ]; then
        cmd="$cmd --cov=. --cov-report=xml:coverage-e2e.xml --cov-report=html:htmlcov-e2e --cov-report=term-missing"
    fi
    
    echo "$cmd"
}

# 运行 E2E 测试
run_e2e_tests() {
    if [ "$CORE_ONLY" = true ]; then
        print_header "运行核心测试集合"
    else
        print_header "运行 E2E 测试"
    fi
    
    local pytest_cmd=$(build_pytest_command)
    print_info "执行命令: $pytest_cmd"
    echo ""
    
    set +e
    eval "$pytest_cmd"
    EXIT_CODE=$?
    set -e
    
    return $EXIT_CODE
}

# 生成报告
generate_report() {
    print_header "测试报告"
    
    local test_type="E2E 测试"
    if [ "$CORE_ONLY" = true ]; then
        test_type="核心测试集合"
    fi
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_success "所有 ${test_type} 通过!"
    elif [ $EXIT_CODE -eq 5 ]; then
        if [ "$CORE_ONLY" = true ]; then
            print_warning "没有找到匹配的核心测试用例"
            print_info "这可能是因为所有测试都被 'slow' 或 'e2e' 标记"
        else
            print_warning "没有找到 E2E 测试用例"
            print_info "请确保测试函数使用 @pytest.mark.e2e 装饰器标记"
        fi
        EXIT_CODE=0  # 没有测试用例不算失败
    else
        print_error "${test_type} 失败 (退出码: $EXIT_CODE)"
    fi
    
    # 覆盖率报告
    if [ "$COVERAGE" = true ] && [ -f "coverage-e2e.xml" ]; then
        echo ""
        print_info "覆盖率报告已生成:"
        print_info "  - XML: coverage-e2e.xml"
        print_info "  - HTML: htmlcov-e2e/index.html"
    fi
}

# CI 模式输出
ci_output() {
    if [ "$MODE" = "ci" ]; then
        echo ""
        local test_type="E2E 测试"
        if [ "$CORE_ONLY" = true ]; then
            test_type="核心测试集合"
        fi
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "::notice::${test_type} 通过"
        elif [ $EXIT_CODE -eq 5 ]; then
            echo "::warning::没有找到匹配的测试用例"
        else
            echo "::error::${test_type} 失败"
        fi
    fi
}

# 主流程
main() {
    if [ "$CORE_ONLY" = true ]; then
        print_header "核心测试运行器"
    else
        print_header "E2E 测试运行器"
    fi
    print_info "模式: $MODE"
    print_info "覆盖率: $COVERAGE"
    print_info "运行全部: $RUN_ALL"
    print_info "仅核心测试: $CORE_ONLY"
    
    check_dependencies
    run_e2e_tests || true
    generate_report
    ci_output
    
    exit $EXIT_CODE
}

# 运行主流程
main
