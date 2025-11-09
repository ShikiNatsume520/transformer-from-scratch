#!/bin/bash

# --- 脚本配置 ---
# 设置包含配置文件的目录
CONFIG_DIR="E:\Code\LLM_Base_and_Application\transformer-from-scratch\configs"
# 设置conda 环境
CONDA_ENV_NAME="deep_learning_py38"
LOG_DIR="logs"
# 这会告诉库不要进行任何网络检查
export HF_DATASETS_OFFLINE=1

# --- 日志和重定向设置 ---
# 1. 创建日志目录 (如果不存在)
mkdir -p "$LOG_DIR"

# 1. 解决中文等非ASCII字符乱码问题
export PYTHONIOENCODING=UTF-8
# --------------------------------------------------------

# --- 日志和重定向设置 ---
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/experiment_log_$(date +'%Y-%m-%d_%H-%M-%S').txt"
exec > >(tee -a "$LOG_FILE") 2>&1

# --- 脚本开始 ---
echo "======================================================"
echo "          Starting Automated Experiment Runner        "
echo "======================================================"
echo "Mode: Skipping failed experiments and continuing."
echo "======================================================"


# 激活 Conda 环境 (如果配置了)
if [ -n "$CONDA_ENV_NAME" ]; then
    echo "Activating Conda environment: $CONDA_ENV_NAME"
    source activate $CONDA_ENV_NAME
fi

# 检查 Configs 目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Directory '$CONFIG_DIR' not found!"
    exit 1
fi

# 初始化失败实验的计数器和列表
failed_experiments_count=0
failed_experiments_list=""

# 查找所有 .yaml 文件
find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.yaml" -print0 | while IFS= read -r -d $'\0' config_file; do
    # 提取文件名用于日志记录
    config_name=$(basename "$config_file")

    echo ""
    echo "------------------------------------------------------"
    echo "Running experiment with config: $config_name"
    echo "------------------------------------------------------"

    # 执行训练命令
    python train.py --config "$config_file"

    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "WARNING: Experiment with config '$config_name' failed."
        echo "Skipping to the next experiment."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        # 记录失败的实验
        failed_experiments_count=$((failed_experiments_count + 1))
        failed_experiments_list+=" - $config_name\n"
    else
        echo "Experiment '$config_name' completed successfully."
    fi
done

echo ""
echo "======================================================"
echo "                All experiments attempted!            "
echo "======================================================"

# --- 最终总结 ---
if [ $failed_experiments_count -eq 0 ]; then
    echo "✅ All experiments finished successfully!"
else
    echo "⚠️  Total failed experiments: $failed_experiments_count"
    echo "List of failed experiments:"
    # 使用 printf 来正确处理换行符
    printf "$failed_experiments_list"
fi
echo "======================================================"
echo "Full log saved to: $LOG_FILE"
echo "======================================================"