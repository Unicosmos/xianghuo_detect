#!/bin/bash

# 训练脚本，方便一键运行YOLO训练

# 设置中文字体支持
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh
export LC_ALL=zh_CN.UTF-8

# 检查Python环境
if ! command -v python &> /dev/null
then
    echo "Python 未安装，请先安装Python 3.11或更高版本"
    exit 1
fi

# 运行训练脚本
python src/train_yolo.py --config configs/training_config.yaml "$@"

# 如果有错误，提供帮助信息
if [ $? -ne 0 ];\ then
    echo "\n训练失败！请检查错误信息。"
    echo "常见问题："
    echo "1. 确保所有依赖已安装: pip install -r requirements.txt"
    echo "2. 检查配置文件路径是否正确"
    echo "3. 确认数据和模型文件存在"
fi