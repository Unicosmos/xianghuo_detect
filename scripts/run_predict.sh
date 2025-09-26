#!/bin/bash

# 预测脚本，方便一键运行YOLO预测

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

# 默认参数
MODEL="../models/best.pt"
SOURCE="."

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--model 模型路径] [--source 图像/视频路径]"
            exit 1
            ;;
    esac
done

# 运行预测脚本
python src/predict_yolo.py --model "$MODEL" --source "$SOURCE"

# 如果有错误，提供帮助信息
if [ $? -ne 0 ];\ then
    echo "\n预测失败！请检查错误信息。"
    echo "常见问题："
    echo "1. 确保所有依赖已安装: pip install -r requirements.txt"
    echo "2. 检查模型文件路径是否正确"
    echo "3. 确认图像/视频文件存在"
fi