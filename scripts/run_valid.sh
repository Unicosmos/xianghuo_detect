#!/bin/bash

# 设置中文字体支持
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到Python3环境。请先安装Python3。"
    exit 1
fi

# 检查配置文件是否存在
CONFIG_FILE="configs/valid_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误：配置文件 $CONFIG_FILE 不存在。请确保已正确配置验证环境。"
    exit 1
fi

# 执行验证命令
echo "开始模型验证..."
python3 src/valid_yolo.py --config "$CONFIG_FILE" "$@"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "验证完成！结果已保存至配置文件指定的目录。"
else
    echo "验证失败！请检查错误信息并修复问题。"
    exit 1
fi

# 常见问题提示
echo "\n提示："
echo "1. 如果验证结果不理想，可以尝试调整配置文件中的参数"
echo "2. 验证结果包括精度、召回率、mAP等指标"
echo "3. 详细日志可以在logs目录下查看"