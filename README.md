# 箱货检测项目 (xianghuo_detect)

基于YOLO的箱货检测系统，用于目标检测、分类和跟踪任务。

## 项目结构

```
xianghuo_detect/
├── src/                  # 核心源代码目录
│   ├── __init__.py       # 包初始化文件
│   ├── train_yolo.py     # YOLO模型训练脚本
│   ├── predict_yolo.py   # 预测脚本
│   ├── valid_yolo.py     # 验证脚本
│   ├── split_dataset.py  # 数据集分割工具
│   └── main.py           # 主入口文件
├── utils/                # 工具函数目录
│   ├── __init__.py       # 包初始化文件
│   └── dataset_toolkit.py  # 数据集处理工具
├── configs/              # 配置文件目录
│   ├── training_config.yaml  # 训练配置
│   └── valid_config.yaml     # 验证配置
├── data/                 # 数据集目录 (已在.gitignore中排除大型数据集)
├── models/               # 模型文件目录 (已在.gitignore中排除)
├── results/              # 训练结果目录
├── logs/                 # 日志文件目录
├── best_production/      # 最佳生产模型目录
├── scripts/              # 辅助脚本目录
├── docs/                 # 文档目录
├── pyproject.toml        # 项目依赖配置
├── uv.lock               # 依赖锁定文件
├── .gitignore            # Git忽略规则
└── README.md             # 项目说明文档
```

## 功能说明

- **目标检测**：使用YOLO模型检测图像中的箱货目标
- **数据集管理**：支持数据集分割、处理和转换
- **模型训练与验证**：完整的训练和评估流程
- **MLflow集成**：支持实验跟踪和模型管理

## 快速开始

### 安装依赖

```bash
# 使用uv安装依赖
uv pip install -e .

# 或使用pip
pip install -r requirements.txt
```

### 训练模型

```bash
python src/train_yolo.py --config configs/training_config.yaml
```

### 验证模型

```bash
python src/valid_yolo.py --config configs/valid_config.yaml
```

### 预测

```bash
python src/predict_yolo.py --model models/best.pt --source path/to/image_or_video
```

## 配置说明

主要配置文件位于`configs/`目录：
- `training_config.yaml`：包含所有训练相关配置
- `valid_config.yaml`：包含所有验证相关配置

## 注意事项

- 大型数据集和模型文件已在`.gitignore`中排除，避免版本控制问题
- 训练结果和日志会分别保存在`results/`和`logs/`目录
- 如需查看详细文档，请参阅`docs/`目录