import os
import psutil
import threading
import time
import argparse
from datetime import datetime
import yaml
import mlflow
from pathlib import Path
from ultralytics import YOLO, settings
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG

# 设置北京时区
BEIJING_TZ = ZoneInfo('Asia/Shanghai')


load_dotenv()

# 设置日志器
logger = getLogger(__name__)

os.environ['LD_PRELOAD'] = '/lib/x86_64-linux-gnu/libtcmalloc.so.4'


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO目标检测训练脚本')
    
    # 基础配置
    parser.add_argument('--config', type=str, 
                       help='训练配置文件路径')
    # MLflow配置
    parser.add_argument('--mlflow-uri', type=str, 
                       default='http://47.96.219.105:50001/',
                       help='MLflow服务器地址')
    parser.add_argument('--disable-mlflow', action='store_true',
                       help='禁用MLflow集成')
    parser.add_argument('--mlflow-experiment-description', type=str, default=None,
                       help='MLflow实验描述信息')
    parser.add_argument('--mlflow-task-type', type=str, default='detect',
                       help='YOLO任务类型，如：detect, segment, classify, pose, obb')
    parser.add_argument('--mlflow-tags', type=str, nargs='*', default=None,
                       help='MLflow实验标签，格式：key1=value1 key2=value2')
    
    # 运行时可调整的关键参数
    parser.add_argument('--name', type=str, default=None,
                       help='实验名称，覆盖配置文件中的设置')
    parser.add_argument('--model', type=str, default=None,
                       help='YOLO基础模型路径，覆盖配置文件中的设置')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数，覆盖配置文件中的设置')
    parser.add_argument('--batch', type=int, default=None,
                       help='批次大小，覆盖配置文件中的设置')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='输入图像尺寸，覆盖配置文件中的设置')
    parser.add_argument('--device', type=str, default=None,
                       help='训练设备，如：0,1,2,3 或 cpu，覆盖配置文件中的设置')
    parser.add_argument('--workers', type=int, default=None,
                       help='数据加载进程数，覆盖配置文件中的设置')
    parser.add_argument('--lr0', type=float, default=None,
                       help='初始学习率，覆盖配置文件中的设置')
    parser.add_argument('--patience', type=int, default=None,
                       help='早停耐心值，覆盖配置文件中的设置')
    parser.add_argument('--save-period', type=int, default=None,
                       help='模型保存周期，覆盖配置文件中的设置')
    parser.add_argument('--cache', action='store_true',
                       help='启用图像缓存到内存')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练的路径')
    parser.add_argument('--fraction', type=float, default=None,
                       help='使用数据集的比例，用于快速测试')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子，确保结果可复现')
    
    # 数据增强参数
    parser.add_argument('--mosaic', type=float, default=None,
                       help='Mosaic增强概率')
    parser.add_argument('--mixup', type=float, default=None,
                       help='MixUp增强概率')
    parser.add_argument('--copy-paste', type=float, default=None,
                       help='Copy-Paste增强概率')
    parser.add_argument('--degrees', type=float, default=None,
                       help='旋转角度范围')
    parser.add_argument('--translate', type=float, default=None,
                       help='平移范围')
    parser.add_argument('--scale', type=float, default=None,
                       help='缩放范围')
    parser.add_argument('--fliplr', type=float, default=None,
                       help='水平翻转概率')
    
    # 其他配置
    parser.add_argument('--profile', action='store_true',
                       help='启用性能分析')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细训练日志')
    
    return parser.parse_args()


def setup_mlflow(args):
    """配置MLflow集成"""
    if args.disable_mlflow:
        logger.info("MLflow集成已禁用")
        return None, None
    
    # 设置实验名称和运行名称
    mlflow_experiment_name = getattr(args, 'name', None)
    mlflow_run = f'{mlflow_experiment_name}_{datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")}'
    
    # 设置MLflow 环境变量
    os.environ['MLFLOW_TRACKING_URI'] = args.mlflow_uri
    os.environ['MLFLOW_EXPERIMENT_NAME'] = mlflow_experiment_name
    os.environ['MLFLOW_RUN'] = mlflow_run
    
    # 启用MLflow集成
    settings.update({"mlflow": True})
    
    # 设置MLflow实验和标签
    try:
        mlflow.set_tracking_uri(args.mlflow_uri)
        
        # 准备实验标签
        experiment_tags = {}
        
        # 添加实验描述
        if args.mlflow_experiment_description:
            experiment_tags['mlflow.note.content'] = args.mlflow_experiment_description
        
        # 添加任务类型
        experiment_tags['yolo_task_type'] = args.mlflow_task_type
        
        # 添加自定义标签
        if args.mlflow_tags:
            for tag in args.mlflow_tags:
                if '=' in tag:
                    key, value = tag.split('=', 1)
                    experiment_tags[key.strip()] = value.strip()
        
        # 设置或获取实验
        experiment = mlflow.set_experiment(
            experiment_name=mlflow_experiment_name,
            tags=experiment_tags if experiment_tags else None
        )
        
        logger.info(f"MLflow实验已设置: {experiment.name} (ID: {experiment.experiment_id})")
        if experiment_tags:
            logger.info(f"实验标签: {experiment_tags}")
            
    except Exception as e:
        logger.warning(f"设置MLflow实验信息时出错: {e}")
    
    logger.info(f"MLflow集成已启用 - 实验: {mlflow_experiment_name}, 运行: {mlflow_run}")
    logger.info(f"MLflow服务器: {args.mlflow_uri}")
    logger.info("ultralytics将自动记录训练指标、参数和模型工件")

def load_and_override_config(config_path, args):
    """加载配置文件并用命令行参数覆盖"""
    # 读取训练配置文件
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"加载配置文件: {config_path}")
    logger.info(f"配置文件内容: {config}")
    
    # 用命令行参数覆盖配置文件中的设置
    if args.model is not None:
        config['model'] = args.model
        logger.info(f"覆盖model: {args.model}")
    
    if args.epochs is not None:
        config['epochs'] = args.epochs
        logger.info(f"覆盖epochs: {args.epochs}")
    
 
    if args.name is not None:
        config['name'] = args.name
        logger.info(f"覆盖name: {args.name}")
    else:
        if "name" in config:
            args.name = config['name']
        else:
            logger.error("错误: name未设置。请在命令行中使用--name参数或在配置文件中设置name字段。")
            raise ValueError("name参数是必需的，请设置该参数后重新运行。")
    # 设置 name 为实验名称+时间戳, 避免本地命名冲突自动+1
    origin_name = config["name"]
    timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")
    config['name'] = f"{origin_name}_{timestamp}"
    
    if args.batch is not None:
        config['batch'] = args.batch
        logger.info(f"覆盖batch: {args.batch}")
    
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
        logger.info(f"覆盖imgsz: {args.imgsz}")
    
    if args.device is not None:
        # 处理设备参数
        if args.device.lower() == 'cpu':
            config['device'] = 'cpu'
        else:
            # 将逗号分隔的设备ID转换为列表
            device_list = [int(d.strip()) for d in args.device.split(',')]
            config['device'] = device_list
        logger.info(f"覆盖device: {config['device']}")
    
    if args.workers is not None:
        config['workers'] = args.workers
        logger.info(f"覆盖workers: {args.workers}")
    
    if args.lr0 is not None:
        config['lr0'] = args.lr0
        logger.info(f"覆盖lr0: {args.lr0}")
    
    if args.patience is not None:
        config['patience'] = args.patience
        logger.info(f"覆盖patience: {args.patience}")
    
    if args.save_period is not None:
        config['save_period'] = args.save_period
        logger.info(f"覆盖save_period: {args.save_period}")
    
    if args.cache:
        config['cache'] = True
        logger.info("启用图像缓存")
    
    if args.resume is not None:
        config['resume'] = args.resume
        logger.info(f"从检查点恢复训练: {args.resume}")
    
    if args.fraction is not None:
        config['fraction'] = args.fraction
        logger.info(f"覆盖fraction: {args.fraction}")
    
    if args.seed is not None:
        config['seed'] = args.seed
        logger.info(f"覆盖seed: {args.seed}")
    
    # 数据增强参数覆盖
    if args.mosaic is not None:
        config['mosaic'] = args.mosaic
        logger.info(f"覆盖mosaic: {args.mosaic}")
    
    if args.mixup is not None:
        config['mixup'] = args.mixup
        logger.info(f"覆盖mixup: {args.mixup}")
    
    if args.copy_paste is not None:
        config['copy_paste'] = args.copy_paste
        logger.info(f"覆盖copy_paste: {args.copy_paste}")
    
    if args.degrees is not None:
        config['degrees'] = args.degrees
        logger.info(f"覆盖degrees: {args.degrees}")
    
    if args.translate is not None:
        config['translate'] = args.translate
        logger.info(f"覆盖translate: {args.translate}")
    
    if args.scale is not None:
        config['scale'] = args.scale
        logger.info(f"覆盖scale: {args.scale}")
    
    if args.fliplr is not None:
        config['fliplr'] = args.fliplr
        logger.info(f"覆盖fliplr: {args.fliplr}")
    
    # 其他参数覆盖
    if args.profile:
        config['profile'] = True
        logger.info("启用性能分析")
    
    if args.verbose:
        config['verbose'] = True
        logger.info("启用详细日志")
    
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载并覆盖配置
    config = load_and_override_config(args.config, args)
    
    # 配置MLflow集成
    setup_mlflow(args)
    
    # MLflow集成状态检查
    if not args.disable_mlflow and not settings.get('mlflow', False):
        logger.info("MLflow集成已禁用")
    

    
    # 获取任务名称，用于组织目录结构
    task = config.get('task', 'default')
    
    # 处理模型路径（支持 OSS）
    logger.info(f"处理模型路径: {config['model']}")

    
    # 处理数据集路径（支持 OSS）
    logger.info(f"处理数据集路径: {config['data']}")
    
    # 使用处理后的本地路径加载模型
    print(config['model'])
    logger.info(config['model'])
    model = YOLO(config['model'])
    
    # 从训练参数中排除model、data，因为它们需要单独处理
    train_args = {k: v for k, v in config.items() if k not in ['model', 'data']}
    
    try:
        logger.info("开始训练...")
        logger.info(f"使用配置文件: {args.config}")
        logger.info(f"模型: {config['model']}")
        logger.info(f"数据集: {config['data']}")
        
        # 开始训练，ultralytics会自动处理MLflow集成
        # 自动记录的内容包括：
        # - 训练参数（学习率、批次大小等）
        # - 每个epoch的指标（loss、mAP等）
        # - 模型工件（权重文件、配置文件等）
        results = model.train(data=config['data'], **train_args)
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        logger.info(f"Memory usage: {psutil.virtual_memory().percent}%")
        raise


if __name__ == '__main__':
    main()

# 使用示例:
# python train_yolo.py --config /path/to/config.yaml --epochs 100 --batch 32
# python train_yolo.py --config /path/to/config.yaml --model yolo11n.pt --epochs 50 --lr0 0.001 --device 0,1
# python train_yolo.py --config /path/to/config.yaml --model yolo11s.pt --disable-mlflow --cache --verbose
# nohup python train_yolo.py --config /path/to/config.yaml --model yolo11m.pt --epochs 200 >out_box.log 2>&1 &
