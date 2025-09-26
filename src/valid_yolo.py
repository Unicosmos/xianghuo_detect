import os
import psutil
import threading
import time
import argparse
from datetime import datetime
import yaml
import mlflow
import mlflow.data
import pandas as pd
from pathlib import Path
from ultralytics import YOLO, settings
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# 设置北京时区
BEIJING_TZ = ZoneInfo("Asia/Shanghai")

# from src.utils.monitor import monitor_memory
# from src.utils.logger import setup_logger


load_dotenv()

from logging import getLogger

# 设置日志器
logger = getLogger(__name__)
os.environ["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libtcmalloc.so.4"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLO目标检测验证脚本")

    # 基础配置
    parser.add_argument("--config", type=str, required=True, help="验证配置文件路径")
    parser.add_argument(
        "--model", type=str, default=None, help="模型权重文件路径，覆盖配置文件中的设置"
    )

    # MLflow配置
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://47.96.219.105:50001/",
        help="MLflow服务器地址",
    )
    parser.add_argument("--disable-mlflow", action="store_true", help="禁用MLflow集成")
    parser.add_argument(
        "--mlflow-experiment-description",
        type=str,
        default=None,
        help="MLflow实验描述信息",
    )
    parser.add_argument(
        "--mlflow-task-type",
        type=str,
        default="detect",
        help="YOLO任务类型，如：detect, segment, classify, pose, obb",
    )
    parser.add_argument(
        "--mlflow-tags",
        type=str,
        nargs="*",
        default=None,
        help="MLflow实验标签，格式：key1=value1 key2=value2",
    )

    # 运行时可调整的关键参数
    parser.add_argument(
        "--name", type=str, default=None, help="实验名称，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="批次大小，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None, help="输入图像尺寸，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="验证设备，如：0,1,2,3 或 cpu，覆盖配置文件中的设置",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="数据加载进程数，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--conf", type=float, default=None, help="置信度阈值，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--iou", type=float, default=None, help="IoU阈值，覆盖配置文件中的设置"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=None,
        help="每张图像最大检测数，覆盖配置文件中的设置",
    )
    parser.add_argument("--half", action="store_true", help="使用FP16半精度推理")
    parser.add_argument("--dnn", action="store_true", help="使用OpenCV DNN进行ONNX推理")
    parser.add_argument("--plots", action="store_true", help="保存验证图表")
    parser.add_argument("--save-txt", action="store_true", help="保存结果到*.txt文件")
    parser.add_argument(
        "--save-conf", action="store_true", help="保存置信度到--save-txt标签"
    )
    parser.add_argument("--save-json", action="store_true", help="保存结果到JSON文件")
    parser.add_argument(
        "--split", type=str, default=None, help="数据集分割，如：val, test, train"
    )
    parser.add_argument("--rect", action="store_true", help="矩形验证")
    parser.add_argument("--augment", action="store_true", help="增强验证")
    parser.add_argument("--verbose", action="store_true", help="显示详细验证日志")

    return parser.parse_args()


def setup_mlflow(args):
    """配置MLflow集成"""
    if args.disable_mlflow:
        logger.info("MLflow集成已禁用")
        return None, None

    # 设置实验名称和运行名称
    mlflow_experiment_name = getattr(args, "name", None)
    mlflow_run = f'{mlflow_experiment_name}_validation_{datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")}'

    # 设置MLflow 环境变量
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name
    os.environ["MLFLOW_RUN"] = mlflow_run

    # 启用MLflow集成
    settings.update({"mlflow": True})

    # 设置MLflow实验和标签
    try:
        mlflow.set_tracking_uri(args.mlflow_uri)

        # 准备实验标签
        experiment_tags = {}

        # 添加实验描述
        if args.mlflow_experiment_description:
            experiment_tags["mlflow.note.content"] = args.mlflow_experiment_description

        # 添加任务类型
        experiment_tags["yolo_task_type"] = args.mlflow_task_type

        # 添加自定义标签
        if args.mlflow_tags:
            for tag in args.mlflow_tags:
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    experiment_tags[key.strip()] = value.strip()

        # 设置或获取实验
        experiment = mlflow.set_experiment(
            experiment_name=mlflow_experiment_name,
            tags=experiment_tags if experiment_tags else None,
        )

        logger.info(
            f"MLflow实验已设置: {experiment.name} (ID: {experiment.experiment_id})"
        )
        if experiment_tags:
            logger.info(f"实验标签: {experiment_tags}")

    except Exception as e:
        logger.warning(f"设置MLflow实验信息时出错: {e}")

    logger.info(
        f"MLflow集成已启用 - 实验: {mlflow_experiment_name}, 运行: {mlflow_run}"
    )
    logger.info(f"MLflow服务器: {args.mlflow_uri}")
    logger.info("ultralytics将自动记录验证指标、参数和结果")


def load_and_override_config(config_path, args):
    """加载配置文件并用命令行参数覆盖"""
    # 读取验证配置文件
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"加载配置文件: {config_path}")
    logger.info(f"配置文件内容: {config}")

    # 用命令行参数覆盖配置文件中的设置
    if args.model is not None:
        config["model"] = args.model
        logger.info(f"覆盖model: {args.model}")

    if args.name is not None:
        config["name"] = args.name
        logger.info(f"覆盖name: {args.name}")
    else:
        if "name" in config:
            args.name = config["name"]
        else:
            logger.error(
                "错误: name未设置。请在命令行中使用--name参数或在配置文件中设置name字段。"
            )
            raise ValueError("name参数是必需的，请设置该参数后重新运行。")

    # 设置 name 为实验名称+时间戳, 避免本地命名冲突自动+1
    origin_name = config["name"]
    timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")
    config["name"] = f"{origin_name}_validation_{timestamp}"

    if args.batch is not None:
        config["batch"] = args.batch
        logger.info(f"覆盖batch: {args.batch}")
    elif config.get("batch", -1) == -1:
        # 如果配置文件中batch为-1，设置一个合理的默认值
        config["batch"] = 16
        logger.info(f"设置默认batch: {config['batch']}")

    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
        logger.info(f"覆盖imgsz: {args.imgsz}")

    if args.device is not None:
        # 处理设备参数
        if args.device.lower() == "cpu":
            config["device"] = "cpu"
        else:
            # 将逗号分隔的设备ID转换为列表
            device_list = [int(d.strip()) for d in args.device.split(",")]
            config["device"] = device_list
        logger.info(f"覆盖device: {config['device']}")

    if args.workers is not None:
        config["workers"] = args.workers
        logger.info(f"覆盖workers: {args.workers}")

    if args.conf is not None:
        config["conf"] = args.conf
        logger.info(f"覆盖conf: {args.conf}")

    if args.iou is not None:
        config["iou"] = args.iou
        logger.info(f"覆盖iou: {args.iou}")

    if args.max_det is not None:
        config["max_det"] = args.max_det
        logger.info(f"覆盖max_det: {args.max_det}")

    if args.half:
        config["half"] = True
        logger.info("启用FP16半精度推理")

    if args.dnn:
        config["dnn"] = True
        logger.info("启用OpenCV DNN")

    if args.plots:
        config["plots"] = True
        logger.info("启用验证图表保存")

    if args.save_txt:
        config["save_txt"] = True
        logger.info("启用结果保存到txt文件")

    if args.save_conf:
        config["save_conf"] = True
        logger.info("启用置信度保存")

    if args.save_json:
        config["save_json"] = True
        logger.info("启用结果保存到JSON文件")

    if args.split is not None:
        config["split"] = args.split
        logger.info(f"覆盖split: {args.split}")

    if args.rect:
        config["rect"] = True
        logger.info("启用矩形验证")

    if args.augment:
        config["augment"] = True
        logger.info("启用增强验证")

    if args.verbose:
        config["verbose"] = True
        logger.info("启用详细日志")

    return config


def log_dataset_info(data_yaml_path, mlflow_enabled=True):
    """记录数据集信息到MLflow"""
    if not mlflow_enabled:
        return

    try:
        # 获取数据集目录名作为数据集名称
        data_yaml_path = Path(data_yaml_path)
        dataset_name = data_yaml_path.parent.name
        dataset_path = str(data_yaml_path.parent)

        # 读取数据集配置信息
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            dataset_config = yaml.safe_load(f)

        # 记录数据集相关参数
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("num_classes", dataset_config.get("nc", 0))
        mlflow.log_param("class_names", list(dataset_config.get("names", {}).values()))
        mlflow.log_param("train_path", dataset_config.get("train", ""))
        mlflow.log_param("val_path", dataset_config.get("val", ""))

        # 记录数据集配置文件作为artifact
        mlflow.log_artifact(str(data_yaml_path), "dataset_config")

        # 创建一个简单的数据集描述用于记录
        dataset_info = pd.DataFrame(
            {
                "property": ["dataset_name", "num_classes", "train_path", "val_path"],
                "value": [
                    dataset_name,
                    str(dataset_config.get("nc", 0)),
                    dataset_config.get("train", ""),
                    dataset_config.get("val", ""),
                ],
            }
        )

        # 使用 from_pandas 创建数据集并记录
        dataset = mlflow.data.from_pandas(
            dataset_info, source=f"file://{dataset_path}", name=dataset_name
        )

        # 记录数据集到MLflow，指定为验证上下文
        mlflow.log_input(dataset, context="validation")

        logger.info(f"数据集信息已记录到MLflow: {dataset_name}")

    except Exception as e:
        logger.error(f"记录数据集信息时发生错误: {e}")


def log_model_info(model_path, mlflow_enabled=True):
    """记录模型信息到MLflow"""
    if not mlflow_enabled:
        return

    try:
        model_path = Path(model_path)
        model_name = model_path.stem

        # 记录模型相关参数
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param(
            "model_size_mb", round(model_path.stat().st_size / (1024 * 1024), 2)
        )

        # 如果模型文件存在，记录为artifact
        if model_path.exists():
            mlflow.log_artifact(str(model_path), "model")

        logger.info(f"模型信息已记录到MLflow: {model_name}")

    except Exception as e:
        logger.error(f"记录模型信息时发生错误: {e}")


def log_validation_metrics(metrics, mlflow_enabled=True):
    """记录验证指标到MLflow"""
    if not mlflow_enabled:
        return

    try:
        # 记录主要指标
        if hasattr(metrics, "box"):
            box_metrics = metrics.box

            # 记录mAP指标
            if hasattr(box_metrics, "map"):
                mlflow.log_metric("metrics/mAP50-95", box_metrics.map)
            if hasattr(box_metrics, "map50"):
                mlflow.log_metric("metrics/mAP50", box_metrics.map50)
            if hasattr(box_metrics, "map75"):
                mlflow.log_metric("metrics/mAP75", box_metrics.map75)

            # 记录每个类别的mAP
            if hasattr(box_metrics, "maps") and box_metrics.maps is not None:
                for i, map_val in enumerate(box_metrics.maps):
                    mlflow.log_metric(f"metrics/mAP50-95_class_{i}", map_val)

            # 记录精确度和召回率
            if hasattr(box_metrics, "mp"):
                mlflow.log_metric("metrics/precision", box_metrics.mp)
            if hasattr(box_metrics, "mr"):
                mlflow.log_metric("metrics/recall", box_metrics.mr)

            # 记录F1分数
            if hasattr(box_metrics, "f1"):
                # 安全处理F1分数，可能是numpy数组
                f1_value = box_metrics.f1
                if hasattr(f1_value, "item"):
                    # 如果是numpy数组，取第一个元素
                    f1_value = f1_value.item() if f1_value.size == 1 else f1_value[0]
                elif hasattr(f1_value, "__len__") and len(f1_value) > 0:
                    # 如果是列表或其他序列
                    f1_value = f1_value[0]
                mlflow.log_metric("f1_score", f1_value)

        logger.info("验证指标已记录到MLflow")

    except Exception as e:
        logger.error(f"记录MLflow指标时发生错误: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 启动内存监控线程
    # memory_thread = threading.Thread(target=monitor_memory, daemon=True)
    # memory_thread.start()

    # 加载并覆盖配置
    config = load_and_override_config(args.config, args)

    # 配置MLflow集成
    setup_mlflow(args)

    # MLflow集成状态检查
    mlflow_enabled = not args.disable_mlflow and settings.get("mlflow", False)
    if not mlflow_enabled:
        logger.info("MLflow集成已禁用")

    # 直接使用配置文件中指定的模型路径加载模型
    model = YOLO(config["model"])

    # 从验证参数中排除model、data，因为它们需要单独处理
    val_args = {k: v for k, v in config.items() if k not in ["model", "data"]}

    try:
        logger.info("开始验证...")
        logger.info(f"使用配置文件: {args.config}")
        logger.info(f"模型: {config['model']}")
        logger.info(f"数据集: {config['data']}")

        # 记录数据集信息到MLflow
        log_dataset_info(config["data"], mlflow_enabled)

        # 记录模型信息到MLflow
        log_model_info(config["model"], mlflow_enabled)

        # 开始验证，ultralytics会自动处理MLflow集成
        # 自动记录的内容包括：
        # - 验证参数（置信度阈值、IoU阈值等）
        # - 验证指标（mAP、精确度、召回率等）
        # - 验证结果文件（如果启用保存）
        results = model.val(data=config["data"], **val_args)

        # 额外记录验证指标
        log_validation_metrics(results, mlflow_enabled)

        # 打印主要验证结果
        if hasattr(results, "box"):
            box_metrics = results.box
            logger.info("=== 验证结果 ===")
            if hasattr(box_metrics, "map"):
                logger.info(f"mAP50-95: {box_metrics.map:.4f}")
            if hasattr(box_metrics, "map50"):
                logger.info(f"mAP50: {box_metrics.map50:.4f}")
            if hasattr(box_metrics, "map75"):
                logger.info(f"mAP75: {box_metrics.map75:.4f}")
            if hasattr(box_metrics, "mp"):
                logger.info(f"Precision: {box_metrics.mp:.4f}")
            if hasattr(box_metrics, "mr"):
                logger.info(f"Recall: {box_metrics.mr:.4f}")
            if hasattr(box_metrics, "f1"):
                # 安全处理F1分数，可能是numpy数组
                f1_value = box_metrics.f1
                if hasattr(f1_value, "item"):
                    # 如果是numpy数组，取第一个元素
                    f1_value = f1_value.item() if f1_value.size == 1 else f1_value[0]
                elif hasattr(f1_value, "__len__") and len(f1_value) > 0:
                    # 如果是列表或其他序列
                    f1_value = f1_value[0]
                logger.info(f"F1-Score: {f1_value:.4f}")

        logger.info("验证完成！")

        # 如果启用了保存功能，显示保存路径
        if (
            config.get("save_json", False)
            or config.get("save_txt", False)
            or config.get("plots", False)
        ):
            save_dir = getattr(results, "save_dir", None)
            if save_dir:
                logger.info(f"验证结果已保存到: {save_dir}")

    except Exception as e:
        logger.error(f"验证过程中发生错误: {e}")
        logger.info(f"Memory usage: {psutil.virtual_memory().percent}%")
        raise


if __name__ == "__main__":
    main()

# 使用示例:
# python valid_yolo.py --config configs/yolo/validation/box_detect_validation.yaml --model runs/detect/train/weights/best.pt
# python valid_yolo.py --config configs/yolo/validation/box_detect_validation.yaml --model yolo11n.pt --batch 16 --imgsz 640
# python valid_yolo.py --config configs/yolo/validation/box_detect_validation.yaml --disable-mlflow --save-json --plots
# python valid_yolo.py --config configs/yolo/validation/box_detect_validation.yaml --name box_detect_val --conf 0.25 --iou 0.6
