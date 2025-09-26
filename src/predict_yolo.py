import os
import psutil
import threading
import time
import argparse
from datetime import datetime
import yaml
import pandas as pd
from pathlib import Path
from ultralytics import YOLO, settings
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import json
import cv2
import numpy as np
from PIL import Image
import gc
import torch

# 设置北京时区
BEIJING_TZ = ZoneInfo("Asia/Shanghai")

# from src.utils.monitor import monitor_memory
# from src.utils.logger import setup_logger
# from src.downloads.model_manager import get_model_manager

load_dotenv()

from logging import getLogger, StreamHandler, Formatter, FileHandler

# 设置日志器
logger = getLogger(__name__)

os.environ["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libtcmalloc.so.4"


def cleanup_memory():
    """清理内存的辅助函数"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_single_result(result, config, task_type, save_dir, result_index):
    """处理单个预测结果并保存JSON文件

    Args:
        result: YOLO预测结果对象
        config: 配置字典
        task_type: 任务类型
        save_dir: 保存目录路径
        result_index: 结果索引

    Returns:
        bool: 是否成功保存
    """
    try:
        # 获取原始JSON格式的结果
        json_str = result.to_json()
        yolo_data = json.loads(json_str)

        # 确定图片路径和文件名
        if hasattr(result, "path") and result.path:
            source_name = Path(result.path).stem
            file_name = Path(result.path).name
            json_file = save_dir / f"{source_name}.json"

            # 根据是否有data_prefix参数来设置image路径
            if config.get("data_prefix"):
                # 移除data_prefix末尾的斜杠，避免双斜杠问题
                prefix = config["data_prefix"].rstrip("/")
                image_path = f"{prefix}/{file_name}"
            else:
                image_path = str(result.path)
        else:
            source_name = f"result_{result_index}"
            file_name = f"result_{result_index}.jpg"
            json_file = save_dir / f"result_{result_index}.json"

            # 根据是否有data_prefix参数来设置image路径
            if config.get("data_prefix"):
                # 移除data_prefix末尾的斜杠，避免双斜杠问题
                prefix = config["data_prefix"].rstrip("/")
                image_path = f"{prefix}/{file_name}"
            else:
                image_path = file_name

        # 获取图片尺寸信息
        original_width = result.orig_shape[1] if hasattr(result, "orig_shape") else 640
        original_height = result.orig_shape[0] if hasattr(result, "orig_shape") else 640

        # 转换为Label Studio格式
        label_studio_data = {
            "data": {"image": image_path},
            "predictions": [
                {
                    "model_version": "yolo",
                    "score": 0.0,  # 整体预测分数，可以设置为平均置信度
                    "result": [],
                }
            ],
        }

        # 计算平均置信度作为整体分数
        if yolo_data:
            avg_confidence = sum(item.get("confidence", 0) for item in yolo_data) / len(
                yolo_data
            )
            label_studio_data["predictions"][0]["score"] = round(avg_confidence, 5)

        # 转换每个检测结果
        for idx, detection in enumerate(yolo_data):
            # 根据任务类型创建不同格式的结果项
            if task_type == "segment" and "segments" in detection:
                # 分割任务：使用polygon格式
                segments = detection.get("segments", {})
                x_coords = segments.get("x", [])
                y_coords = segments.get("y", [])

                if x_coords and y_coords and len(x_coords) == len(y_coords):
                    # 将像素坐标转换为百分比坐标
                    points = []
                    for x, y in zip(x_coords, y_coords):
                        x_pct = (x / original_width) * 100
                        y_pct = (y / original_height) * 100
                        points.append([round(x_pct, 2), round(y_pct, 2)])

                    result_item = {
                        "id": f"result_{idx}",
                        "type": "polygonlabels",
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "points": points,
                            "polygonlabels": [detection.get("name", "unknown")],
                        },
                    }

                    # 添加置信度信息
                    if "confidence" in detection:
                        result_item["value"]["confidence"] = round(
                            detection["confidence"], 5
                        )

                    label_studio_data["predictions"][0]["result"].append(result_item)
            elif task_type == "classify":
                # 分类任务：使用choices格式
                result_item = {
                    "id": f"result_{idx}",
                    "type": "choices",
                    "from_name": "choice",
                    "to_name": "image",
                    "value": {"choices": [detection.get("name", "unknown")]},
                }

                # 添加置信度信息
                if "confidence" in detection:
                    result_item["value"]["confidence"] = round(
                        detection["confidence"], 5
                    )

                label_studio_data["predictions"][0]["result"].append(result_item)
            else:
                # 检测任务：使用rectangle格式
                box = detection.get("box", {})

                # 将像素坐标转换为百分比坐标（Label Studio要求）
                x1_pct = (box.get("x1", 0) / original_width) * 100
                y1_pct = (box.get("y1", 0) / original_height) * 100
                x2_pct = (box.get("x2", 0) / original_width) * 100
                y2_pct = (box.get("y2", 0) / original_height) * 100

                width_pct = x2_pct - x1_pct
                height_pct = y2_pct - y1_pct

                # 创建Label Studio格式的结果项
                result_item = {
                    "id": f"result_{idx}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": round(x1_pct, 2),
                        "y": round(y1_pct, 2),
                        "width": round(width_pct, 2),
                        "height": round(height_pct, 2),
                        "rectanglelabels": [detection.get("name", "unknown")],
                    },
                }

                # 添加置信度信息（可选）
                if "confidence" in detection:
                    result_item["value"]["confidence"] = round(
                        detection["confidence"], 5
                    )

                label_studio_data["predictions"][0]["result"].append(result_item)

        # 保存Label Studio格式的JSON文件
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(label_studio_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Label Studio格式JSON结果已保存到: {json_file}")
        return True

    except Exception as e:
        logger.error(f"保存第{result_index}个结果的Label Studio格式JSON文件时出错: {e}")
        return False


def process_results_streaming(model, source, config, predict_args, task_type):
    """流式处理预测结果，避免内存累积

    Args:
        model: YOLO模型实例
        source: 输入源
        config: 配置字典
        predict_args: 预测参数
        task_type: 任务类型

    Returns:
        dict: 处理结果统计
    """
    # 使用stream=True启用流式处理
    predict_args_streaming = predict_args.copy()
    predict_args_streaming["stream"] = True

    # 开始流式预测
    results_generator = model.predict(source=source, **predict_args_streaming)

    processed_count = 0
    json_saved_count = 0
    detection_summary = {}
    total_detections = 0
    save_dir = None

    # 确定保存目录
    if config.get("save_json", False):
        save_dir = Path(
            predict_args.get("project", f"runs/{task_type}")
        ) / predict_args.get("name", "predict")
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JSON文件将保存到: {save_dir}")

    logger.info("开始流式处理预测结果...")

    for result in results_generator:
        try:
            processed_count += 1

            # 统计检测结果
            if task_type == "classify" or (
                hasattr(result, "probs") and result.probs is not None
            ):
                # 分类任务
                if hasattr(result, "probs") and result.probs is not None:
                    top1_idx = result.probs.top1
                    top1_name = result.names.get(top1_idx, f"class_{top1_idx}")
                    detection_summary[top1_name] = (
                        detection_summary.get(top1_name, 0) + 1
                    )
                    total_detections += 1
            elif hasattr(result, "boxes") and result.boxes is not None:
                # 检测/分割任务
                num_detections = len(result.boxes)
                total_detections += num_detections

                # 统计每个类别的检测数量
                if hasattr(result.boxes, "cls") and result.boxes.cls is not None:
                    for cls_id in result.boxes.cls:
                        cls_id = int(cls_id.item())
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")
                        detection_summary[cls_name] = (
                            detection_summary.get(cls_name, 0) + 1
                        )

            # 处理JSON保存
            if config.get("save_json", False) and save_dir:
                if process_single_result(
                    result, config, task_type, save_dir, processed_count
                ):
                    json_saved_count += 1

            # 定期清理内存和报告进度
            if processed_count % 50 == 0:
                cleanup_memory()
                current_memory = psutil.virtual_memory().percent
                logger.info(
                    f"已处理 {processed_count} 张图片，当前内存使用: {current_memory:.1f}%"
                )

                # 如果内存使用超过85%，强制清理
                if current_memory > 85:
                    logger.warning(
                        f"内存使用过高 ({current_memory:.1f}%)，执行强制清理"
                    )
                    cleanup_memory()
                    time.sleep(0.1)  # 短暂暂停让系统回收内存

            # 每1000张图片报告一次详细统计
            if processed_count % 1000 == 0:
                logger.info(
                    f"处理进度: {processed_count} 张图片，总检测数: {total_detections}"
                )
                if detection_summary:
                    logger.info(
                        f"类别统计: {dict(list(detection_summary.items())[:5])}..."
                    )  # 只显示前5个类别

        except Exception as e:
            logger.error(f"处理第 {processed_count} 个结果时出错: {e}")
            continue

    # 最终清理
    cleanup_memory()

    return {
        "processed_count": processed_count,
        "total_detections": total_detections,
        "detection_summary": detection_summary,
        "json_saved_count": json_saved_count,
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLO目标检测预测脚本")

    # 基础配置
    parser.add_argument("--config", type=str, help="预测配置文件路径（可选）")
    parser.add_argument(
        "--model", type=str, required=False, help="YOLO模型权重文件路径"
    )
    parser.add_argument(
        "--source", type=str, required=False, help="输入源：图片路径、目录、URL、视频等"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["detect", "obb", "segment", "classify", "pose"],
        help="YOLO任务类型：detect(检测), obb(旋转框检测), segment(分割), classify(分类), pose(姿态估计)",
    )

    # 预测参数
    parser.add_argument("--name", type=str, default=None, help="实验名称")
    parser.add_argument("--imgsz", type=int, help="输入图像尺寸")
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU阈值")
    parser.add_argument("--max-det", type=int, help="每张图像最大检测数")
    parser.add_argument(
        "--device", type=str, default="", help="推理设备，如：0,1,2,3 或 cpu"
    )
    parser.add_argument("--half", action="store_true", help="使用FP16半精度推理")
    parser.add_argument("--dnn", action="store_true", help="使用OpenCV DNN进行ONNX推理")
    parser.add_argument("--vid-stride", type=int, default=1, help="视频帧步长")
    parser.add_argument(
        "--stream-buffer", action="store_true", help="缓冲所有流帧（默认为True）"
    )

    # 输出配置
    parser.add_argument("--save", action="store_true", help="保存预测结果图像")
    parser.add_argument("--save-txt", action="store_true", help="保存结果到*.txt文件")
    parser.add_argument(
        "--save-conf", action="store_true", help="保存置信度到--save-txt标签"
    )
    parser.add_argument("--save-crop", action="store_true", help="保存裁剪的预测框")
    parser.add_argument("--save-json", action="store_true", help="保存结果到JSON文件")
    parser.add_argument(
        "--data-prefix",
        type=str,
        default=None,
        dest="data_prefix",
        help="JSON文件中data.image字段的路径前缀",
    )
    parser.add_argument(
        "--project", type=str, default="runs/detect", help="保存结果的项目目录"
    )
    parser.add_argument(
        "--exist-ok", action="store_true", help="现有项目/名称可以，不递增"
    )

    # 显示配置
    parser.add_argument("--show", action="store_true", help="显示预测结果")
    parser.add_argument(
        "--show-labels", action="store_true", default=True, help="显示预测标签"
    )
    parser.add_argument(
        "--show-conf", action="store_true", default=True, help="显示预测置信度"
    )
    parser.add_argument(
        "--show-boxes", action="store_true", default=True, help="显示预测框"
    )
    parser.add_argument(
        "--line-width", type=int, default=None, help="边界框线宽（像素）"
    )

    # 其他配置
    parser.add_argument("--verbose", action="store_true", help="显示详细预测日志")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="按类别过滤：--classes 0, 或 --classes 0 2 3",
    )
    parser.add_argument("--agnostic-nms", action="store_true", help="类别无关的NMS")
    parser.add_argument("--augment", action="store_true", help="增强推理")
    parser.add_argument("--visualize", action="store_true", help="可视化特征")
    parser.add_argument("--update", action="store_true", help="更新所有模型")

    return parser.parse_args()


def load_and_override_config(config_path, args):
    """加载配置文件并用命令行参数覆盖"""
    config = {}

    # 如果提供了配置文件，则加载它
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"加载配置文件: {config_path}")
        logger.info(f"配置文件内容: {config}")

    # 用命令行参数覆盖配置文件中的设置
    if args.model is not None:
        config["model"] = args.model
        logger.info(f"命令行覆盖model: {args.model}")

    if args.source is not None:
        config["source"] = args.source
        logger.info(f"命令行覆盖source: {args.source}")

    if args.task is not None:
        config["task"] = args.task
        logger.info(f"命令行设置task: {args.task}")

    if args.name is not None:
        config["name"] = args.name
        logger.info(f"设置name: {args.name}")
    else:
        # 设置默认名称
        timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d%H%M%S")
        config["name"] = f"predict_{timestamp}"

    # 预测参数
    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
        logger.info(f"命令行覆盖imgsz: {args.imgsz}")
    if args.conf is not None:
        config["conf"] = args.conf
        logger.info(f"命令行覆盖conf: {args.conf}")
    if args.iou is not None:
        config["iou"] = args.iou
        logger.info(f"命令行覆盖iou: {args.iou}")
    if args.max_det is not None:
        config["max_det"] = args.max_det
        logger.info(f"命令行覆盖max_det: {args.max_det}")

    if args.device:
        # 处理设备参数
        if args.device.lower() == "cpu":
            config["device"] = "cpu"
        else:
            # 将逗号分隔的设备ID转换为列表
            device_list = [int(d.strip()) for d in args.device.split(",")]
            config["device"] = device_list
        logger.info(f"设置device: {config['device']}")

    if args.half:
        config["half"] = True
        logger.info("启用FP16半精度推理")

    if args.dnn:
        config["dnn"] = True
        logger.info("启用OpenCV DNN")

    if args.vid_stride != 1:
        config["vid_stride"] = args.vid_stride
        logger.info(f"设置vid_stride: {args.vid_stride}")

    if args.stream_buffer:
        config["stream_buffer"] = True
        logger.info("启用流缓冲")

    # 输出配置
    if args.save:
        config["save"] = True
        logger.info("启用结果保存")

    if args.save_txt:
        config["save_txt"] = True
        logger.info("启用结果保存到txt文件")

    if args.save_conf:
        config["save_conf"] = True
        logger.info("启用置信度保存")

    if args.save_crop:
        config["save_crop"] = True
        logger.info("启用裁剪保存")

    if args.save_json:
        config["save_json"] = True
        logger.info("启用结果保存到JSON文件")

    if hasattr(args, "data_prefix") and args.data_prefix is not None:
        config["data_prefix"] = args.data_prefix
        logger.info(f"设置data_prefix: {args.data_prefix}")

    if args.project != "runs/detect":
        config["project"] = args.project
        logger.info(f"设置project: {args.project}")

    if args.exist_ok:
        config["exist_ok"] = True
        logger.info("允许现有项目/名称")

    # 显示配置
    if args.show:
        config["show"] = True
        logger.info("启用结果显示")

    config["show_labels"] = args.show_labels
    config["show_conf"] = args.show_conf
    config["show_boxes"] = args.show_boxes

    if args.line_width is not None:
        config["line_width"] = args.line_width
        logger.info(f"设置line_width: {args.line_width}")

    # 其他配置
    if args.verbose:
        config["verbose"] = True
        logger.info("启用详细日志")

    if args.classes is not None:
        config["classes"] = args.classes
        logger.info(f"设置classes: {args.classes}")

    if args.agnostic_nms:
        config["agnostic_nms"] = True
        logger.info("启用类别无关NMS")

    if args.augment:
        config["augment"] = True
        logger.info("启用增强推理")

    if args.visualize:
        config["visualize"] = True
        logger.info("启用特征可视化")

    if args.update:
        config["update"] = True
        logger.info("启用模型更新")

    return config


def print_prediction_summary(results, source_path, task_type=None):
    """根据任务类型打印预测结果摘要"""
    logger.info("=== 预测结果摘要 ===")
    logger.info(f"输入源: {source_path}")
    logger.info(f"任务类型: {task_type or '自动检测'}")
    logger.info(f"处理图像数: {len(results)}")

    total_detections = 0
    detection_summary = {}

    for i, result in enumerate(results):
        if task_type == "classify" or (
            hasattr(result, "probs") and result.probs is not None
        ):
            # 分类任务
            if hasattr(result, "probs") and result.probs is not None:
                top1_idx = result.probs.top1
                top1_conf = result.probs.top1conf.item()
                top1_name = result.names.get(top1_idx, f"class_{top1_idx}")
                logger.info(
                    f"图像 {i+1}: 分类结果 - {top1_name} (置信度: {top1_conf:.3f})"
                )
                detection_summary[top1_name] = detection_summary.get(top1_name, 0) + 1
                total_detections += 1

                if args.verbose and hasattr(result.probs, "top5"):
                    logger.info(f"  Top5预测:")
                    for j, (idx, conf) in enumerate(
                        zip(result.probs.top5, result.probs.top5conf)
                    ):
                        name = result.names.get(idx.item(), f"class_{idx.item()}")
                        logger.info(f"    {j+1}. {name}: {conf.item():.3f}")

        elif task_type == "pose" or (
            hasattr(result, "keypoints") and result.keypoints is not None
        ):
            # 姿态估计任务
            if hasattr(result, "keypoints") and result.keypoints is not None:
                num_poses = len(result.keypoints)
                total_detections += num_poses
                logger.info(f"图像 {i+1}: 检测到 {num_poses} 个姿态")

                if args.verbose and hasattr(result.keypoints, "xy"):
                    for j, keypoints in enumerate(result.keypoints.xy):
                        logger.info(f"  姿态 {j+1}: {len(keypoints)} 个关键点")

        elif task_type == "segment" or (
            hasattr(result, "masks") and result.masks is not None
        ):
            # 分割任务
            if hasattr(result, "masks") and result.masks is not None:
                num_masks = len(result.masks)
                total_detections += num_masks
                logger.info(f"图像 {i+1}: 检测到 {num_masks} 个分割对象")

                # 统计类别（分割任务也有检测框）
                if (
                    hasattr(result, "boxes")
                    and result.boxes is not None
                    and hasattr(result.boxes, "cls")
                ):
                    for cls_id in result.boxes.cls:
                        cls_id = int(cls_id.item())
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")
                        detection_summary[cls_name] = (
                            detection_summary.get(cls_name, 0) + 1
                        )

        elif task_type == "obb" or (hasattr(result, "obb") and result.obb is not None):
            # OBB 检测任务
            num_detections = len(result.obb)
            total_detections += num_detections
            logger.info(f"图像 {i+1}: 检测到 {num_detections} 个 OBB 目标")

            # 统计每个类别的检测数量
            if hasattr(result.obb, "cls") and result.obb.cls is not None:
                for cls_id in result.obb.cls:
                    cls_id = int(cls_id.item())
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")
                    detection_summary[cls_name] = detection_summary.get(cls_name, 0) + 1

            # 打印详细检测信息
            if args.verbose and hasattr(result.obb, "conf"):
                for j, (conf, cls_id) in enumerate(
                    zip(result.obb.conf, result.obb.cls)
                ):
                    cls_name = result.names.get(
                        int(cls_id.item()), f"class_{int(cls_id.item())}"
                    )
                    logger.info(
                        f"  OBB检测 {j+1}: {cls_name} (置信度: {conf.item():.3f})"
                    )

                    # 打印 OBB 坐标信息
                    if (
                        hasattr(result.obb, "xyxyxyxy")
                        and result.obb.xyxyxyxy is not None
                    ):
                        obb_coords = result.obb.xyxyxyxy[j].cpu().numpy()
                        logger.info(f"    OBB坐标: {obb_coords}")

        else:
            # 普通检测任务（默认）
            if hasattr(result, "boxes") and result.boxes is not None:
                num_detections = len(result.boxes)
                total_detections += num_detections
                logger.info(f"图像 {i+1}: 检测到 {num_detections} 个检测目标")

                # 统计每个类别的检测数量
                if hasattr(result.boxes, "cls") and result.boxes.cls is not None:
                    for cls_id in result.boxes.cls:
                        cls_id = int(cls_id.item())
                        cls_name = result.names.get(cls_id, f"class_{cls_id}")
                        detection_summary[cls_name] = (
                            detection_summary.get(cls_name, 0) + 1
                        )

                # 打印详细检测信息
                if args.verbose and hasattr(result.boxes, "conf"):
                    for j, (conf, cls_id) in enumerate(
                        zip(result.boxes.conf, result.boxes.cls)
                    ):
                        cls_name = result.names.get(
                            int(cls_id.item()), f"class_{int(cls_id.item())}"
                        )
                        logger.info(
                            f"  检测 {j+1}: {cls_name} (置信度: {conf.item():.3f})"
                        )

    if task_type != "classify":
        logger.info(f"总检测数: {total_detections}")
        logger.info(
            f"平均每张图像检测数: {total_detections / len(results) if len(results) > 0 else 0:.2f}"
        )

    if detection_summary:
        logger.info("类别统计:")
        for cls_name, count in sorted(detection_summary.items()):
            logger.info(f"  {cls_name}: {count}")


def get_source_info(source_path):
    """获取输入源信息"""
    source_path = Path(source_path)

    if source_path.is_file():
        if source_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".webp",
        ]:
            return {"type": "image", "path": str(source_path)}
        elif source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]:
            return {"type": "video", "path": str(source_path)}
        else:
            return {"type": "file", "path": str(source_path)}
    elif source_path.is_dir():
        return {"type": "directory", "path": str(source_path)}
    elif str(source_path).startswith(("http://", "https://")):
        return {"type": "url", "path": str(source_path)}
    elif str(source_path).startswith(("rtsp://", "rtmp://")):
        return {"type": "stream", "path": str(source_path)}
    elif str(source_path).isdigit():
        return {"type": "webcam", "path": str(source_path)}
    else:
        return {"type": "unknown", "path": str(source_path)}


def main():
    """主函数"""
    # 解析命令行参数
    global args
    args = parse_args()

    # 记录初始内存使用情况
    initial_memory_usage = psutil.virtual_memory().percent
    logger.info(f"程序启动时内存使用: {initial_memory_usage:.1f}%")

    # 加载并覆盖配置
    config = load_and_override_config(args.config, args)

    # 检查必需参数
    if "model" not in config or not config["model"]:
        logger.error("错误：必须指定模型路径（通过 --model 参数或配置文件）")
        return

    if "source" not in config or not config["source"]:
        logger.error("错误：必须指定输入源（通过 --source 参数或配置文件）")
        return

    # 处理模型路径（支持 OSS）
    logger.info(f"处理模型路径: {config['model']}")

    # 使用处理后的本地路径加载模型
    model = YOLO(config["model"])

    # 确定任务类型：优先使用用户指定的，否则使用模型自动检测的
    model_task_type = model.task
    user_task_type = config.get("task")

    if user_task_type:
        task_type = user_task_type
        logger.info(f"使用用户指定的任务类型: {task_type}")
        if task_type != model_task_type:
            logger.warning(
                f"用户指定任务类型 ({task_type}) 与模型任务类型 ({model_task_type}) 不匹配"
            )
    else:
        task_type = model_task_type
        logger.info(f"自动检测到模型任务类型: {task_type}")

    # 获取输入源信息
    source_info = get_source_info(config["source"])

    # 从预测参数中排除model、source、task、data_prefix，因为它们需要单独处理
    predict_args = {
        k: v
        for k, v in config.items()
        if k not in ["model", "source", "task", "data_prefix"]
    }

    # 根据任务类型设置正确的 project 路径
    if any(
        config.get(save_param, False)
        for save_param in ["save", "save_txt", "save_json", "save_crop"]
    ):
        if "project" not in predict_args:
            predict_args["project"] = f"runs/{task_type}"
            logger.info(f"根据任务类型设置保存路径: runs/{task_type}")

    try:
        logger.info("开始预测...")
        logger.info(f"模型: {config['model']}")
        logger.info(f"输入源: {config['source']}")
        logger.info(f"输入源类型: {source_info['type']}")

        # 使用流式处理执行预测
        logger.info("开始流式预测处理...")
        processing_stats = process_results_streaming(
            model, config["source"], config, predict_args, task_type
        )

        # 打印预测结果摘要
        logger.info(f"预测完成，共处理 {processing_stats['processed_count']} 张图片")
        logger.info(f"总检测数: {processing_stats['total_detections']}")

        if processing_stats["detection_summary"]:
            logger.info(f"类别统计: {processing_stats['detection_summary']}")

        if config.get("save_json", False):
            logger.info(
                f"成功保存 {processing_stats['json_saved_count']} 个Label Studio格式的JSON文件"
            )

        # 创建一个空的results列表用于兼容后续代码
        results = []

        # 检查实际保存的文件并显示保存路径
        if results and hasattr(results[0], "save_dir"):
            save_dir = results[0].save_dir
            saved_files = []

            # 检查各种保存格式的文件是否存在
            if config.get("save", False):
                # 检查是否有图片文件保存
                image_files = list(Path(save_dir).glob("*.jpg")) + list(
                    Path(save_dir).glob("*.png")
                )
                if image_files:
                    saved_files.append(f"图片文件({len(image_files)}个)")

            if config.get("save_txt", False):
                # 检查是否有txt标签文件保存
                labels_dir = Path(save_dir) / "labels"
                if labels_dir.exists():
                    txt_files = list(labels_dir.glob("*.txt"))
                    if txt_files:
                        saved_files.append(f"标签文件({len(txt_files)}个)")

            if config.get("save_json", False):
                # 检查是否有JSON文件保存（包括手动保存的）
                json_files = list(Path(save_dir).glob("*.json"))
                if json_files:
                    saved_files.append(f"JSON文件({len(json_files)}个)")
                else:
                    logger.warning("警告: 启用了save_json但未找到JSON文件")

            if config.get("save_crop", False):
                # 检查是否有裁剪文件保存
                crops_dir = Path(save_dir) / "crops"
                if crops_dir.exists():
                    crop_files = list(crops_dir.rglob("*.jpg")) + list(
                        crops_dir.rglob("*.png")
                    )
                    if crop_files:
                        saved_files.append(f"裁剪文件({len(crop_files)}个)")

            if saved_files:
                logger.info(f"预测结果已保存到: {save_dir}")
                logger.info(f"保存的文件类型: {', '.join(saved_files)}")
            else:
                logger.warning(
                    f"警告: 虽然启用了保存功能，但在 {save_dir} 中未找到保存的文件"
                )

        # 最终内存清理和监控
        cleanup_memory()
        final_memory_usage = psutil.virtual_memory().percent
        logger.info(f"预测完成后内存使用: {final_memory_usage:.1f}%")

        # 计算内存使用改善情况
        memory_improvement = initial_memory_usage - final_memory_usage
        if memory_improvement > 0:
            logger.info(f"内存使用改善: {memory_improvement:.1f}%")
        else:
            logger.warning(f"内存使用增加: {abs(memory_improvement):.1f}%")

        # 记录处理统计信息
        logger.info(f"处理统计: 成功处理 {processing_stats['processed_count']} 张图片")
        if processing_stats.get("json_saved_count", 0) > 0:
            logger.info(f"JSON保存统计: {processing_stats['json_saved_count']} 个文件")

        logger.info("预测完成！")

    except Exception as e:
        # 异常时也进行内存清理
        cleanup_memory()
        error_memory_usage = psutil.virtual_memory().percent
        logger.error(f"预测过程中发生错误: {e}")
        logger.info(f"错误发生时内存使用: {error_memory_usage:.1f}%")
        raise


if __name__ == "__main__":
    main()

# 使用示例:
# python predict_yolo.py --model yolo11n.pt --source bus.jpg --save --save-txt
# python predict_yolo.py --model runs/detect/train/weights/best.pt --source images/ --conf 0.5 --save
# python predict_yolo.py --model yolo11n.pt --source https://ultralytics.com/images/bus.jpg --show
# python predict_yolo.py --model yolo11n.pt --source 0 --show  # 摄像头
# python predict_yolo.py --model yolo11n.pt --source video.mp4 --save --save-json --data-prefix "data-fies=xxx"
# python predict_yolo.py --model yolo11n.pt --source "path/*.jpg" --save-crop --classes 0 2
