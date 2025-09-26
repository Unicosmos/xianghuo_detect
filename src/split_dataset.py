#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集切分脚本

基于 ultralytics.data.split 模块实现的通用数据集切分工具
支持目标检测和分类数据集的切分

作者: AI Assistant
创建时间: 2025-01-27
"""

import argparse
import os
import random
import shutil
import sys
import yaml
from pathlib import Path
from typing import List, Tuple, Union

try:
    from ultralytics.data.split import autosplit, split_classify_dataset
    from ultralytics.utils import LOGGER
except ImportError:
    print("错误: 无法导入 ultralytics 库，请确保已安装 ultralytics")
    print("安装命令: pip install ultralytics")
    sys.exit(1)


def detect_dataset_type(dataset_path: Path) -> str:
    """
    检测数据集类型
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        数据集类型: 'detection' 或 'classification'
    """
    dataset_path = Path(dataset_path)
    
    # 检查是否有 images 和 labels 目录 (目标检测格式)
    if (dataset_path / "images").exists() and (dataset_path / "labels").exists():
        return "detection"
    
    # 检查是否有多个子目录，每个子目录包含图片 (分类格式)
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if len(subdirs) > 1:
        # 检查子目录是否包含图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for subdir in subdirs[:3]:  # 检查前3个子目录
            files = list(subdir.iterdir())
            if any(f.suffix.lower() in image_extensions for f in files):
                return "classification"
    
    # 默认返回检测类型
    return "detection"


def split_detection_dataset(
    dataset_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    annotated_only: bool = True,
    copy_files: bool = True
) -> Path:
    """
    切分目标检测数据集
    
    Args:
        dataset_path: 原始数据集路径
        output_path: 输出路径，如果为 None 则在原路径基础上添加 '_split' 后缀
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        annotated_only: 是否只包含有标注的图片
        copy_files: 是否复制文件，False 则只生成文件列表
        
    Returns:
        切分后的数据集路径
    """
    dataset_path = Path(dataset_path)
    
    if output_path is None:
        # 构造包含比例的目录名称
        ratio_str = f"{int(train_ratio*100)}-{int(val_ratio*100)}"
        if test_ratio > 0:
            ratio_str += f"-{int(test_ratio*100)}"
        output_path = dataset_path.parent / f"{dataset_path.name}_split_{ratio_str}"
    else:
        output_path = Path(output_path)
    
    # 检查比例是否合理
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须为 1.0，当前为 {total_ratio}")
    
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    
    # 使用 ultralytics 的 autosplit 功能生成文件列表
    LOGGER.info(f"开始切分目标检测数据集: {dataset_path}")
    
    # 设置权重
    weights = (train_ratio, val_ratio, test_ratio)
    
    # 使用 autosplit 生成文件列表
    autosplit(
        path=images_dir,
        weights=weights,
        annotated_only=annotated_only
    )
    
    if copy_files:
        # 创建输出目录结构
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = ['train', 'val', 'test']
        split_files = [
            images_dir.parent / "autosplit_train.txt",
            images_dir.parent / "autosplit_val.txt", 
            images_dir.parent / "autosplit_test.txt"
        ]
        
        for split, split_file in zip(splits, split_files):
            if not split_file.exists():
                continue
                
            # 创建目录
            split_images_dir = output_path / split / "images"
            split_labels_dir = output_path / split / "labels"
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # 读取文件列表并复制文件
            with open(split_file, 'r', encoding='utf-8') as f:
                for line in f:
                    img_path = line.strip()
                    if img_path.startswith('./'):
                        img_path = img_path[2:]
                    
                    src_img = dataset_path / img_path
                    if src_img.exists():
                        # 复制图片
                        dst_img = split_images_dir / src_img.name
                        shutil.copy2(src_img, dst_img)
                        
                        # 复制对应的标签文件
                        label_name = src_img.stem + '.txt'
                        src_label = labels_dir / label_name
                        if src_label.exists():
                            dst_label = split_labels_dir / label_name
                            shutil.copy2(src_label, dst_label)
        
        # 复制其他文件
        for file_name in ['classes.txt', 'notes.json']:
            src_file = dataset_path / file_name
            if src_file.exists():
                shutil.copy2(src_file, output_path / file_name)
        
        # 生成新的 data.yaml 文件
        generate_data_yaml(
            output_path=output_path,
            dataset_type="detection",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            original_dataset_path=dataset_path
        )
        
        # 清理临时文件
        for split_file in split_files:
            if split_file.exists():
                split_file.unlink()
    
    LOGGER.info(f"数据集切分完成: {output_path}")
    return output_path


def split_classification_dataset_wrapper(
    dataset_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    train_ratio: float = 0.8
) -> Path:
    """
    切分分类数据集的包装函数
    
    Args:
        dataset_path: 原始数据集路径
        output_path: 输出路径
        train_ratio: 训练集比例
        
    Returns:
        切分后的数据集路径
    """
    if output_path is not None:
        # 如果指定了输出路径，需要先复制数据集
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        temp_path = output_path.parent / f"{dataset_path.name}_temp"
        
        if temp_path.exists():
            shutil.rmtree(temp_path)
        shutil.copytree(dataset_path, temp_path)
        
        # 使用 ultralytics 的函数切分
        result_path = split_classify_dataset(temp_path, train_ratio)
        
        # 重命名到目标路径
        if output_path.exists():
            shutil.rmtree(output_path)
        result_path.rename(output_path)
        
        # 生成 data.yaml 文件
        generate_data_yaml(
            output_path=output_path,
            dataset_type="classification",
            train_ratio=train_ratio,
            val_ratio=1-train_ratio,
            test_ratio=0.0,
            original_dataset_path=dataset_path
        )
        
        # 清理临时目录
        if temp_path.exists():
            shutil.rmtree(temp_path)
            
        return output_path
    else:
        # 直接使用 ultralytics 的函数
        result_path = split_classify_dataset(dataset_path, train_ratio)
        
        # 生成 data.yaml 文件
        generate_data_yaml(
            output_path=result_path,
            dataset_type="classification",
            train_ratio=train_ratio,
            val_ratio=1-train_ratio,
            test_ratio=0.0,
            original_dataset_path=dataset_path
        )
        
        return result_path


def count_dataset_files(dataset_path: Path, dataset_type: str) -> dict:
    """
    统计数据集文件数量
    
    Args:
        dataset_path: 数据集路径
        dataset_type: 数据集类型
        
    Returns:
        文件统计信息
    """
    stats = {}
    
    if dataset_type == "detection":
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        if images_dir.exists():
            image_files = list(images_dir.glob("*.*"))
            image_files = [f for f in image_files if f.suffix.lower() in 
                          {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}]
            stats['images'] = len(image_files)
        else:
            stats['images'] = 0
            
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            stats['labels'] = len(label_files)
        else:
            stats['labels'] = 0
            
    elif dataset_type == "classification":
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        stats['classes'] = len(class_dirs)
        stats['images'] = 0
        
        for class_dir in class_dirs:
            image_files = list(class_dir.glob("*.*"))
            image_files = [f for f in image_files if f.suffix.lower() in 
                          {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}]
            stats['images'] += len(image_files)
    
    return stats


def generate_data_yaml(
    output_path: Path,
    dataset_type: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    original_dataset_path: Path = None
) -> None:
    """
    生成 data.yaml 配置文件
    
    Args:
        output_path: 切分后的数据集路径
        dataset_type: 数据集类型
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        original_dataset_path: 原始数据集路径
    """
    data_yaml_path = output_path / "data.yaml"
    
    # 基本配置
    config = {
        "path": str(output_path.absolute()),
        "train": "train/images" if dataset_type == "detection" else "train",
        "val": "val/images" if dataset_type == "detection" else "val"
    }
    
    # 如果有测试集，添加测试集配置
    if test_ratio > 0 and (output_path / "test").exists():
        config["test"] = "test/images" if dataset_type == "detection" else "test"
    
    # 获取类别信息
    if dataset_type == "detection":
        # 尝试从原始数据集读取类别信息
        classes = []
        
        # 首先尝试从原始数据集的 data.yaml 读取
        if original_dataset_path and (original_dataset_path / "data.yaml").exists():
            try:
                with open(original_dataset_path / "data.yaml", 'r', encoding='utf-8') as f:
                    original_config = yaml.safe_load(f)
                    if 'names' in original_config:
                        if isinstance(original_config['names'], dict):
                            classes = list(original_config['names'].values())
                        elif isinstance(original_config['names'], list):
                            classes = original_config['names']
            except Exception as e:
                LOGGER.warning(f"无法读取原始 data.yaml: {e}")
        
        # 如果没有找到，尝试从 classes.txt 读取
        if not classes:
            classes_file = original_dataset_path / "classes.txt" if original_dataset_path else output_path / "classes.txt"
            if classes_file.exists():
                try:
                    with open(classes_file, 'r', encoding='utf-8') as f:
                        classes = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    LOGGER.warning(f"无法读取 classes.txt: {e}")
        
        # 如果还是没有找到，使用默认类别
        if not classes:
            classes = ["object"]
        
        # 设置类别信息
        config["nc"] = len(classes)
        config["names"] = {i: name for i, name in enumerate(classes)}
        
    elif dataset_type == "classification":
        # 对于分类数据集，从训练目录获取类别
        train_dir = output_path / "train"
        if train_dir.exists():
            class_dirs = [d.name for d in train_dir.iterdir() if d.is_dir()]
            class_dirs.sort()  # 确保顺序一致
            config["nc"] = len(class_dirs)
            config["names"] = {i: name for i, name in enumerate(class_dirs)}
        else:
            config["nc"] = 1
            config["names"] = {0: "unknown"}
    
    # 写入 data.yaml 文件
    try:
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            # 添加注释
            f.write(f"# Ultralytics YOLO 🚀, AGPL-3.0 license\n")
            f.write(f"# Dataset configuration for {output_path.name}\n")
            f.write(f"# Generated automatically by split_dataset.py\n")
            f.write(f"# Split ratios - train: {train_ratio:.1%}, val: {val_ratio:.1%}")
            if test_ratio > 0:
                f.write(f", test: {test_ratio:.1%}")
            f.write(f"\n\n")
            
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        LOGGER.info(f"已生成 data.yaml 配置文件: {data_yaml_path}")
        
    except Exception as e:
        LOGGER.error(f"生成 data.yaml 失败: {e}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="数据集切分工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 切分目标检测数据集 (默认 8:2 比例)
  python split_dataset.py /path/to/detection/dataset
  
  # 切分分类数据集 (指定比例 7:3)
  python split_dataset.py /path/to/classification/dataset --split-ratio 0.7,0.3
  
  # 指定输出路径
  python split_dataset.py /path/to/dataset --output /path/to/output
  
  # 包含测试集的三分割 (7:2:1)
  python split_dataset.py /path/to/dataset --split-ratio 0.7,0.2,0.1
        """
    )
    
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出路径 (默认在原路径基础上添加 '_split' 后缀)"
    )
    
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.8,0.2",
        help="数据集切分比例，用逗号分隔。如果只有一个逗号，表示train_ratio,val_ratio；如果有两个逗号，表示train_ratio,val_ratio,test_ratio (默认: 0.8,0.2)"
    )
    
    parser.add_argument(
        "--dataset-type",
        choices=["detection", "classification", "auto"],
        default="auto",
        help="数据集类型 (默认: auto 自动检测)"
    )
    
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="仅包含有标注的图片 (仅对目标检测数据集有效)"
    )
    
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="不复制文件，仅生成文件列表 (仅对目标检测数据集有效)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    
    args = parser.parse_args()
    
    # 解析切分比例
    try:
        ratios = [float(x.strip()) for x in args.split_ratio.split(',')]
        if len(ratios) == 2:
            train_ratio, val_ratio = ratios
            test_ratio = 0.0
        elif len(ratios) == 3:
            train_ratio, val_ratio, test_ratio = ratios
        else:
            print(f"错误: 切分比例格式不正确，应为 'train,val' 或 'train,val,test'，当前为: {args.split_ratio}")
            sys.exit(1)
        
        # 检查比例是否合理
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"错误: 比例总和必须为 1.0，当前为 {total_ratio}")
            sys.exit(1)
            
        if any(r < 0 for r in ratios):
            print(f"错误: 比例不能为负数")
            sys.exit(1)
            
    except ValueError as e:
        print(f"错误: 无法解析切分比例 '{args.split_ratio}': {e}")
        sys.exit(1)
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 检查数据集路径
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    # 检测数据集类型
    if args.dataset_type == "auto":
        dataset_type = detect_dataset_type(dataset_path)
        print(f"自动检测数据集类型: {dataset_type}")
    else:
        dataset_type = args.dataset_type
    
    # 统计原始数据集信息
    stats = count_dataset_files(dataset_path, dataset_type)
    print(f"\n原始数据集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    try:
        if dataset_type == "detection":
            # 切分目标检测数据集
            output_path = split_detection_dataset(
                dataset_path=dataset_path,
                output_path=args.output,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                annotated_only=args.annotated_only,
                copy_files=not args.no_copy
            )
        elif dataset_type == "classification":
            # 切分分类数据集
            if test_ratio > 0:
                print("警告: 分类数据集切分暂不支持测试集，将忽略测试集比例")
            
            output_path = split_classification_dataset_wrapper(
                dataset_path=dataset_path,
                output_path=args.output,
                train_ratio=train_ratio
            )
        else:
            print(f"错误: 不支持的数据集类型: {dataset_type}")
            sys.exit(1)
        
        print(f"\n✅ 数据集切分完成!")
        print(f"输出路径: {output_path}")
        
        # 统计切分后的数据集信息
        if output_path.exists():
            print(f"\n切分后数据集结构:")
            for split_dir in ['train', 'val', 'test']:
                split_path = output_path / split_dir
                if split_path.exists():
                    if dataset_type == "detection":
                        images_count = len(list((split_path / "images").glob("*.*")))
                        labels_count = len(list((split_path / "labels").glob("*.txt")))
                        print(f"  {split_dir}: {images_count} 图片, {labels_count} 标签")
                    elif dataset_type == "classification":
                        total_images = 0
                        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
                        for class_dir in class_dirs:
                            images = list(class_dir.glob("*.*"))
                            total_images += len(images)
                        print(f"  {split_dir}: {total_images} 图片, {len(class_dirs)} 类别")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()