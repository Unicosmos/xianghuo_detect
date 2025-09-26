#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""数据集管理工具包
统一管理多个数据集处理功能，包括：
- 复制标签文件
- 删除不匹配的图片
- 选择匹配/不匹配的文件
- 备份同名文件

使用示例:
# 复制标签文件
python dataset_toolkit.py copy-labels --source ./labels_source --target ./labels_target

# 删除与JSON不匹配的图片
python dataset_toolkit.py remove-unmatched --images-dir ./images --reference-dir ./jsons --reference-type json

# 复制与标签匹配的图片
python dataset_toolkit.py copy-matched --source-dir ./images --reference-dir ./labels --output-dir ./matched_images

# 复制与标签不匹配的图片
python dataset_toolkit.py copy-matched --source-dir ./images --reference-dir ./labels --output-dir ./unmatched_images --no-match

# 移动同名标签文件到备份目录
python dataset_toolkit.py move-matched --source-dir ./labels1 --reference-dir ./labels2 --backup-dir ./backup
"""

import os
import shutil
import argparse
from pathlib import Path

# 设置中文显示
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 文件类型配置
def get_extensions(file_type):
    """根据文件类型返回对应的扩展名列表"""
    extensions_map = {
        'image': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
        'label': ['.txt'],
        'json': ['.json']
    }
    return extensions_map.get(file_type, [])

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建目录: {directory}")
    return True

def get_filenames_without_extension(directory, *extensions):
    """获取目录中指定扩展名的所有文件名（不含扩展名）"""
    if not os.path.exists(directory):
        print(f"警告: 目录 '{directory}' 不存在")
        return {}
    
    # 处理可变参数，如果传入的是列表则展开
    ext_list = []
    for ext in extensions:
        if isinstance(ext, list):
            ext_list.extend(ext)
        else:
            ext_list.append(ext)
    
    filenames = {}
    for f in os.listdir(directory):
        file_path = os.path.join(directory, f)
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(f)
            if ext.lower() in ext_list:
                filenames[name] = f
    
    return filenames

def compare_directories(source_dir, reference_dir, source_extensions, reference_extensions):
    """比较两个目录中的文件，返回匹配和不匹配的文件名集合"""
    # 获取两个文件夹中的文件名（不含扩展名）
    source_files = get_filenames_without_extension(source_dir, source_extensions)
    reference_files = set(get_filenames_without_extension(reference_dir, reference_extensions).keys())
    
    print(f"在源目录找到 {len(source_files)} 个文件")
    print(f"在参考目录找到 {len(reference_files)} 个文件")
    
    # 找到匹配和不匹配的文件
    matched_files = set(source_files.keys()) & reference_files
    unmatched_files = set(source_files.keys()) - reference_files
    
    print(f"找到 {len(matched_files)} 个匹配的文件")
    print(f"找到 {len(unmatched_files)} 个不匹配的文件")
    
    return source_files, matched_files, unmatched_files

def copy_files(source_dir, target_dir, file_pattern='*.txt'):
    """将源目录中的文件复制到目标目录"""
    # 转换为Path对象以便跨平台兼容
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 检查源目录是否存在
    if not source_path.exists():
        print(f"错误: 源目录 '{source_dir}' 不存在")
        return False
    
    # 创建目标目录（如果不存在）
    try:
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"目标目录 '{target_dir}' 已创建或已存在")
    except Exception as e:
        print(f"创建目标目录时出错: {e}")
        return False
    
    # 获取源目录中的所有匹配文件
    files = list(source_path.glob(file_pattern))
    if not files:
        print(f"警告: 源目录 '{source_dir}' 中没有找到匹配 '{file_pattern}' 的文件")
        return False
    
    # 复制文件
    success_count = 0
    fail_count = 0
    
    print(f"开始复制 {len(files)} 个文件...")
    
    for file in files:
        try:
            target_file = target_path / file.name
            shutil.copy2(file, target_file)
            success_count += 1
        except Exception as e:
            print(f"复制文件 '{file.name}' 时出错: {e}")
            fail_count += 1
    
    print(f"文件复制完成! 成功: {success_count}, 失败: {fail_count}")
    return True

def copy_labels(source_dir, target_dir):
    """将源目录中的所有标签文件复制到目标目录"""
    return copy_files(source_dir, target_dir, '*.txt')

def remove_files(files_dict, directory, files_to_remove, backup_dir=None):
    """删除指定的文件，可选备份"""
    # 如果设置了备份目录，先备份要删除的文件
    if backup_dir and files_to_remove:
        ensure_directory_exists(backup_dir)
        print(f"正在备份文件到 {backup_dir}...")
        
        backed_up_count = 0
        for filename in files_to_remove:
            original_name = files_dict[filename]
            source_path = os.path.join(directory, original_name)
            dest_path = os.path.join(backup_dir, original_name)
            try:
                shutil.copy2(source_path, dest_path)
                backed_up_count += 1
            except Exception as e:
                print(f"备份文件 '{original_name}' 失败: {e}")
        print(f"成功备份 {backed_up_count} 个文件")
    
    # 删除文件
    removed_count = 0
    for filename in files_to_remove:
        original_name = files_dict[filename]
        file_path = os.path.join(directory, original_name)
        
        try:
            os.remove(file_path)
            removed_count += 1
            print(f"已删除: {original_name}")
        except Exception as e:
            print(f"删除失败 {original_name}: {e}")
    
    print(f"成功删除 {removed_count} 个文件")
    
    # 返回剩余的文件数量
    remaining_count = len(files_dict) - removed_count
    print(f"剩余文件数量: {remaining_count}")
    
    return list(files_to_remove), removed_count

def remove_unmatched_images(images_dir, reference_dir, reference_extensions, backup_dir=None):
    """删除图片文件夹中与参考文件夹不同名的图片文件"""
    image_extensions = get_extensions('image')
    
    # 比较目录获取匹配和不匹配的文件
    image_files, _, unmatched_files = compare_directories(
        images_dir, reference_dir, image_extensions, reference_extensions
    )
    
    # 删除不匹配的文件
    return remove_files(image_files, images_dir, unmatched_files, backup_dir)

def copy_files_to_directory(files_dict, source_dir, target_files, output_dir):
    """将指定的文件复制到输出目录"""
    # 确保输出目录存在
    ensure_directory_exists(output_dir)
    
    # 复制文件到输出目录
    copied_count = 0
    for filename in target_files:
        # 获取原始文件名（包含扩展名）
        original_name = files_dict[filename]
        source_path = os.path.join(source_dir, original_name)
        dest_path = os.path.join(output_dir, original_name)
        
        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"复制文件 '{original_name}' 失败: {e}")
    
    print(f"成功复制 {copied_count} 个文件到 {output_dir}")
    
    # 返回处理的文件列表
    return list(target_files)

def copy_matching_files(source_dir, reference_dir, source_extensions, reference_extensions, output_dir, match=True):
    """找到源文件夹中与参考文件夹同名/不同名的文件，并复制到输出目录"""
    # 比较目录获取匹配和不匹配的文件
    source_files, matched_files, unmatched_files = compare_directories(
        source_dir, reference_dir, source_extensions, reference_extensions
    )
    
    # 选择要复制的文件
    if match:
        target_files = matched_files
        print(f"准备复制 {len(target_files)} 个匹配的文件")
    else:
        target_files = unmatched_files
        print(f"准备复制 {len(target_files)} 个不匹配的文件")
    
    # 复制文件
    return copy_files_to_directory(source_files, source_dir, target_files, output_dir)

def move_files_to_directory(files_dict, source_dir, target_files, backup_dir):
    """将指定的文件移动到备份目录"""
    # 确保备份目录存在
    ensure_directory_exists(backup_dir)
    
    # 移动文件到备份目录
    moved_count = 0
    for filename in target_files:
        # 获取原始文件名（包含扩展名）
        original_filename = files_dict[filename]
        source_path = os.path.join(source_dir, original_filename)
        dest_path = os.path.join(backup_dir, original_filename)
        
        try:
            shutil.move(source_path, dest_path)
            moved_count += 1
            print(f"已移动: {original_filename}")
        except Exception as e:
            print(f"移动文件 '{original_filename}' 失败: {e}")
    
    print(f"成功移动 {moved_count} 个文件到备份目录 {backup_dir}")
    
    # 返回移动的文件列表
    return list(target_files)

def move_matching_files(source_dir, reference_dir, extensions, backup_dir):
    """将源文件夹中与参考文件夹同名的文件移动到备份目录"""
    # 比较目录获取匹配的文件
    source_files, matched_files, _ = compare_directories(
        source_dir, reference_dir, extensions, extensions
    )
    
    # 移动匹配的文件
    return move_files_to_directory(source_files, source_dir, matched_files, backup_dir)

def print_file_list(files, max_display=10):
    """打印文件列表，限制显示数量"""
    if files:
        print("\n已处理的文件列表：")
        for filename in sorted(files)[:max_display]:
            print(f"  {filename}")
        if len(files) > max_display:
            print(f"  ... 还有 {len(files) - max_display} 个文件")

def main():
    # 创建主解析器
    parser = argparse.ArgumentParser(
        description='数据集管理工具包',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 复制标签文件
  python dataset_toolkit.py copy-labels --source ./labels_source --target ./labels_target
  
  # 删除与JSON不匹配的图片
  python dataset_toolkit.py remove-unmatched --images-dir ./images --reference-dir ./jsons --reference-type json
  
  # 复制与标签匹配的图片
  python dataset_toolkit.py copy-matched --source-dir ./images --reference-dir ./labels --output-dir ./matched_images
  
  # 复制与标签不匹配的图片
  python dataset_toolkit.py copy-matched --source-dir ./images --reference-dir ./labels --output-dir ./unmatched_images --no-match
  
  # 移动同名标签文件到备份目录
  python dataset_toolkit.py move-matched --source-dir ./labels1 --reference-dir ./labels2 --backup-dir ./backup
        """)
    
    subparsers = parser.add_subparsers(dest='command', help='支持的命令')
    
    # 复制标签文件命令
    copy_labels_parser = subparsers.add_parser('copy-labels', help='复制所有标签文件到目标目录')
    copy_labels_parser.add_argument('--source', type=str, required=True, help='源标签目录路径')
    copy_labels_parser.add_argument('--target', type=str, required=True, help='目标标签目录路径')
    
    # 删除不匹配图片命令
    remove_unmatched_parser = subparsers.add_parser('remove-unmatched', help='删除与参考文件不匹配的图片')
    remove_unmatched_parser.add_argument('--images-dir', type=str, required=True, help='图片文件夹路径')
    remove_unmatched_parser.add_argument('--reference-dir', type=str, required=True, help='参考文件夹路径')
    remove_unmatched_parser.add_argument('--reference-type', type=str, choices=['json', 'label'], default='json', help='参考文件类型（默认：json）')
    remove_unmatched_parser.add_argument('--backup-dir', type=str, help='备份目录路径（可选）')
    
    # 复制匹配文件命令
    copy_matched_parser = subparsers.add_parser('copy-matched', help='复制与参考文件匹配或不匹配的文件')
    copy_matched_parser.add_argument('--source-dir', type=str, required=True, help='源文件目录路径')
    copy_matched_parser.add_argument('--reference-dir', type=str, required=True, help='参考文件目录路径')
    copy_matched_parser.add_argument('--source-type', type=str, choices=['image', 'label'], default='image', help='源文件类型（默认：image）')
    copy_matched_parser.add_argument('--reference-type', type=str, choices=['image', 'label', 'json'], default='label', help='参考文件类型（默认：label）')
    copy_matched_parser.add_argument('--output-dir', type=str, required=True, help='输出目录路径')
    copy_matched_parser.add_argument('--match', action='store_true', default=True, help='复制匹配的文件（默认）')
    copy_matched_parser.add_argument('--no-match', dest='match', action='store_false', help='复制不匹配的文件')
    
    # 移动匹配文件命令
    move_matched_parser = subparsers.add_parser('move-matched', help='移动与参考文件匹配的文件到备份目录')
    move_matched_parser.add_argument('--source-dir', type=str, required=True, help='源文件目录路径')
    move_matched_parser.add_argument('--reference-dir', type=str, required=True, help='参考文件目录路径')
    move_matched_parser.add_argument('--file-type', type=str, choices=['label'], default='label', help='文件类型（默认：label）')
    move_matched_parser.add_argument('--backup-dir', type=str, required=True, help='备份目录路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 根据命令执行相应的功能
    if args.command == 'copy-labels':
        result = copy_labels(args.source, args.target)
        if result:
            print("标签文件复制任务已成功完成")
        else:
            print("标签文件复制任务失败")
    
    elif args.command == 'remove-unmatched':
        # 根据参考文件类型设置扩展名
        reference_extensions = get_extensions(args.reference_type)
        
        # 确认操作
        print("警告：此操作将删除图片文件夹中与参考文件夹不同名的图片文件！")
        if args.backup_dir:
            print(f"文件将备份到: {args.backup_dir}")
        
        confirm = input("确定要继续吗？(y/n): ")
        if confirm.lower() != "y":
            print("操作已取消")
            return
        
        # 执行操作
        unmatched_files, removed_count = remove_unmatched_images(
            args.images_dir, args.reference_dir, reference_extensions, args.backup_dir
        )
        
        # 显示处理结果
        print_file_list(unmatched_files)
    
    elif args.command == 'copy-matched':
        # 设置源文件和参考文件扩展名
        source_extensions = get_extensions(args.source_type)
        reference_extensions = get_extensions(args.reference_type)
        
        # 执行操作
        target_files = copy_matching_files(
            args.source_dir, args.reference_dir, 
            source_extensions, reference_extensions, 
            args.output_dir, args.match
        )
        
        # 显示处理结果
        print_file_list(target_files)
    
    elif args.command == 'move-matched':
        # 设置文件扩展名
        extensions = get_extensions(args.file_type)
        
        # 执行操作
        moved_files = move_matching_files(
            args.source_dir, args.reference_dir, extensions, args.backup_dir
        )
        
        # 显示处理结果
        print_file_list(moved_files)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()