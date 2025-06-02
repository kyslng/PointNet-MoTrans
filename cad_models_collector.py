import os
import numpy as np
import trimesh
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, 'ModelNet40'))  # 确保路径正确
OUTPUT_PATH = os.path.abspath(os.path.join(BASE_DIR, '../dataset/cadmulobj/cad_models.npy'))

# 配置参数
TARGET_CLASSES = ['airplane', 'bowl', 'desk', 'keyboard', 'person', 'sofa', 'tv_stand',
                  'bathtub', 'car', 'door', 'lamp', 'piano', 'stairs', 'vase',
                  'bed', 'chair', 'dresser', 'laptop', 'plant', 'stool', 'wardrobe',
                  'bench', 'cone', 'flower_pot', 'mantel', 'radio', 'table', 'xbox',
                  'bookshelf', 'cup', 'glass_box', 'monitor', 'range_hood', 'tent',
                  'bottle', 'curtain', 'guitar', 'night_stand', 'sink', 'toilet']
NUM_POINTS = 2048
NORMALIZE = True

def load_and_process_off(file_path):
    """OFF文件处理器"""
    try:
        # 强制指定为三角网格类型
        mesh = trimesh.load(file_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("非三角网格类型")
            
        # 验证面片数量
        if len(mesh.faces) == 0:
            print(f"警告：{file_path} 无有效面片，跳过")
            return None
            
        # 采样点云
        points, _ = trimesh.sample.sample_surface(mesh, NUM_POINTS)
        
        # 强制形状验证
        if points.shape != (NUM_POINTS, 3):
            print(f"采样异常：{file_path} 形状为 {points.shape}，期望({NUM_POINTS}, 3)")
            return None
            
        # 归一化处理
        if NORMALIZE:
            centroid = np.mean(points, axis=0)
            points -= centroid
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 1e-6:  # 避免除以零
                points /= max_dist
        return points
        
    except Exception as e:
        print(f"文件处理失败：{file_path}\n错误详情：{str(e)}")
        return None

def generate_dataset():
    # 创建输出目录
    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建类别映射字典
    class_to_idx = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}

    # 存储每个类的数据
    collected_data_dict = {cls: [] for cls in TARGET_CLASSES}
    
    # collected_data = []
    # collected_labels = []

    # 遍历每个目标类别
    for class_name, class_idx in tqdm(class_to_idx.items(), desc="Processing classes"):
        class_dir = os.path.join(DATA_ROOT, class_name)
        if not os.path.exists(class_dir):
            print(f"警告：类别目录 {class_dir} 不存在，跳过")
            continue

        # 遍历train和test目录
        for split in ['train', 'test']:
            split_dir = os.path.join(class_dir, split)
            if not os.path.exists(split_dir):
                continue

            # 遍历所有OFF文件
            for file_name in tqdm(os.listdir(split_dir), desc=f"{class_name}-{split}", leave=False):
                if not file_name.endswith('.off'):
                    continue

                file_path = os.path.join(split_dir, file_name)
                points = load_and_process_off(file_path)
                
                if points is not None:
                    collected_data_dict[class_name].append(points)
                    # collected_data.append(points)
                    # collected_labels.append(class_idx)
    
    # 过滤空数据 
    # valid_indices = [i for i, p in enumerate(collected_data_dict) if p is not None]
    # collected_data_dict = [collected_data_dict[i] for i in valid_indices]
    # valid_indices = [i for i, p in enumerate(collected_data) if p is not None]
    # collected_data = [collected_data[i] for i in valid_indices]
    # collected_labels = [collected_labels[i] for i in valid_indices]
    
    # 将每个类别的数据转换为numpy数组
    for class_name in collected_data_dict:
        collected_data_dict[class_name] = np.array(collected_data_dict[class_name], dtype=np.float32)
    # 转换为numpy数组
    # data_array = np.array(collected_data, dtype=np.float32)
    # labels_array = np.array(collected_labels, dtype=np.int64)
    
    # 保存数据集
    np.save(OUTPUT_PATH, collected_data_dict)
    # dataset_dict = {
    #     'data': data_array,
    #     'labels': labels_array,
    #     'class_map': TARGET_CLASSES
    # }
    # np.save(OUTPUT_PATH, dataset_dict)

if __name__ == '__main__':
    generate_dataset()
    print(f"数据集已保存到 {OUTPUT_PATH}")