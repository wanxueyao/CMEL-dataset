import os
import json

def count_images_in_folder(root_folder):
    # 初始化图像计数器
    image_count = 0

    # 遍历根文件夹中的每个子文件夹
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # 确保当前路径是一个文件夹
        if os.path.isdir(subfolder_path):
            json_file_path = os.path.join(subfolder_path, 'kv_store_image_data.json')
            
            # 检查文件是否存在
            if os.path.exists(json_file_path):
                try:
                    # 打开并读取 JSON 文件
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 统计 JSON 文件中的图像数量
                        image_count += len(data)
                except Exception as e:
                    print(f"无法读取文件 {json_file_path}: {e}")

    return image_count

root_folder = './fusion_research/fusion_dataset/paper'
image_count = count_images_in_folder(root_folder)

print(f"图像总数: {image_count}")