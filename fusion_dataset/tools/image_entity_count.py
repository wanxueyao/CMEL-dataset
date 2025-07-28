import os
import json

def count_image_entities_in_json_files(folder_path):
    # 创建一个列表，用于记录每个子文件夹的图像实体数量
    total_image_entities_list = []

    # 遍历文件夹中的所有子文件夹
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        # 确保它是文件夹
        if os.path.isdir(subfolder_path):
            json_file_path = os.path.join(subfolder_path, 'kv_store_image_knowledge_graph.json')

            # 确保文件存在
            if os.path.exists(json_file_path):
                try:
                    # 读取并解析 JSON 文件
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # 统计每个子文件夹中的图像实体数量
                        total_image_entities_count = 0
                        for image_key, image_entities in data.items():
                            # 获取实体列表的长度，统计实体数量
                            total_image_entities_count += len(image_entities)

                        # 将当前子文件夹的图像实体数量添加到列表中
                        total_image_entities_list.append(total_image_entities_count)

                except Exception as e:
                    print(f"无法读取或解析文件 {json_file_path}: {e}")
    
    return total_image_entities_list

# 调用函数，指定文件夹路径
folder_path = '/Users/xueyaowan/Documents/硕士毕业设计/newproject/fusion_research/fusion_dataset/news'
total_image_entities_list = count_image_entities_in_json_files(folder_path)

# 输出最终的图像实体数量列表
print(f"所有子文件夹中的图像实体总数列表为: {total_image_entities_list}")
