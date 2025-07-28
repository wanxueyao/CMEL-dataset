import os
import json

def count_no_match_in_json_files(folder_path):
    # 统计 no match 和 no_match 出现的次数
    no_match_count_1 = 0
    no_match_count_2 = 0

    # 遍历文件夹中的子文件夹
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        # 确保它是文件夹
        if os.path.isdir(subfolder_path):
            json_file_path = os.path.join(subfolder_path, 'aligned_image_entity.json')

            # 确保文件存在
            if os.path.exists(json_file_path):
                try:
                    # 读取并解析 JSON 文件
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # 遍历每个图像项
                        for image_key, image_data in data.items():
                            # 获取 entity_name 和 matched_chunk_entity_name
                            entity_name = image_data.get("entity_name", "")
                            matched_chunk_entity_name = image_data.get("matched_chunk_entity_name", "")

                            # 检查是否为 no match 或 no_match
                            if entity_name == "no match" or entity_name == "no_match":
                                no_match_count_1 += 1
                            if matched_chunk_entity_name == "no match" or matched_chunk_entity_name == "no_match":
                                no_match_count_2 += 1

                except Exception as e:
                    print(f"无法读取或解析文件 {json_file_path}: {e}")
    
    return no_match_count_1,no_match_count_2

# 调用函数，指定文件夹路径
folder_path = '/Users/xueyaowan/Documents/硕士毕业设计/newproject/fusion_research/fusion_dataset/novel'
no_match_count_1,no_match_count_2 = count_no_match_in_json_files(folder_path)

# 打印 no match 和 no_match 的总数
print(f"任务1总共有 {no_match_count_2} 次 'no match' 或 'no_match' 出现；任务3总共有 {no_match_count_1} 次 'no match' 或 'no_match' 出现。")