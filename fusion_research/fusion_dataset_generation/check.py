import json
import os
import re

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_entity_name(entity_name):
    """清理实体名称，去除多余的引号和转义字符"""
    # 去掉引号
    cleaned_name = entity_name.strip('"')  # 去掉两边的引号

    # 使用正则表达式清理无效的转义字符
    # 仅替换有效的 Unicode 转义字符
    cleaned_name = re.sub(r'\\u[0-9A-Fa-f]{4}', lambda match: match.group(0).encode('utf-8').decode('unicode_escape'), cleaned_name)

    # 处理其余的可能的无效转义字符
    return cleaned_name

def check_entities_in_knowledge_graph(fusion_dataset_path, output_file):
    # 直接指定 news, novel 和 paper 子文件夹
    categories = ['news', 'novel', 'paper']

    # 打开输出文件进行写入
    with open(output_file, 'w', encoding='utf-8') as output:
        for category in categories:
            category_path = os.path.join(fusion_dataset_path, category)

            # 获取该类别下所有子文件夹（每个子文件夹表示一个文档）
            for folder_name in os.listdir(category_path):
                folder_path = os.path.join(category_path, folder_name)

                # 检查文件夹是否包含必要的 JSON 文件
                if os.path.isdir(folder_path):
                    aligned_text_entity_path = os.path.join(folder_path, 'aligned_text_entity.json')
                    kv_store_chunk_knowledge_graph_path = os.path.join(folder_path, 'kv_store_chunk_knowledge_graph.json')
                    kv_store_image_knowledge_graph_path = os.path.join(folder_path, 'kv_store_image_knowledge_graph.json')

                    # 确保这三个文件都存在
                    if os.path.exists(aligned_text_entity_path) and \
                       os.path.exists(kv_store_chunk_knowledge_graph_path) and \
                       os.path.exists(kv_store_image_knowledge_graph_path):

                        # 加载 JSON 文件
                        aligned_text_entity_data = load_json(aligned_text_entity_path)
                        kv_chunk_data = load_json(kv_store_chunk_knowledge_graph_path)
                        kv_image_data = load_json(kv_store_image_knowledge_graph_path)

                        # 提取知识图谱中的实体名称集合，并统一为大写
                        chunk_entities = {clean_entity_name(entity['entity_name']).upper() for chunk in kv_chunk_data.values() for entity in chunk['entities']}
                        image_entities = {clean_entity_name(entity['entity_name']).upper() for entities in kv_image_data.values() for entity in entities}

                        # 标记文件夹是否已经输出过
                        folder_has_issues = False

                        # 遍历 aligned_text_entity.json 中的所有图片，检查 source_image_entities 和 source_text_entities
                        for image_key, image_data in aligned_text_entity_data.items():
                            for entity in image_data:
                                # 检查 source_text_entities 是否在 kv_store_chunk_knowledge_graph 中存在
                                for text_entity in entity.get('source_text_entities', []):
                                    cleaned_text_entity = clean_entity_name(text_entity).upper()
                                    if cleaned_text_entity not in chunk_entities:
                                        if not folder_has_issues:
                                            # 如果文件夹第一次出现问题，输出文件夹名称
                                            output.write(f"文件夹 {folder_name} 存在问题:\n")
                                            folder_has_issues = True
                                        output.write(f"  在图像 {image_key} 中不存在的文本实体: {text_entity}\n")

                                # 进一步检查 source_image_entities 是否包含在 kv_store_image_knowledge_graph 中的对应图像 ID 的实体列表中
                                if image_key in kv_image_data:
                                    image_knowledge_graph_entities = {clean_entity_name(entity['entity_name']).upper() for entity in kv_image_data[image_key]}
                                    for image_entity in entity.get('source_image_entities', []):
                                        cleaned_image_entity = clean_entity_name(image_entity).upper()
                                        if cleaned_image_entity not in image_knowledge_graph_entities:
                                            if not folder_has_issues:
                                                output.write(f"文件夹 {folder_name} 存在问题:\n")
                                                folder_has_issues = True
                                            output.write(f"  图像 {image_key} 中的实体 {image_entity} 没有出现在知识图谱的实体列表中\n")

                        # 新增的任务：检查关系边中的 src_id 和 tgt_id 是否在 entities 中存在
                        for chunk_key, chunk_value in kv_chunk_data.items():
                            # 遍历每个关系边
                            if 'relationships' in chunk_value:
                                edges_to_remove = []
                                for relationship in chunk_value['relationships']:
                                    # 清理 src_id 和 tgt_id
                                    src_id = clean_entity_name(relationship.get('src_id', '')).upper()
                                    tgt_id = clean_entity_name(relationship.get('tgt_id', '')).upper()

                                    # 检查 src_id 和 tgt_id 是否在 entities 中存在
                                    src_entity_exists = any(clean_entity_name(entity['entity_name']).upper() == src_id for entity in chunk_value['entities'])
                                    tgt_entity_exists = any(clean_entity_name(entity['entity_name']).upper() == tgt_id for entity in chunk_value['entities'])

                                    # 如果存在无效的边，记录并标记删除
                                    if not src_entity_exists or not tgt_entity_exists:
                                        edges_to_remove.append(relationship)

                                # 删除无效的关系边
                                chunk_value['relationships'] = [edge for edge in chunk_value['relationships'] if edge not in edges_to_remove]

                        # 将更新后的数据写回文件
                        with open(kv_store_chunk_knowledge_graph_path, 'w', encoding='utf-8') as f:
                            json.dump(kv_chunk_data, f, ensure_ascii=False, indent=4)

fusion_dataset_path = '/Users/xueyaowan/Documents/硕士毕业设计/newproject/fusion_research/fusion_dataset'
output_file = '/Users/xueyaowan/Documents/硕士毕业设计/newproject/fusion_research/fusion_dataset_generation/check.txt'
check_entities_in_knowledge_graph(fusion_dataset_path, output_file)