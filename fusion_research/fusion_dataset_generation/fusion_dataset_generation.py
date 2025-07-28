import os
import shutil
import sys
import json
import base64
import multiprocessing

import xml.etree.ElementTree as ET
from pull_llm import get_llm_response, get_mmllm_response, normalize_to_json, normalize_to_json_list
from fusion_prompt import PROMPTS

# 获取mmgraphrag的目录
project_root = "./"
# 将 mmgraphrag 文件夹添加到 sys.path
mmgraphrag_path = os.path.join(project_root, 'mmgraphrag')
sys.path.append(mmgraphrag_path)

from mmgraphrag import MMGraphRAG
from time import time

def index(pdf_path,working_dir):
    rag = MMGraphRAG(
        working_dir=working_dir,
        input_mode=2
    )
    start = time()
    rag.index(pdf_path)
    print('success!ヾ(✿ﾟ▽ﾟ)ノ')
    print("indexing time:", time() - start)
    def clean_folder(folder_path):
        # 要保留的文件名和文件夹名
        keep_files = {"kv_store_text_chunks.json", "kv_store_chunk_knowledge_graph.json", "kv_store_image_data.json"}
        keep_folders = {"images"}

        # 遍历文件夹内容
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # 如果是文件且不在保留名单，删除文件
            if os.path.isfile(item_path) and item not in keep_files:
                os.remove(item_path)
                print(f"Deleted file: {item_path}")

            # 如果是文件夹且不在保留名单，删除文件夹及其内容
            elif os.path.isdir(item_path) and item not in keep_folders:
                for root, dirs, files in os.walk(item_path, topdown=False):
                    # 先删除文件
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    # 再删除空文件夹
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        os.rmdir(dir_path)
                        print(f"Deleted folder: {dir_path}")
                # 删除目标文件夹本身
                os.rmdir(item_path)
                print(f"Deleted folder: {item_path}")
    clean_folder(working_dir)
    shutil.copy(pdf_path, working_dir)
    """
    # 去掉文本知识图谱中的关系
    def remove_relationships_from_json(file_path):
        # 读取 JSON 文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 遍历数据并移除 "relationships" 键
        for key in data:
            if "relationships" in data[key]:
                del data[key]["relationships"]

        # 将修改后的数据写回 JSON 文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"Removed all 'relationships' from {file_path}")
    remove_relationships_from_json(os.path.join(working_dir, 'kv_store_chunk_knowledge_graph.json'))
    """
    # 将图像知识图谱存储为json格式
    image_knowledge_graph_path = os.path.join(working_dir, 'kv_store_image_knowledge_graph.json')
    image_data_path = os.path.join(working_dir, 'kv_store_image_data.json')
    with open(image_data_path,'r') as f:
        image_data = json.load(f)
    def extract_image_entities(img_entity_name):
        # 构建 GraphML 文件路径
        image_knowledge_graph_path = os.path.join(working_dir, f"images/{img_entity_name}/graph_{img_entity_name}_entity_relation.graphml")
        
        # 检查文件是否存在
        if not os.path.exists(image_knowledge_graph_path):
            print(f"GraphML file not found: {image_knowledge_graph_path}")
            return

        # 解析 GraphML 文件
        tree = ET.parse(image_knowledge_graph_path)
        root = tree.getroot()
        image_entities = []
        # 定义命名空间
        namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        # 遍历所有 'node' 元素
        for node in root.findall('graphml:graph/graphml:node', namespace):
            # 提取实体信息
            entity_name = node.get('id').strip('"')
            for data in node.findall('graphml:data', namespace):
                if data.get('key') == 'd0':  # 'd0' 对应实体类型
                    entity_type = data.text.strip('"')  # 获取实体类型并去掉引号
            for data in node.findall('graphml:data', namespace):
                if data.get('key') == 'd1':  # 'd1' 对应描述
                    description = data.text.strip('"')  # 获取描述并去掉引号

            # 准备节点数据
            node_data = {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "description": description
            }
            image_entities.append(node_data)
        return image_entities
    image_knowledge_graph_info = {}
    for img_entity_name in image_data:
        image_knowledge_graph_info[img_entity_name] = extract_image_entities(img_entity_name)
    with open(image_knowledge_graph_path, 'w', encoding='utf-8') as json_file:
        json.dump(image_knowledge_graph_info, json_file, ensure_ascii=False, indent=4)
    print(f"Image entities have been saved to {image_knowledge_graph_path}")

def check(working_dir):
    image_data_path = os.path.join(working_dir, 'kv_store_image_data.json')
    with open(image_data_path,'r') as f:
        image_data = json.load(f)
    text_chunks_path = os.path.join(working_dir, 'kv_store_text_chunks.json')
    with open(text_chunks_path,'r') as f:
        text_chunks = json.load(f)
    chunk_knowledge_graph_path = os.path.join(working_dir, 'kv_store_chunk_knowledge_graph.json')
    with open(chunk_knowledge_graph_path,'r') as f:
        chunk_knowledge_graph = json.load(f)
    def get_all_neighbors(data1, data2):
        # 提取所有 chunk_order_index 列表
        chunk_indices = [item["chunk_order_index"] for item in data1.values()]
        
        # 合并所有邻居索引到一个集合
        neighbors = set()
        for i, index in enumerate(chunk_indices):
            # 获取当前索引和邻居索引
            start_index = max(0, index - 1)
            end_index = min(len(data2) - 1, index + 1)
            neighbors.update(range(start_index,end_index + 1))
        chunks_to_check = []
        for key, value in data2.items():
            if value.get("chunk_order_index") in neighbors:
                chunks_to_check.append(key)
        return chunks_to_check
    chunks_to_check = get_all_neighbors(image_data, text_chunks)
    log_dir = os.path.join(working_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    def process_single_chunk(chunk_key):
        chunk_text = text_chunks[chunk_key]["content"]
        
        for value in chunk_knowledge_graph.values():
            if value.get("chunk_key") == chunk_key:
                entities = value.get("entities", [])
                # 去掉每个实体的 source_id
                for entity in entities:
                    entity.pop("source_id", None)
        prompt_user = PROMPTS["merging_entities_user"].format(chunk_text=chunk_text,entity_list=entities)
        chunk_merged_entities = get_llm_response(prompt_user,PROMPTS["merging_entities_system"])
        return normalize_to_json_list(chunk_merged_entities)
    merged_entities = {}
    for chunk_key in chunks_to_check:
        merged_entities[chunk_key] = process_single_chunk(chunk_key)
    with open(os.path.join(log_dir, f"merged_entities.json"), 'w', encoding='utf-8') as json_file:
        json.dump(merged_entities, json_file, ensure_ascii=False, indent=4)
    print(f"Merged entities have been saved to {log_dir}, please check the log file.")

def merge(working_dir):
    merged_entities_path = os.path.join(working_dir, 'log/merged_entities.json')
    with open(merged_entities_path,'r') as f:
        merged_entities = json.load(f)
    chunk_knowledge_graph_path = os.path.join(working_dir, 'kv_store_chunk_knowledge_graph.json')
    with open(chunk_knowledge_graph_path,'r') as f:
        chunk_knowledge_graph = json.load(f)
    def process_chunk_knowledge_graph(merged_entities, chunk_knowledge_graph):
        # 1. 将所有实体名称统一为大写
        def normalize_entity_name(entity_name):
            return entity_name.strip().upper()

        # 2. 创建一个映射，将每个待融合的实体映射到融合后的实体
        fusion_map = {}  # 用于存储实体融合的映射关系
        entity_descriptions = {}  # 用于存储每个实体的描述
        
        # 将merged_entities中的实体融合信息加载到fusion_map和entity_descriptions中
        for chunk_key, entities in merged_entities.items():
            for entity in entities:
                # 获取融合后的实体名称
                fused_entity_name = normalize_entity_name(entity['entity_name'])
                # 存储该实体的描述（只取第一个融合实体的描述）
                entity_descriptions[fused_entity_name] = entity['description']
                # 将每个源实体映射到融合后的实体
                for source_entity in entity['source_entities']:
                    fusion_map[normalize_entity_name(source_entity)] = fused_entity_name

        # 3. 处理每个 chunk 中的实体和关系
        for chunk_id, chunk_data in chunk_knowledge_graph.items():
            # 处理实体部分
            new_entities = []
            seen_entities = set()  # 用于追踪已经出现的实体，避免重复
            for entity in chunk_data['entities']:
                original_entity_name = entity['entity_name'].strip("\"")
                normalized_entity_name = normalize_entity_name(original_entity_name)
                # 如果该实体在fusion_map中，则替换为融合后的实体
                if normalized_entity_name in fusion_map:
                    new_entity_name = fusion_map[normalized_entity_name]
                else:
                    new_entity_name = normalized_entity_name
                
                # 如果这个实体已经出现过，就跳过
                if new_entity_name not in seen_entities:
                    seen_entities.add(new_entity_name)
                    # 使用融合后的描述
                    new_entities.append({
                        'entity_name': f'"{new_entity_name}"',
                        'entity_type': entity['entity_type'],
                        'description': entity_descriptions.get(new_entity_name, entity['description']),
                        'source_id': entity['source_id']
                    })
            
            chunk_data['entities'] = new_entities

            # 处理关系部分
            new_relationships = []
            seen_relationships = set()  # 用于去重源和目标实体之间的关系
            for relationship in chunk_data['relationships']:
                src_id = relationship['src_id'].strip("\"")
                tgt_id = relationship['tgt_id'].strip("\"")
                
                # 关系中的src_id和tgt_id需要根据fusion_map更新
                if src_id in fusion_map:
                    src_id = fusion_map[src_id]
                if tgt_id in fusion_map:
                    tgt_id = fusion_map[tgt_id]

                # 构建关系的唯一标识，考虑源和目标实体的顺序
                relationship_id = tuple(sorted([src_id, tgt_id]))  # 排序后组合成元组，确保方向不影响去重

                # 检查是否已经存在相同的关系
                if relationship_id not in seen_relationships and src_id != tgt_id:
                    seen_relationships.add(relationship_id)
                    new_relationships.append({
                        'src_id': f'"{src_id}"',
                        'tgt_id': f'"{tgt_id}"',
                        'weight': relationship['weight'],
                        'description': relationship['description'],
                        'source_id': relationship['source_id']
                    })
            
            chunk_data['relationships'] = new_relationships
        
        return chunk_knowledge_graph

    # 处理 chunk_knowledge_graph
    updated_chunk_knowledge_graph = process_chunk_knowledge_graph(merged_entities, chunk_knowledge_graph)
    
    # 将更新后的 chunk_knowledge_graph 保存回原文件
    with open(chunk_knowledge_graph_path, 'w') as f:
        json.dump(updated_chunk_knowledge_graph, f, indent=4)

    print(f"Updated chunk knowledge graph has been saved to {chunk_knowledge_graph_path}")

def generation(working_dir):
    image_data_path = os.path.join(working_dir, 'kv_store_image_data.json')
    log_dir = os.path.join(working_dir, 'log')
    with open(image_data_path,'r') as f:
        image_data = json.load(f)
    text_chunks_path = os.path.join(working_dir, 'kv_store_text_chunks.json')
    with open(text_chunks_path,'r') as f:
        text_chunks = json.load(f)
    chunk_knowledge_graph_path = os.path.join(working_dir, 'kv_store_chunk_knowledge_graph.json')
    with open(chunk_knowledge_graph_path,'r') as f:
        chunk_knowledge_graph = json.load(f)
    image_knowledge_graph_path = os.path.join(working_dir, 'kv_store_image_knowledge_graph.json')
    with open(image_knowledge_graph_path,'r') as f:
        image_knowledge_graph = json.load(f)
    def get_nearby_entities(data, index):
        # 获取前后两个数字的范围
        start_index = max(0, index - 1)  # 如果是0，则只取0和1
        end_index = min(len(data) - 1, index + 1)  # 如果是最后一个数字，则只取自己和前一个
        
        # 提取指定范围的entities
        entities = []
        for i in range(start_index, end_index + 1):
            entities.extend(data.get(str(i), {}).get("entities", []))
        # 去掉每个实体的 source_id
        for entity in entities:
            entity.pop("source_id", None)
        return entities
    def align_single_image_entity(img_entity_name):
        image_path = image_data[img_entity_name]["image_path"]
        img_entity_description = image_data[img_entity_name]["description"]
        chunk_key = image_data[img_entity_name]["chunk_id"]
        chunk_text = text_chunks[chunk_key]["content"]
        chunk_order_index = image_data[img_entity_name]["chunk_order_index"]
        nearby_entities = get_nearby_entities(chunk_knowledge_graph, chunk_order_index)
        with open(image_path, "rb") as image_file:
            img_base = base64.b64encode(image_file.read()).decode("utf-8")
        entity_type = PROMPTS["DEFAULT_ENTITY_TYPES"]
        entity_type = [item.upper() for item in entity_type]
        alignment_prompt_user = PROMPTS["image_entity_alignment_user"].format(entity_type = entity_type, img_entity=img_entity_name, img_entity_description=img_entity_description, chunk_text=chunk_text, nearby_entities=nearby_entities)
        aligned_image_entity = get_mmllm_response(alignment_prompt_user, PROMPTS["image_entity_alignment_system"], img_base)
        return normalize_to_json(aligned_image_entity)
    aligned_image_entities = {}
    for img_entity_name in image_data:
        aligned_image_entities[img_entity_name] = align_single_image_entity(img_entity_name)
    with open(os.path.join(log_dir, 'aligned_image_entity.json'), 'w', encoding='utf-8') as json_file:
        json.dump(aligned_image_entities, json_file, ensure_ascii=False, indent=4)
    def align_single_text_image_entity(img_entity_name):
        image_entities = image_knowledge_graph[img_entity_name]
        text_image_entities = [entity for entity in image_entities if entity["entity_type"] not in ["ORI_IMG", "IMG"]]
        chunk_key = image_data[img_entity_name]["chunk_id"]
        chunk_text = text_chunks[chunk_key]["content"]
        chunk_order_index = image_data[img_entity_name]["chunk_order_index"]
        nearby_entities = get_nearby_entities(chunk_knowledge_graph, chunk_order_index)
        # 第一次执行对齐
        alignment_prompt_user = PROMPTS["text_entity_alignment_user"].format(image_entities=text_image_entities, chunk_text=chunk_text, nearby_text_entities=nearby_entities)
        aligned_text_entity = get_llm_response(alignment_prompt_user, PROMPTS["text_entity_alignment_system"])
        normalized_aligned_text_entity = normalize_to_json_list(aligned_text_entity)
        
        # 如果第一次结果为空，重新执行一次
        if not normalized_aligned_text_entity:
            print("First alignment result is empty, re-aligning...")
            alignment_prompt_user = PROMPTS["text_entity_alignment_user2"].format(image_entities=text_image_entities, nearby_text_entities=nearby_entities)
            aligned_text_entity = get_llm_response(alignment_prompt_user, PROMPTS["text_entity_alignment_system2"])
            normalized_aligned_text_entity = normalize_to_json_list(aligned_text_entity)
            return aligned_text_entity
        return normalized_aligned_text_entity
    aligned_text_entities = {}
    for img_entity_name in image_data:
        aligned_text_entities[img_entity_name] = align_single_text_image_entity(img_entity_name)
    with open(os.path.join(log_dir, 'aligned_text_entity.json'), 'w', encoding='utf-8') as json_file:
        json.dump(aligned_text_entities, json_file, ensure_ascii=False, indent=4)

def process_single_pdf(pdf_path, dataset_dir):
    # 获取pdf文件名称
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    working_dir = os.path.join(dataset_dir, pdf_filename)
    process = multiprocessing.Process(target=index,args=(pdf_path,working_dir))
    process.start()
    process.join()
    check(working_dir)

def main(mode, dataset_dir, folder_path):
    if mode == "index":
        # 确保 folder_path 参数存在
        if not folder_path:
            raise ValueError("folder_path must be provided when mode is 'index'")
        
        # 处理 PDF 文件
        for pdf_file in os.listdir(folder_path):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(folder_path, pdf_file)
                print(f"Processing {pdf_file}...")
                process_single_pdf(pdf_path,dataset_dir)
    elif mode == "generation":
        # 确保 dataset_dir 参数存在
        if not dataset_dir:
            raise ValueError("dataset_dir must be provided when mode is 'generation'")
        
        # 处理数据集文件夹
        for subfolder in os.listdir(dataset_dir):
            working_dir = os.path.join(dataset_dir, subfolder)
            if os.path.isdir(working_dir):
                # 检查是否存在 aligned_image_entity.json 文件
                if "aligned_text_entity.json" in os.listdir(working_dir):
                    print(f"Skipping folder: {subfolder} (aligned_text_entity.json found)")
                    continue
                # 执行 generation 函数
                print(f"Processing {subfolder}...")
                merge(working_dir)
                generation(working_dir)
    else:
        raise ValueError("Invalid mode. Please use 'index' or 'generation'.")
    
if __name__ == "__main__":
    # index/generation
    mode = "generation"
    folder_path = './fusion_research/fusion_dataset/fusion_dataset_pdf/paper'
    dataset_dir = './fusion_research/fusion_dataset/paper'

    main(mode, dataset_dir=dataset_dir, folder_path=folder_path)
