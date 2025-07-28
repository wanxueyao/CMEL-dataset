import base64

from pull_llm import get_mmllm_response, get_llm_response, normalize_to_json, normalize_to_json_list
from prompt import PROMPTS
from clustering_function import get_possible_entities_image_clustering, get_possible_entities_text_clustering, judge_text_entity_alignment_clustering

from sklearn.metrics.pairwise import cosine_similarity

def get_nearby_chunks(data, index):
    # 获取前后两个数字的范围
    start_index = max(0, index - 1)  # 如果是0，则只取0和1
    end_index = min(len(data) - 1, index + 1)  # 如果是最后一个数字，则只取自己和前一个
    
    all_index = list(range(start_index, end_index + 1))
    nearby_chunks = []
    for key, value in data.items():
            if value.get("chunk_order_index") in all_index:
                nearby_chunks.append(value.get("content"))
    return nearby_chunks

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

def get_nearby_relationships(data, index):
        # 获取前后两个数字的范围
        start_index = max(0, index - 1)  # 如果是0，则只取0和1
        end_index = min(len(data) - 1, index + 1)  # 如果是最后一个数字，则只取自己和前一个
        
        # 提取指定范围的relationships
        relationships = []
        for i in range(start_index, end_index + 1):
            relationships.extend(data.get(str(i), {}).get("relationships", []))
        # 去掉每个关系的 source_id
        for relationship in relationships:
            relationship.pop("source_id", None)
        return relationships

def align_single_image_entity(img_entity_name, image_data, text_chunks):
    image_path = image_data[img_entity_name]["image_path"]
    img_entity_description = image_data[img_entity_name]["description"]
    chunk_order_index = image_data[img_entity_name]["chunk_order_index"]
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    entity_type = PROMPTS["DEFAULT_ENTITY_TYPES"]
    entity_type = [item.upper() for item in entity_type]
    with open(image_path, "rb") as image_file:
        img_base = base64.b64encode(image_file.read()).decode("utf-8")
    alignment_prompt_user = PROMPTS["image_entity_alignment_user"].format(entity_type = entity_type, img_entity=img_entity_name, img_entity_description=img_entity_description, chunk_text=nearby_chunks)
    aligned_image_entity = get_mmllm_response(alignment_prompt_user, PROMPTS["image_entity_alignment_system"], img_base)
    return normalize_to_json(aligned_image_entity)

def judge_image_entity_alignment(image_entity_name, image_entity_description, possible_image_matched_entities, nearby_chunks):
    image_entity_judgement_user = PROMPTS["image_entity_judgement_user"].format(img_entity=image_entity_name, img_entity_description=image_entity_description, possible_matched_entities=possible_image_matched_entities, chunk_text=nearby_chunks)
    matched_entity_name = get_llm_response(image_entity_judgement_user, PROMPTS["image_entity_judgement_system"])
    return matched_entity_name

def judge_text_entity_alignment(possible_text_matched_entities):
    image_entity_judgement_user = PROMPTS["text_entity_judgement_user"].format(possible_matched_entities=possible_text_matched_entities)
    aligned_text_entity_list = get_llm_response(image_entity_judgement_user, PROMPTS["text_entity_judgement_system"])
    normalized_aligned_text_entity_list = normalize_to_json_list(aligned_text_entity_list)
    if not normalized_aligned_text_entity_list:
        print(aligned_text_entity_list)
    return normalized_aligned_text_entity_list

def entity_alignment_embedding(image_id, image_data, text_chunks, chunk_knowledge_graph, image_knowledge_graph, model, threshold):
    # 获取特定image_id下的chunk_order_index
    chunk_order_index = image_data[image_id].get("chunk_order_index")
    image_entity = align_single_image_entity(image_id, image_data, text_chunks)
    if image_entity is not None:
        image_entity_name = image_entity.get("entity_name")
        image_entity_description = image_entity.get("description")
    else:
        image_entity_name = "no match"
        image_entity_description = "None."
    image_entity_list = image_knowledge_graph.get(image_id,[])
    exclude_types=["ORI_IMG", "IMG"]
    filtered_image_entity_list = [entity for entity in image_entity_list if entity['entity_type'] not in exclude_types]
    nearby_text_entity_list = get_nearby_entities(chunk_knowledge_graph, chunk_order_index)
    def compute_similarity_image(image_entity_description, nearby_text_entity_list, threshold):
        """
        计算图像实体描述和文本实体描述之间的语义相似度，
        并筛选出相似度高于阈值的实体。
        
        参数:
        - image_entity_description: 图像实体的描述
        - nearby_text_entity_list: 文本实体列表
        - threshold: 相似度阈值
        
        返回:
        - 保留的实体列表
        """
        # 计算图像实体的嵌入向量
        image_embedding = model.encode([image_entity_description])[0]
        
        # 存储每个实体及其相似度
        similar_entities = []
        
        for entity in nearby_text_entity_list:
            entity_description = entity.get('description', '')
            
            # 计算文本实体的嵌入向量
            entity_embedding = model.encode([entity_description])[0]
            
            # 计算图像实体和文本实体之间的余弦相似度
            similarity = cosine_similarity([image_embedding], [entity_embedding])[0][0]
            
            # 如果相似度大于阈值，则保留该实体
            if similarity >= threshold:
                similar_entities.append({
                    'entity_name': entity['entity_name'],
                    'entity_type': entity['entity_type'],
                    'description': entity['description'],
                    'similarity': similarity
                })
        
        # 按照相似度排序，并选择前3个
        similar_entities = sorted(similar_entities, key=lambda x: x['similarity'], reverse=True)[:3]
        
        return similar_entities
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    possible_image_matched_entities = compute_similarity_image(image_entity_description, nearby_text_entity_list, threshold) 
    matched_entity_name = judge_image_entity_alignment(image_entity_name, image_entity_description, possible_image_matched_entities, nearby_chunks)
    def compute_similarity_text(filtered_image_entity_list, nearby_text_entity_list, threshold):
        # 提取所有实体的描述文本
        image_descriptions = [entity['description'] for entity in filtered_image_entity_list]
        chunk_descriptions = [entity['description'] for entity in nearby_text_entity_list]
        # 使用 Sentence-BERT 模型将文本描述转换为嵌入
        image_embeddings = model.encode(image_descriptions)
        chunk_embeddings = model.encode(chunk_descriptions)
        # 计算所有图像文本实体与文本实体之间的余弦相似度
        similarity_matrix = cosine_similarity(image_embeddings, chunk_embeddings)

        # 结果列表
        possible_text_matched_entities = []

        # 对于每个图像文本实体，找到最相似的文本实体并满足相似度阈值
        for i, image_entity in enumerate(filtered_image_entity_list):
            max_similarity = -1
            most_similar_entity = None
            
            for j, chunk_entity in enumerate(nearby_text_entity_list):
                similarity = similarity_matrix[i][j]
                
                if similarity > threshold and similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_entity = chunk_entity
            
            if most_similar_entity:
                # 将最相似的图像文本实体和文本实体的所有信息一起保存
                possible_text_matched_entities.append({
                    'image_entity': image_entity,
                    'chunk_entity': most_similar_entity,
                    'similarity': max_similarity
                })
        return possible_text_matched_entities
    possible_text_matched_entities = compute_similarity_text(filtered_image_entity_list, nearby_text_entity_list, threshold)
    aligned_text_entity_list = judge_text_entity_alignment(possible_text_matched_entities)   
    return matched_entity_name, aligned_text_entity_list

def entity_alignment_llm(image_id, image_data, text_chunks, chunk_knowledge_graph, image_knowledge_graph):
    chunk_order_index = image_data[image_id].get("chunk_order_index")
    image_entity = align_single_image_entity(image_id, image_data, text_chunks)
    if image_entity is not None:
        image_entity_name = image_entity.get("entity_name")
        image_entity_description = image_entity.get("description")
    else:
        image_entity_name = "no match"
        image_entity_description = "None."
    image_entity_list = image_knowledge_graph.get(image_id,[])
    exclude_types=["ORI_IMG", "IMG"]
    filtered_image_entity_list = [entity for entity in image_entity_list if entity['entity_type'] not in exclude_types]
    nearby_text_entity_list = get_nearby_entities(chunk_knowledge_graph, chunk_order_index)
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    def get_possible_entities_image(image_entity_description, nearby_text_entity_list):
        llm_alignment_user = PROMPTS["llm_image_entity_alignment_user"].format(img_entity_description=image_entity_description, entity_list=nearby_text_entity_list)
        possible_image_matched_entities = get_llm_response(llm_alignment_user, PROMPTS["llm_image_entity_alignment_system"])
        return normalize_to_json_list(possible_image_matched_entities)
    possible_image_matched_entities = get_possible_entities_image(image_entity_description, nearby_text_entity_list) 
    matched_entity_name = judge_image_entity_alignment(image_entity_name, image_entity_description, possible_image_matched_entities, nearby_chunks)
    def get_possible_entities_text(filtered_image_entity_list, nearby_text_entity_list):
        llm_alignment_user = PROMPTS["llm_text_entity_alignment_user"].format(image_entity_list=filtered_image_entity_list,text_entity_list=nearby_text_entity_list)
        possible_text_matched_entities = get_llm_response(llm_alignment_user, PROMPTS["llm_text_entity_alignment_system"])
        return normalize_to_json_list(possible_text_matched_entities)
    possible_text_matched_entities = get_possible_entities_text(filtered_image_entity_list, nearby_text_entity_list)
    aligned_text_entity_list = judge_text_entity_alignment(possible_text_matched_entities)
    return matched_entity_name, aligned_text_entity_list

def entity_alignment_clustering(model, clustering_method, classify_method, image_id, image_data, text_chunks, chunk_knowledge_graph, image_knowledge_graph):
    chunk_order_index = image_data[image_id].get("chunk_order_index")
    image_entity = align_single_image_entity(image_id, image_data, text_chunks)
    if image_entity is not None:
        image_entity_name = image_entity.get("entity_name")
        image_entity_description = image_entity.get("description")
    else:
        image_entity_name = "no match"
        image_entity_description = "None."
    image_entity_list = image_knowledge_graph.get(image_id,[])
    exclude_types=["ORI_IMG", "IMG"]
    filtered_image_entity_list = [entity for entity in image_entity_list if entity['entity_type'] not in exclude_types]
    nearby_text_entity_list = get_nearby_entities(chunk_knowledge_graph, chunk_order_index)
    nearby_relationship_list = get_nearby_relationships(chunk_knowledge_graph, chunk_order_index)
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    possible_image_matched_entities = get_possible_entities_image_clustering(model, clustering_method, classify_method, image_entity_description, nearby_text_entity_list, nearby_relationship_list) 
    matched_entity_name = judge_image_entity_alignment(image_entity_name, image_entity_description, possible_image_matched_entities, nearby_chunks)
    image_entity_with_labels, text_clustering_results = get_possible_entities_text_clustering(model, clustering_method, classify_method, filtered_image_entity_list, nearby_text_entity_list, nearby_relationship_list)
    aligned_text_entity_list = judge_text_entity_alignment_clustering(image_entity_with_labels, text_clustering_results)
    return matched_entity_name, aligned_text_entity_list