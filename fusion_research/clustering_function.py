import math
import networkx as nx
import numpy as np
import leidenalg
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from igraph import Graph
from pull_llm import get_llm_response, normalize_to_json_list

def get_possible_entities_image_clustering(
    model, clustering_method, classify_method, image_entity_description, nearby_text_entity_list, nearby_relationship_list
):
    """
    聚类和分类函数，支持 KMeans/DBSCAN/Pagerank/Leiden 聚类及 KNN/LLM 分类。

    Parameters:
        clustering_method (str): 聚类方法 ("KMeans", "DBSCAN", "Pagerank", 或 "Leiden")。
        classify_method (str): 分类方法 ("knn" 或 "llm")。
        image_entity_description (str): 图像实体描述。
        nearby_text_entity_list (list): 附近文本实体列表，每个实体包含 "entity_name"、"entity_type" 和 "description"。
        nearby_relationship_list (list): 实体之间的关系列表，每个关系包含 "src_id"、"tgt_id"、"weight" 和 "description"。

    Returns:
        result_entities (list): 属于同一类别的实体列表。
    """
    # Step 0: 排序关系列表，根据权重降序
    nearby_relationship_list = sorted(nearby_relationship_list, key=lambda x: x['weight'], reverse=True)
    
    # Step 1: 获取所有实体描述的嵌入
    descriptions = [entity["description"] for entity in nearby_text_entity_list]
    entity_names = [entity["entity_name"] for entity in nearby_text_entity_list]
    embeddings = model.encode(descriptions)

    # Step 2: 聚类方法选择
    labels = None
    cluster_centers = None
    if clustering_method == "kmeans":
        # 使用实体数量的平方根作为聚类数量，并向上取整，最小为2
        num_clusters = max(2, math.ceil(math.sqrt(len(nearby_text_entity_list))))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
    elif clustering_method == "dbscan":
        # DBSCAN 参数设置
        min_samples = max(1, math.ceil(len(nearby_text_entity_list) / 10))
        dbscan = DBSCAN(eps=0.5, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(embeddings)
    elif clustering_method == "pagerank":
        # 构建关系图
        G = nx.DiGraph()  # 创建有向图
        for rel in nearby_relationship_list:
            G.add_edge(rel["src_id"], rel["tgt_id"], weight=rel["weight"])

        # 计算 Pagerank
        pagerank_scores = nx.pagerank(G, weight="weight")

        # 根据 nearby_relationship_list 的大小动态确定类别数量
        num_categories = max(2, len(nearby_relationship_list) // 10)  # 可以通过调整比例来控制类别数量
        # 如果需要更多的类别，可以增加比例因子
        pagerank_sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

        # 为每个节点分配类别标签
        labels_dict = {}
        nodes_per_category = len(pagerank_sorted_nodes) // num_categories

        for idx, (node, score) in enumerate(pagerank_sorted_nodes):
            # 计算节点应该属于哪个类别
            category = idx // nodes_per_category
            labels_dict[node] = category

        # 创建一个字典，初始化所有实体标签为 -1，表示未分类
        entity_labels = {entity["entity_name"]: -1 for entity in nearby_text_entity_list}  # 默认所有实体标签为-1

        # 将图中的节点和实体列表映射
        # 确保我们使用 entity_name 来保证实体与标签的正确映射
        entity_name_to_idx = {entity["entity_name"]: idx for idx, entity in enumerate(nearby_text_entity_list)}

        # 为每个图中的节点分配标签
        for node in G.nodes():
            if node in entity_name_to_idx:
                entity_name = node
                entity_labels[entity_name] = labels_dict[node]

        # 处理未在图中提到的实体（没有任何边连接的实体）
        next_available_label = max([label for label in entity_labels.values() if label != -1], default=1) + 1  # 找到一个新的标签编号
        for entity in nearby_text_entity_list:
            entity_name = entity["entity_name"]
            if entity_labels[entity_name] == -1:  # 如果没有标签
                entity_labels[entity_name] = next_available_label
                next_available_label += 1

        # 按照 nearby_text_entity_list 的顺序输出 labels
        labels = [entity_labels[entity["entity_name"]] for entity in nearby_text_entity_list]
    elif clustering_method == "leiden":
        # 使用 Leiden 算法
        # 构建图
        G = nx.Graph()
        for rel in nearby_relationship_list:
            G.add_edge(rel["src_id"], rel["tgt_id"], weight=rel["weight"])

        # 转换为 igraph 格式
        ig = Graph.from_networkx(G)

        # 运行 Leiden 聚类算法
        partition = leidenalg.find_partition(ig, leidenalg.ModularityVertexPartition)

        # 获取每个节点的标签
        labels = np.array(partition.membership)

        # 创建一个字典，将所有实体初始化为独立的类别
        entity_labels = {entity["entity_name"]: -1 for entity in nearby_text_entity_list}  # 默认所有实体标签为-1

        # 将图中的节点和实体列表映射
        # 确保我们使用 entity_name 来保证实体与标签的正确映射
        entity_name_to_idx = {entity["entity_name"]: idx for idx, entity in enumerate(nearby_text_entity_list)}

        # 为每个图中的节点分配标签
        for idx, node in enumerate(G.nodes()):
            # 根据节点名称映射到实体列表中的索引
            if node in entity_name_to_idx:
                entity_name = node
                entity_labels[entity_name] = labels[idx]

        # 处理未在图中提到的实体（没有任何边连接的实体）
        next_available_label = max(labels) + 1  # 找到一个新的标签编号
        for entity in nearby_text_entity_list:
            entity_name = entity["entity_name"]
            if entity_labels[entity_name] == -1:  # 如果没有标签
                entity_labels[entity_name] = next_available_label
                next_available_label += 1

        # 按照 nearby_text_entity_list 的顺序输出 labels
        labels = [entity_labels[entity["entity_name"]] for entity in nearby_text_entity_list]
    elif clustering_method == "spectral":
        # 计算相似度矩阵（余弦相似度）
        similarity_matrix = cosine_similarity(embeddings)

        # 根据关系权重修改度矩阵
        for relation in nearby_relationship_list:
            # 只有当 src_id 和 tgt_id 都在 entity_names 中时才执行
            if relation["src_id"] in entity_names and relation["tgt_id"] in entity_names:
                src_idx = entity_names.index(relation["src_id"])
                tgt_idx = entity_names.index(relation["tgt_id"])
            else:
                continue

            weight = relation["weight"]
            similarity_matrix[src_idx, tgt_idx] *= weight
            similarity_matrix[tgt_idx, src_idx] *= weight  # 确保邻接矩阵是对称的
        
        # 计算度矩阵
        degree_matrix = np.zeros_like(similarity_matrix)
        for i in range(len(similarity_matrix)):
            degree_matrix[i, i] = np.sum(similarity_matrix[i, :])

        # 计算拉普拉斯矩阵 L = D - A
        laplacian_matrix = degree_matrix - similarity_matrix

        # 计算拉普拉斯矩阵的特征值和特征向量
        eigvals, eigvecs = np.linalg.eig(laplacian_matrix)

        # 选择前k个最小的特征值对应的特征向量
        k = max(2, math.ceil(math.sqrt(len(nearby_text_entity_list))))
        eigvecs_selected = eigvecs[:, np.argsort(eigvals)[:k]]

        # 使用 DBSCAN 聚类
        min_samples = max(1, math.ceil(len(nearby_text_entity_list) / 10))
        dbscan = DBSCAN(eps=0.5, min_samples=min_samples)  # 调整 eps 和 min_samples 参数
        dbscan_labels = dbscan.fit_predict(eigvecs_selected)

        # 输出每个节点的聚类标签
        labels = dbscan_labels

        # 按照 nearby_text_entity_list 的顺序输出 labels
        labels = [labels[entity_names.index(entity["entity_name"])] for entity in nearby_text_entity_list]

    # Step 3: 判断输入描述的类别
    input_embedding = model.encode([image_entity_description])
    if classify_method == "knn":
        if clustering_method == "kmeans":
            # 使用 KMeans 中心点找到最近的簇
            nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(cluster_centers)
            _, cluster_index = nn.kneighbors(input_embedding)
            target_label = cluster_index[0][0]
        elif clustering_method in ["pagerank", "leiden", "spectral"]:
            # 找到最近邻并使用 Pagerank,Leiden或Spectral的标签
            nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(embeddings)
            _, nearest_idx = nn.kneighbors(input_embedding)
            target_label = labels[nearest_idx[0][0]]
    elif classify_method == "llm":
        # 动态生成 LLM 的 Prompt，适配任意类别数量
        clusters = {}
        for label in set(labels):
            if label == -1:  # DBSCAN 中的噪声点
                continue
            clusters[label] = [
                {
                    "entity_name": nearby_text_entity_list[idx]["entity_name"],
                    "entity_type": nearby_text_entity_list[idx]["entity_type"],
                    "description": nearby_text_entity_list[idx]["description"],
                }
                for idx in range(len(labels))
                if labels[idx] == label
            ]

        prompt_user = f"""
Each cluster contains entities, where each entity has a name, type, and description. The clusters are identified with unique numeric labels.

Here is the clustering information:
{{
    "clusters": [
        {", ".join([f'{{"label": {label}, "entities": {cluster}}}' for label, cluster in clusters.items()])}
    ]
}}

Input description:
"{image_entity_description}"

Question: Based on the clustering, which numeric label does the input description belong to? Respond only with a single numeric label (e.g., "0", "1", or "2") and nothing else. Do not include any explanations or additional text.
"""
        prompt_system = """You are an advanced AI assistant trained to categorize descriptions. Your task is to determine the category of an input description based on the following entity clusters."""
        # 调用 LLM 进行分类
        try:
            target_label = int(get_llm_response(cur_prompt=prompt_user, system_content=prompt_system))
        except (ValueError, TypeError):  # 捕获转换失败或返回值类型错误
            target_label = 1  # 默认值为1

    # Step 4: 输出属于该类别的所有实体信息
    result_entities = [
        entity
        for entity, label in zip(nearby_text_entity_list, labels)
        if label == target_label
    ]

    return result_entities

def get_possible_entities_text_clustering(
    model, clustering_method, classify_method, filtered_image_entity_list, nearby_text_entity_list, nearby_relationship_list
):
    """
    聚类和分类函数，支持 KMeans/DBSCAN/Pagerank/Leiden 聚类及 KNN/LLM 分类。

    Parameters:
        clustering_method (str): 聚类方法 ("KMeans", "DBSCAN", "Pagerank", 或 "Leiden")。
        classify_method (str): 分类方法 ("knn" 或 "llm")。
        filtered_image_entity_list (list): 过滤后的图像实体列表，每个实体包含 "entity_name" 和 "description"。
        nearby_text_entity_list (list): 附近文本实体列表，每个实体包含 "entity_name"、"entity_type" 和 "description"。
        nearby_relationship_list (list): 实体之间的关系列表，每个关系包含 "src_id"、"tgt_id"、"weight" 和 "description"。

    Returns:
        image_entity_with_labels (list): 图像实体及其对应类别的列表，每项为 {"entity_name": ..., "label": ..., "description": ..., "entity_type": ...}。
        text_clustering_results (list): 聚类后的文本实体列表，每项为 {"label": ..., "entities": [...]}。
    """
    # Step 0: 排序关系列表，根据权重降序
    nearby_relationship_list = sorted(nearby_relationship_list, key=lambda x: x['weight'], reverse=True)

    # Step 1: 获取文本实体描述的嵌入
    descriptions = [entity["description"] for entity in nearby_text_entity_list]
    entity_names = [entity["entity_name"] for entity in nearby_text_entity_list]
    embeddings = model.encode(descriptions)

    # Step 2: 聚类方法选择
    labels = None
    cluster_centers = None
    if clustering_method == "kmeans":
        num_clusters = max(2, math.ceil(math.sqrt(len(nearby_text_entity_list))))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
    elif clustering_method == "dbscan":
        min_samples = max(1, math.ceil(len(nearby_text_entity_list) / 10))
        dbscan = DBSCAN(eps=0.5, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(embeddings)
    elif clustering_method == "pagerank":
        # 构建关系图
        G = nx.DiGraph()  # 创建有向图
        for rel in nearby_relationship_list:
            G.add_edge(rel["src_id"], rel["tgt_id"], weight=rel["weight"])

        # 计算 Pagerank
        pagerank_scores = nx.pagerank(G, weight="weight")

        # 根据 nearby_relationship_list 的大小动态确定类别数量
        num_categories = max(2, len(nearby_relationship_list) // 10)  # 可以通过调整比例来控制类别数量
        # 如果需要更多的类别，可以增加比例因子
        pagerank_sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

        # 为每个节点分配类别标签
        labels_dict = {}
        nodes_per_category = len(pagerank_sorted_nodes) // num_categories

        for idx, (node, score) in enumerate(pagerank_sorted_nodes):
            # 计算节点应该属于哪个类别
            category = idx // nodes_per_category
            labels_dict[node] = category

        # 创建一个字典，初始化所有实体标签为 -1，表示未分类
        entity_labels = {entity["entity_name"]: -1 for entity in nearby_text_entity_list}  # 默认所有实体标签为-1

        # 将图中的节点和实体列表映射
        # 确保我们使用 entity_name 来保证实体与标签的正确映射
        entity_name_to_idx = {entity["entity_name"]: idx for idx, entity in enumerate(nearby_text_entity_list)}

        # 为每个图中的节点分配标签
        for node in G.nodes():
            if node in entity_name_to_idx:
                entity_name = node
                entity_labels[entity_name] = labels_dict[node]

        # 处理未在图中提到的实体（没有任何边连接的实体）
        next_available_label = max([label for label in entity_labels.values() if label != -1], default=1) + 1  # 找到一个新的标签编号
        for entity in nearby_text_entity_list:
            entity_name = entity["entity_name"]
            if entity_labels[entity_name] == -1:  # 如果没有标签
                entity_labels[entity_name] = next_available_label
                next_available_label += 1

        # 按照 nearby_text_entity_list 的顺序输出 labels
        labels = [entity_labels[entity["entity_name"]] for entity in nearby_text_entity_list]
    elif clustering_method == "leiden":
        # 使用 Leiden 算法
        # 构建图
        G = nx.Graph()
        for rel in nearby_relationship_list:
            G.add_edge(rel["src_id"], rel["tgt_id"], weight=rel["weight"])

        # 转换为 igraph 格式
        ig = Graph.from_networkx(G)

        # 运行 Leiden 聚类算法
        partition = leidenalg.find_partition(ig, leidenalg.ModularityVertexPartition)

        # 获取每个节点的标签
        labels = np.array(partition.membership)

        # 创建一个字典，将所有实体初始化为独立的类别
        entity_labels = {entity["entity_name"]: -1 for entity in nearby_text_entity_list}  # 默认所有实体标签为-1

        # 将图中的节点和实体列表映射
        # 确保我们使用 entity_name 来保证实体与标签的正确映射
        entity_name_to_idx = {entity["entity_name"]: idx for idx, entity in enumerate(nearby_text_entity_list)}

        # 为每个图中的节点分配标签
        for idx, node in enumerate(G.nodes()):
            # 根据节点名称映射到实体列表中的索引
            if node in entity_name_to_idx:
                entity_name = node
                entity_labels[entity_name] = labels[idx]

        # 处理未在图中提到的实体（没有任何边连接的实体）
        next_available_label = max(labels) + 1  # 找到一个新的标签编号
        for entity in nearby_text_entity_list:
            entity_name = entity["entity_name"]
            if entity_labels[entity_name] == -1:  # 如果没有标签
                entity_labels[entity_name] = next_available_label
                next_available_label += 1

        # 按照 nearby_text_entity_list 的顺序输出 labels
        labels = [entity_labels[entity["entity_name"]] for entity in nearby_text_entity_list]
    elif clustering_method == "spectral":
        # 计算相似度矩阵（余弦相似度）
        similarity_matrix = cosine_similarity(embeddings)

        # 根据关系权重修改度矩阵
        for relation in nearby_relationship_list:
            # 只有当 src_id 和 tgt_id 都在 entity_names 中时才执行
            if relation["src_id"] in entity_names and relation["tgt_id"] in entity_names:
                src_idx = entity_names.index(relation["src_id"])
                tgt_idx = entity_names.index(relation["tgt_id"])
            else:
                continue

            weight = relation["weight"]
            similarity_matrix[src_idx, tgt_idx] *= weight
            similarity_matrix[tgt_idx, src_idx] *= weight  # 确保邻接矩阵是对称的
        
        # 计算度矩阵
        degree_matrix = np.zeros_like(similarity_matrix)
        for i in range(len(similarity_matrix)):
            degree_matrix[i, i] = np.sum(similarity_matrix[i, :])

        # 计算拉普拉斯矩阵 L = D - A
        laplacian_matrix = degree_matrix - similarity_matrix

        # 计算拉普拉斯矩阵的特征值和特征向量
        eigvals, eigvecs = np.linalg.eig(laplacian_matrix)

        # 选择前k个最小的特征值对应的特征向量
        k = max(2, math.ceil(math.sqrt(len(nearby_text_entity_list))))
        eigvecs_selected = eigvecs[:, np.argsort(eigvals)[:k]]

        # 使用 DBSCAN 聚类
        min_samples = max(1, math.ceil(len(nearby_text_entity_list) / 10))
        dbscan = DBSCAN(eps=0.5, min_samples=min_samples)  # 调整 eps 和 min_samples 参数
        dbscan_labels = dbscan.fit_predict(eigvecs_selected)

        # 输出每个节点的聚类标签
        labels = dbscan_labels

        # 创建一个字典，初始化所有实体标签为 -1，表示未分类
        entity_labels = {entity["entity_name"]: -1 for entity in nearby_text_entity_list}

        # 按照聚类结果分配标签
        for idx, entity in enumerate(nearby_text_entity_list):
            entity_labels[entity["entity_name"]] = labels[idx]

        # 按照 nearby_text_entity_list 的顺序输出 labels
        labels = [entity_labels[entity["entity_name"]] for entity in nearby_text_entity_list]

    # Step 3: 分类图像实体到聚类类别
    image_entity_with_labels = []
    input_embeddings = model.encode([entity["description"] for entity in filtered_image_entity_list])

    if classify_method == "knn":
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings)
        for image_entity, input_embedding in zip(filtered_image_entity_list, input_embeddings):
            _, nearest_idx = nn.kneighbors([input_embedding])
            target_label = labels[nearest_idx[0][0]]
            image_entity_with_labels.append({
                "entity_name": image_entity["entity_name"],
                "label": target_label,
                "description": image_entity["description"],
                "entity_type": image_entity.get("entity_type", "image")
            })
    elif classify_method == "llm":
        clusters = {}
        for label in set(labels):
            if label == -1:  # DBSCAN 的噪声点
                continue
            clusters[label] = [
                {
                    "entity_name": nearby_text_entity_list[idx]["entity_name"],
                    "entity_type": nearby_text_entity_list[idx]["entity_type"],
                    "description": nearby_text_entity_list[idx]["description"],
                }
                for idx in range(len(labels))
                if labels[idx] == label
            ]

        for image_entity in filtered_image_entity_list:
            prompt_user = f"""
Each cluster contains entities, where each entity has a name, type, and description. The clusters are identified with unique numeric labels.

Here is the clustering information:
{{
    "clusters": [
        {", ".join([f'{{"label": {label}, "entities": {cluster}}}' for label, cluster in clusters.items()])}
    ]
}}

Input description:
"{image_entity['description']}"

Question: Based on the clustering, which numeric label does the input description belong to? Respond only with a single numeric label (e.g., "0", "1", or "2") and nothing else. Do not include any explanations or additional text.
"""
            prompt_system = """You are an advanced AI assistant trained to categorize descriptions. Your task is to determine the category of an input description based on the following entity clusters."""
            try:
                target_label = int(get_llm_response(cur_prompt=prompt_user, system_content=prompt_system))
            except (ValueError, TypeError):  # 捕获转换失败或返回值类型错误
                target_label = 1  # 默认值为1
            image_entity_with_labels.append({
                "entity_name": image_entity["entity_name"],
                "label": target_label,
                "description": image_entity["description"],
                "entity_type": image_entity.get("entity_type", "N/A")
            })

    # Step 4: 生成聚类结果
    text_clustering_results = []
    for label in set(labels):
        text_clustering_results.append({
            "label": label,
            "entities": [
                {
                    "entity_name": nearby_text_entity_list[idx]["entity_name"],
                    "entity_type": nearby_text_entity_list[idx]["entity_type"],
                    "description": nearby_text_entity_list[idx]["description"],
                }
                for idx in range(len(labels))
                if labels[idx] == label
            ]
        })

    return image_entity_with_labels, text_clustering_results

def judge_text_entity_alignment_clustering(image_entity_with_labels, text_clustering_results):
    """
    使用 LLM 判断是否需要融合实体，并输出融合结果。

    Parameters:
        image_entity_with_labels (list): 图像实体及其对应类别的列表，每项为 {"entity_name": ..., "label": ..., "description": ..., "entity_type": ...}。
        text_clustering_results (list): 聚类后的文本实体列表，每项为 {"label": ..., "entities": [...]}。

    Returns:
        merged_entities (list): 融合的实体列表，每项为 {
            "entity_name": ..., 
            "entity_type": ..., 
            "description": ..., 
            "source_image_entities": [...], 
            "source_text_entities": [...]
        }。
    """
    # 构建融合任务的上下文
    clusters_info = []
    for cluster in text_clustering_results:
        clusters_info.append({
            "label": cluster["label"],
            "text_entities": [
                {
                    "entity_name": entity["entity_name"],
                    "entity_type": entity["entity_type"],
                    "description": entity["description"],
                }
                for entity in cluster["entities"]
            ]
        })

    # 构建输入 prompt
    prompt_user = f"""
You are tasked with aligning image entities and text entities based on their labels and descriptions. Below are the clusters and the entities they contain.

Clusters information:
{{
    "clusters": [
        {", ".join([f'{{"label": {c["label"]}, "text_entities": {c["text_entities"]}}}' for c in clusters_info])}
    ]
}}

Image entities with labels:
{[
    {
        "entity_name": e["entity_name"],
        "label": e["label"],
        "description": e["description"],
        "entity_type": e["entity_type"]
    }
    for e in image_entity_with_labels
]}

Instruction:
1. For each image entity, look at the corresponding cluster (same label).
2. Compare the description and type of the image entity with the text entities in the same cluster.
3. Identify matching entities between the image entities and text entities within the same cluster (same label).
4. For each match, create a new unified entity by merging the descriptions and including the source entities under "source_image_entities" and "source_text_entities".
5. Output a JSON list where each item represents a merged entity with the following structure:
    {{
        "entity_name": "Newly merged entity name",
        "entity_type": "Type of the merged entity",
        "description": "Merged description of the entity",
        "source_image_entities": ["List of matched image entity names"],
        "source_text_entities": ["List of matched text entity names"]
    }}
Include only one JSON list as the output, strictly following the structure above.
"""
    prompt_system = """You are an AI assistant skilled in aligning entities based on semantic descriptions and cluster information. Use the provided instructions to merge entities accurately."""

    # 调用 LLM 获取融合结果
    merged_entities = get_llm_response(cur_prompt=prompt_user, system_content=prompt_system)
    normalized_merged_entities = normalize_to_json_list(merged_entities)
    return [item for item in normalized_merged_entities if item["source_image_entities"] and item["source_text_entities"]]


# 示例调用
if __name__ == "__main__":
    # 加载模型
    import os
    from sentence_transformers import SentenceTransformer
    cache_path = "./cache"
    embedding_model_dir = os.path.join(cache_path, "all-MiniLM-L6-v2")
    model = SentenceTransformer(embedding_model_dir, device="cpu")
    nearby_entities1 = [
        {
            "entity_name": "ALBUS DUMBLEDORE (DUMBLEDORE)",
            "entity_type": "PERSON",
            "description": "Albus Dumbledore is the highly respected headmaster of Hogwarts, considered one of the greatest wizards, who allowed Hagrid to stay on as gamekeeper after his expulsion."
        },
        {
            "entity_name": "BARACK OBAMA",
            "entity_type": "PERSON",
            "description": "Barack Obama is the 44th president of the United States, known for his landmark healthcare reform and efforts in economic recovery after the 2008 financial crisis. He was awarded the Nobel Peace Prize in 2009."
        },
        {
            "entity_name": "MICHELLE OBAMA",
            "entity_type": "PERSON",
            "description": "Michelle Obama is a former First Lady of the United States, known for her work on education and healthy eating initiatives. She is the wife of Barack Obama and has become an influential figure in her own right."
        },
        {
            "entity_name": "THE OBAMA FOUNDATION",
            "entity_type": "ORGANIZATION",
            "description": "The Obama Foundation is a non-profit organization created by Barack Obama and Michelle Obama to promote leadership and empowerment. It focuses on global initiatives for the next generation of leaders."
        },
        {
            "entity_name": "VOLDEMORT (YOU-KNOW-WHO)",
            "entity_type": "PERSON",
            "description": "Voldemort, also known as You-Know-Who, is a powerful dark wizard who killed Harry's parents and tried to kill Harry but failed, marking a significant event in the wizarding world."
        },
        {
            "entity_name": "THE WHITE HOUSE",
            "entity_type": "ORGANIZATION",
            "description": "The White House is the official residence and workplace of the President of the United States. Barack Obama served as the President and worked here during his two terms in office."
        },
        {
            "entity_name": "NOBEL PEACE PRIZE",
            "entity_type": "AWARD",
            "description": "The Nobel Peace Prize is awarded annually to individuals or organizations that have made significant contributions to peace. Barack Obama won this award in 2009 for his efforts to strengthen international diplomacy."
        }
    ]

    nearby_relationships1 = [
    {
        "src_id": "BARACK OBAMA",
        "tgt_id": "MICHELLE OBAMA",
        "weight": 10.0,
        "description": "Barack Obama and Michelle Obama are married."
    },
    {
        "src_id": "BARACK OBAMA",
        "tgt_id": "NOBEL PEACE PRIZE",
        "weight": 5.0,
        "description": "Barack Obama won the Nobel Peace Prize in 2009."
    },
    {
        "src_id": "BARACK OBAMA",
        "tgt_id": "THE WHITE HOUSE",
        "weight": 7.0,
        "description": "Barack Obama served as the President of the United States and worked at the White House from 2009 to 2017."
    }
    ]

    nearby_entities2 = [
        {
            "entity_name": "ALBUS DUMBLEDORE (DUMBLEDORE)",
            "entity_type": "PERSON",
            "description": "Albus Dumbledore is the highly respected headmaster of Hogwarts, considered one of the greatest wizards, who allowed Hagrid to stay on as gamekeeper after his expulsion."
        },
        {
            "entity_name": "THE WHITE HOUSE",
            "entity_type": "ORGANIZATION",
            "description": "The White House is the official residence and workplace of the President of the United States. Barack Obama served as the President and worked here during his two terms in office."
        },
        {
            "entity_name": "NOBEL PEACE PRIZE",
            "entity_type": "AWARD",
            "description": "The Nobel Peace Prize is awarded annually to individuals or organizations that have made significant contributions to peace. Barack Obama won this award in 2009 for his efforts to strengthen international diplomacy."
        },
        {
            "entity_name": "BARACK OBAMA",
            "entity_type": "PERSON",
            "description": "Barack Obama is the 44th president of the United States, known for his landmark healthcare reform and efforts in economic recovery after the 2008 financial crisis. He was awarded the Nobel Peace Prize in 2009."
        },
        {
            "entity_name": "MICHELLE OBAMA",
            "entity_type": "PERSON",
            "description": "Michelle Obama is a former First Lady of the United States, known for her work on education and healthy eating initiatives. She is the wife of Barack Obama and has become an influential figure in her own right."
        },
        {
            "entity_name": "THE OBAMA FOUNDATION",
            "entity_type": "ORGANIZATION",
            "description": "The Obama Foundation is a non-profit organization created by Barack Obama and Michelle Obama to promote leadership and empowerment. It focuses on global initiatives for the next generation of leaders."
        },
        {
            "entity_name": "VOLDEMORT (YOU-KNOW-WHO)",
            "entity_type": "PERSON",
            "description": "Voldemort, also known as You-Know-Who, is a powerful dark wizard who killed Harry's parents and tried to kill Harry but failed, marking a significant event in the wizarding world."
        }
    ]

    nearby_relationships2 = [
    {
        "src_id": "BARACK OBAMA",
        "tgt_id": "MICHELLE OBAMA",
        "weight": 10.0,
        "description": "Barack Obama and Michelle Obama are married."
    },
    {
        "src_id": "BARACK OBAMA",
        "tgt_id": "THE OBAMA FOUNDATION",
        "weight": 8.0,
        "description": "Barack Obama co-founded the Obama Foundation with Michelle Obama."
    },
    {
        "src_id": "MICHELLE OBAMA",
        "tgt_id": "THE OBAMA FOUNDATION",
        "weight": 8.0,
        "description": "Michelle Obama co-founded the Obama Foundation with Barack Obama."
    },
    {
        "src_id": "BARACK OBAMA",
        "tgt_id": "NOBEL PEACE PRIZE",
        "weight": 5.0,
        "description": "Barack Obama won the Nobel Peace Prize in 2009."
    },
    {
        "src_id": "BARACK OBAMA",
        "tgt_id": "THE WHITE HOUSE",
        "weight": 7.0,
        "description": "Barack Obama served as the President of the United States and worked at the White House from 2009 to 2017."
    },
    {
        "src_id": "MICHELLE OBAMA",
        "tgt_id": "THE WHITE HOUSE",
        "weight": 7.0,
        "description": "Michelle Obama served as the First Lady of the United States and lived at the White House from 2009 to 2017."
    },
    {
        "src_id": "ALBUS DUMBLEDORE (DUMBLEDORE)",
        "tgt_id": "VOLDEMORT (YOU-KNOW-WHO)",
        "weight": 9.0,
        "description": "Dumbledore is aware of Voldemort's actions and their consequences on the Potter family and himself.",
    }
    ]


    filtered_image_entity_list = [
    {
        "entity_name": "OLD MAN",
        "entity_type": "PERSON",
        "description": "Professor McGonagall is anxious to discuss the disappearance of Dumbledore and the rumors surrounding the Potter family."
    },
    {
        "entity_name": "MAN",
        "entity_type": "PERSON",
        "description": "Barack Obama is the 44th president of the United States, known for his efforts to strengthen international diplomacy."
    },
    {
        "entity_name": "WOMAN",
        "entity_type": "PERSON",
        "description": "Michelle Obama is the first African American First Lady of the United States, known for her advocacy on education and healthy eating."
    }
    ]

    description1 = "Professor McGonagall is anxious to discuss the disappearance of Dumbledore and the rumors surrounding the Potter family."
    description2 = "Barack Obama is the 44th president of the United States..."
    description3 = "Michelle Obama is the first African American First Lady of the United States..."
    description4 = "The Nobel Prize in Peace is awarded annually to individuals or organizations that have made significant contributions to peace."
  
    result1 = get_possible_entities_image_clustering(
        model=model,
        clustering_method="spectral",
        classify_method="knn",
        image_entity_description=description1,
        nearby_text_entity_list=nearby_entities1,
        nearby_relationship_list=nearby_relationships1,
    )

    print(result1)
    print("---")

    image_entity_with_labels, text_clustering_results = get_possible_entities_text_clustering(
        model=model,
        clustering_method="spectral",
        classify_method="knn",
        filtered_image_entity_list=filtered_image_entity_list,
        nearby_text_entity_list=nearby_entities1,
        nearby_relationship_list=nearby_relationships1,
    )
    
    #result2 = judge_text_entity_alignment_clustering(image_entity_with_labels, text_clustering_results)
    print(image_entity_with_labels)
    print("---")
    print(text_clustering_results)
    print("---")
    #print(result2)