import json
import os

from methods import entity_alignment_embedding, entity_alignment_llm, align_single_image_entity, entity_alignment_clustering
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from difflib import SequenceMatcher

# embedding方法的重要参数
model = SentenceTransformer(
    '/cpfs02/user/lidong1/model/stella_en_1.5B_v5', trust_remote_code=True, device="cuda:0"
)
embedding_threshold = 0.35
# clustering方法的重要参数
# 聚类方法：kmeans,dbscan;pagerank,leiden;spectral
clustering_method = "pagerank"
# 分类方法：knn,llm
classify_method = "knn"

# 数据集存储路径
dataset_dir = "./fusion_research/mini_dataset"
output_file_name = 'clustering_pagerank_knn_results.txt'

# methods:task3, embedding, llm, clustering
method = "clustering"
def process_single_file_embedding(file_dir, model, embedding_threshold):
    result_path = os.path.join(file_dir, 'result.json')
    
    # 检查 result.json 是否存在
    if os.path.exists(result_path):
        print("result.json 已存在，跳过执行。")
        return
    image_data_path = os.path.join(file_dir, 'kv_store_image_data.json')
    with open(image_data_path,'r') as f:
        image_data = json.load(f)
    text_chunks_path = os.path.join(file_dir, 'kv_store_text_chunks.json')
    with open(text_chunks_path,'r') as f:
        text_chunks = json.load(f)
    chunk_knowledge_graph_path = os.path.join(file_dir, 'kv_store_chunk_knowledge_graph.json')
    with open(chunk_knowledge_graph_path,'r') as f:
        chunk_knowledge_graph = json.load(f)
    image_knowledge_graph_path = os.path.join(file_dir, 'kv_store_image_knowledge_graph.json')
    with open(image_knowledge_graph_path,'r') as f:
        image_knowledge_graph = json.load(f)
    result = {}
    for img_entity_name in image_data:
        print(f"Processing image entity {img_entity_name}...")
        matched_entity, merged_entities = entity_alignment_embedding(img_entity_name, image_data, text_chunks, chunk_knowledge_graph, image_knowledge_graph, model, embedding_threshold)
        result[img_entity_name] = {
            "matched_entity": matched_entity,
            "merged_entities": merged_entities,
        }
    with open(result_path, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

def process_single_file_llm(file_dir):
    result_path = os.path.join(file_dir, 'result.json')
    
    # 检查 result.json 是否存在
    if os.path.exists(result_path):
        print("result.json 已存在，跳过执行。")
        return
    image_data_path = os.path.join(file_dir, 'kv_store_image_data.json')
    with open(image_data_path,'r') as f:
        image_data = json.load(f)
    text_chunks_path = os.path.join(file_dir, 'kv_store_text_chunks.json')
    with open(text_chunks_path,'r') as f:
        text_chunks = json.load(f)
    chunk_knowledge_graph_path = os.path.join(file_dir, 'kv_store_chunk_knowledge_graph.json')
    with open(chunk_knowledge_graph_path,'r') as f:
        chunk_knowledge_graph = json.load(f)
    image_knowledge_graph_path = os.path.join(file_dir, 'kv_store_image_knowledge_graph.json')
    with open(image_knowledge_graph_path,'r') as f:
        image_knowledge_graph = json.load(f)
    result = {}
    for img_entity_name in image_data:
        print(f"Processing image entity {img_entity_name}...")
        matched_entity, merged_entities = entity_alignment_llm(img_entity_name, image_data, text_chunks, chunk_knowledge_graph, image_knowledge_graph)
        result[img_entity_name] = {
            "matched_entity": matched_entity,
            "merged_entities": merged_entities,
        }
    with open(result_path, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

def process_single_file_task3(file_dir):
    result_path = os.path.join(file_dir, 'result.json')
    
    # 检查 result.json 是否存在
    if os.path.exists(result_path):
        print("result.json 已存在，跳过执行。")
        return
    image_data_path = os.path.join(file_dir, 'kv_store_image_data.json')
    with open(image_data_path,'r') as f:
        image_data = json.load(f)
    text_chunks_path = os.path.join(file_dir, 'kv_store_text_chunks.json')
    with open(text_chunks_path,'r') as f:
        text_chunks = json.load(f)
    result = {}
    for img_entity_name in image_data:
        print(f"Processing image entity {img_entity_name}...")
        aligned_image_entity = align_single_image_entity(img_entity_name, image_data, text_chunks)
        if aligned_image_entity:
            result[img_entity_name] = aligned_image_entity
        else:
            result[img_entity_name] = "no match"
    with open(result_path, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

def process_single_file_clustering(file_dir, model, clustering_method, classify_method):
    result_path = os.path.join(file_dir, 'result.json')
    
    # 检查 result.json 是否存在
    if os.path.exists(result_path):
        print("result.json 已存在，跳过执行。")
        return
    image_data_path = os.path.join(file_dir, 'kv_store_image_data.json')
    with open(image_data_path,'r') as f:
        image_data = json.load(f)
    text_chunks_path = os.path.join(file_dir, 'kv_store_text_chunks.json')
    with open(text_chunks_path,'r') as f:
        text_chunks = json.load(f)
    chunk_knowledge_graph_path = os.path.join(file_dir, 'kv_store_chunk_knowledge_graph.json')
    with open(chunk_knowledge_graph_path,'r') as f:
        chunk_knowledge_graph = json.load(f)
    image_knowledge_graph_path = os.path.join(file_dir, 'kv_store_image_knowledge_graph.json')
    with open(image_knowledge_graph_path,'r') as f:
        image_knowledge_graph = json.load(f)
    result = {}
    for img_entity_name in image_data:
        print(f"Processing image entity {img_entity_name}...")
        matched_entity, merged_entities = entity_alignment_clustering(model, clustering_method, classify_method, img_entity_name, image_data, text_chunks, chunk_knowledge_graph, image_knowledge_graph)
        result[img_entity_name] = {
            "matched_entity": matched_entity,
            "merged_entities": merged_entities,
        }
    with open(result_path, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

def process_dataset(dataset_dir, method):
    for category in ["news", "novel", "paper"]:
        category_path = os.path.join(dataset_dir, category)

        if os.path.isdir(category_path):
            subfolders = os.listdir(category_path)
            for subfolder in tqdm(subfolders, desc=f"Processing subfolders in {category}", leave=False):
                subfolder_path = os.path.join(category_path, subfolder)
                print(subfolder)
                if os.path.isdir(subfolder_path):
                    if method == "embedding":
                        process_single_file_embedding(subfolder_path, model, embedding_threshold)
                    elif method == "llm":
                        process_single_file_llm(subfolder_path)
                    elif method == "task3":
                        process_single_file_task3(subfolder_path)
                    elif method == "clustering":
                        process_single_file_clustering(subfolder_path, model, clustering_method, classify_method)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

def calculate_task1_accuracy(aligned_file, result_file):
    # 读取 JSON 数据
    with open(aligned_file, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    with open(result_file, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    total_questions = correct_predictions = 0

    for image_key in aligned_data:
        aligned_entity = aligned_data[image_key]['matched_chunk_entity_name'].lower()
        result_entity = result_data[image_key]['matched_entity'].lower()

        total_questions += 1
        # 完全一致匹配
        if aligned_entity == result_entity:
            correct_predictions += 1
        # 包含匹配
        elif aligned_entity in result_entity or result_entity in aligned_entity:
            correct_predictions += 1
        # 字符匹配超过50%
        else:
            match_ratio = SequenceMatcher(None, aligned_entity, result_entity).ratio()
            if match_ratio > 0.5:
                correct_predictions += 1

    accuracy = correct_predictions / total_questions
    return accuracy, total_questions

def calculate_task2_accuracy(aligned_file, result_file):
    # 读取 JSON 数据
    with open(aligned_file, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    with open(result_file, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    total_entities = 0
    correct_entities = 0

    for image_key, aligned_entities in aligned_data.items():
        if image_key in result_data:
            result_entities = result_data[image_key].get('merged_entities') or []
            for aligned_entity in aligned_entities:
                aligned_image_entities = set(aligned_entity['source_image_entities'])
                aligned_text_entities = set(aligned_entity['source_text_entities'])

                for result_entity in result_entities:
                    result_image_entities = set(result_entity['source_image_entities'])
                    result_text_entities = set(result_entity['source_text_entities'])

                    if aligned_image_entities == result_image_entities and aligned_text_entities == result_text_entities:
                        correct_entities += 1

                total_entities += 1

    accuracy = correct_entities / total_entities if total_entities > 0 else 0
    return accuracy, total_entities

def calculate_task3_accuracy(aligned_file, result_file):
    # 读取 JSON 数据
    with open(aligned_file, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    with open(result_file, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    total_questions = 0
    correct_predictions = 0

    for image_key in aligned_data:
        aligned_entity = aligned_data[image_key]['entity_name'].lower()
        result_entity = result_data[image_key]['entity_name'].lower()

        total_questions += 1
        # 完全一致匹配
        if aligned_entity == result_entity:
            correct_predictions += 1
        # 包含匹配
        elif aligned_entity in result_entity or result_entity in aligned_entity:
            correct_predictions += 1
        # 字符匹配超过70%
        else:
            match_ratio = SequenceMatcher(None, aligned_entity, result_entity).ratio()
            if match_ratio > 0.7:
                correct_predictions += 1

    accuracy = correct_predictions / total_questions
    return accuracy, total_questions

def evaluate_dataset_task3(dataset_dir):
    categories = ['news', 'paper', 'novel']
    task3_correct = 0
    task3_total = 0
    task3_macro_accuracy = []

    task3_stats = {category: {'file_accuracies': [], 'correct': 0, 'total': 0} for category in categories}
    
    for category in categories:
        category_dir = os.path.join(dataset_dir, category)

        if not os.path.exists(category_dir):
            continue

        for file in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file)

            aligned_file = os.path.join(file_path, 'aligned_image_entity.json')
            result_file = os.path.join(file_path, 'result.json')

            if os.path.exists(aligned_file) and os.path.exists(result_file):
                accuracy, total_questions = calculate_task3_accuracy(aligned_file, result_file)
                correct_questions = accuracy * total_questions

                # 更新状态记录
                task3_stats[category]['file_accuracies'].append(accuracy)
                task3_stats[category]['correct'] += correct_questions
                task3_stats[category]['total'] += total_questions

                # 更新整体结果
                task3_correct += correct_questions
                task3_total += total_questions
                task3_macro_accuracy.append(accuracy)
    # 计算整体结果
    task3_micro_accuracy = task3_correct / task3_total if task3_total > 0 else 0
    category_results = {}

    for category in categories:
        file_accuracies = task3_stats[category]['file_accuracies']
        macro_accuracy = sum(file_accuracies) / len(file_accuracies) if file_accuracies else 0
        total = task3_stats[category]['total']
        correct = task3_stats[category]['correct']
        micro_accuracy = correct / total if total > 0 else 0

        category_results[category] = {
            'micro_accuracy': micro_accuracy,
            'macro_accuracy': macro_accuracy,
            'total_questions': total
        }            
    return {
        'task3': {
            'overall_micro_accuracy': task3_micro_accuracy,
            'overall_macro_accuracy': (sum(task3_macro_accuracy) / len(task3_macro_accuracy)) if isinstance(task3_macro_accuracy, list) and task3_macro_accuracy else (task3_macro_accuracy if isinstance(task3_macro_accuracy, float) else 0),
            'overall_total_questions': task3_total,
            'category_results': category_results
        }
    }

def evaluate_dataset(dataset_dir, method):
    if method == 'task3':
        return evaluate_dataset_task3(dataset_dir)
    categories = ['news', 'paper', 'novel']
    task1_overall_correct = 0
    task1_overall_total = 0
    task1_overall_macro_accuracy = []

    task2_overall_correct = 0
    task2_overall_total = 0
    task2_overall_macro_accuracy = []

    task1_stats = {category: {'file_accuracies': [], 'correct': 0, 'total': 0} for category in categories}
    task2_stats = {category: {'file_accuracies': [], 'correct': 0, 'total': 0} for category in categories}

    for category in categories:
        category_dir = os.path.join(dataset_dir, category)

        if not os.path.exists(category_dir):
            continue

        for file in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file)

            aligned_file = os.path.join(file_path, 'aligned_image_entity.json')
            result_file = os.path.join(file_path, 'result.json')
            aligned_task2_file = os.path.join(file_path, 'aligned_text_entity.json')

            if os.path.exists(aligned_file) and os.path.exists(result_file):
                accuracy, total_questions = calculate_task1_accuracy(aligned_file, result_file)
                correct_questions = accuracy * total_questions

                # 更新状态记录
                task1_stats[category]['file_accuracies'].append(accuracy)
                task1_stats[category]['correct'] += correct_questions
                task1_stats[category]['total'] += total_questions

                # 更新整体结果
                task1_overall_correct += correct_questions
                task1_overall_total += total_questions
                task1_overall_macro_accuracy.append(accuracy)

            if os.path.exists(aligned_task2_file) and os.path.exists(result_file):
                task2_accuracy, total_entities = calculate_task2_accuracy(aligned_task2_file, result_file)
                correct_entities = task2_accuracy * total_entities

                task2_stats[category]['file_accuracies'].append(task2_accuracy)
                task2_stats[category]['correct'] += correct_entities
                task2_stats[category]['total'] += total_entities

                task2_overall_correct += correct_entities
                task2_overall_total += total_entities
                task2_overall_macro_accuracy.append(task2_accuracy)

    # 计算整体结果
    task1_overall_micro_accuracy = task1_overall_correct / task1_overall_total if task1_overall_total > 0 else 0

    task2_overall_micro_accuracy = task2_overall_correct / task2_overall_total if task2_overall_total > 0 else 0

    category_results = {}
    task2_category_results = {}

    for category in categories:
        # Task 1
        task1_file_accuracies = task1_stats[category]['file_accuracies']
        task1_macro_accuracy = sum(task1_file_accuracies) / len(task1_file_accuracies) if task1_file_accuracies else 0
        task1_total = task1_stats[category]['total']
        task1_correct = task1_stats[category]['correct']
        task1_micro_accuracy = task1_correct / task1_total if task1_total > 0 else 0

        category_results[category] = {
            'micro_accuracy': task1_micro_accuracy,
            'macro_accuracy': task1_macro_accuracy,
            'total_questions': task1_total
        }

        # Task 2
        task2_file_accuracies = task2_stats[category]['file_accuracies']
        task2_macro_accuracy = sum(task2_file_accuracies) / len(task2_file_accuracies) if task2_file_accuracies else 0
        task2_total = task2_stats[category]['total']
        task2_correct = task2_stats[category]['correct']
        task2_micro_accuracy = task2_correct / task2_total if task2_total > 0 else 0

        task2_category_results[category] = {
            'micro_accuracy': task2_micro_accuracy,
            'macro_accuracy': task2_macro_accuracy,
            'total_entities': task2_total
        }

    return {
        'task1': {
            'overall_micro_accuracy': task1_overall_micro_accuracy,
            'overall_macro_accuracy': (sum(task1_overall_macro_accuracy) / len(task1_overall_macro_accuracy)) if isinstance(task1_overall_macro_accuracy, list) and task1_overall_macro_accuracy else (task1_overall_macro_accuracy if isinstance(task1_overall_macro_accuracy, float) else 0),
            'overall_total_questions': task1_overall_total,
            'category_results': category_results
        },
        'task2': {
            'overall_micro_accuracy': task2_overall_micro_accuracy,
            'overall_macro_accuracy': (sum(task2_overall_macro_accuracy) / len(task2_overall_macro_accuracy)) if isinstance(task2_overall_macro_accuracy, list) and task2_overall_macro_accuracy else (task2_overall_macro_accuracy if isinstance(task2_overall_macro_accuracy, float) else 0),
            'overall_total_entities': task2_overall_total,
            'category_results': task2_category_results
        }
    }

def write_results_to_txt(results, output_file, method):
    with open(output_file, 'w', encoding='utf-8') as f:
        if method == 'task3':
            # Task 3 Results
            f.write("Task 3 Results:\n")
            f.write(f"Overall Micro Accuracy: {results['task3']['overall_micro_accuracy']:.4f}\n")
            f.write(f"Overall Macro Accuracy: {results['task3']['overall_macro_accuracy']:.4f}\n")
            f.write(f"Overall Total Questions: {results['task3']['overall_total_questions']}\n\n")

            for category, stats in results['task3']['category_results'].items():
                f.write(f"Category: {category}\n")
                f.write(f"  Micro Accuracy: {stats['micro_accuracy']:.4f}\n")
                f.write(f"  Macro Accuracy: {stats['macro_accuracy']:.4f}\n")
                f.write(f"  Total Questions: {stats['total_questions']}\n\n")
        else:
            # Task 1 Results
            f.write("Task 1 Results:\n")
            f.write(f"Overall Micro Accuracy: {results['task1']['overall_micro_accuracy']:.4f}\n")
            f.write(f"Overall Macro Accuracy: {results['task1']['overall_macro_accuracy']:.4f}\n")
            f.write(f"Overall Total Questions: {results['task1']['overall_total_questions']}\n\n")

            for category, stats in results['task1']['category_results'].items():
                f.write(f"Category: {category}\n")
                f.write(f"  Micro Accuracy: {stats['micro_accuracy']:.4f}\n")
                f.write(f"  Macro Accuracy: {stats['macro_accuracy']:.4f}\n")
                f.write(f"  Total Questions: {stats['total_questions']}\n\n")

            # Task 2 Results
            f.write("Task 2 Results:\n")
            f.write(f"Overall Micro Accuracy: {results['task2']['overall_micro_accuracy']:.4f}\n")
            f.write(f"Overall Macro Accuracy: {results['task2']['overall_macro_accuracy']:.4f}\n")
            f.write(f"Overall Total Entities: {results['task2']['overall_total_entities']}\n\n")

            for category, stats in results['task2']['category_results'].items():
                f.write(f"Category: {category}\n")
                f.write(f"  Micro Accuracy: {stats['micro_accuracy']:.4f}\n")
                f.write(f"  Macro Accuracy: {stats['macro_accuracy']:.4f}\n")
                f.write(f"  Total Entities: {stats['total_entities']}\n\n")

if __name__ == "__main__":
    process_dataset(dataset_dir, method)
    results = evaluate_dataset(dataset_dir, method)
    output_file = os.path.join(dataset_dir, output_file_name)
    write_results_to_txt(results, output_file, method)
