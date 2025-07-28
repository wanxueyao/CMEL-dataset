import os
import json
from lxml import etree

def update_json_file(old_path, new_path, json_file_path):
    """更新 JSON 文件中的路径"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated = False
    for key, value in data.items():
        if "image_path" in value and old_path in value["image_path"]:
            value["image_path"] = value["image_path"].replace(old_path, new_path)
            updated = True

    if updated:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Updated JSON file: {json_file_path}")


def update_graphml_file(old_path, new_path, graphml_file_path):
    """更新 GraphML 文件中的路径"""
    tree = etree.parse(graphml_file_path)
    root = tree.getroot()

    namespaces = {"graphml": "http://graphml.graphdrawing.org/xmlns"}
    data_elements = root.xpath("//graphml:data", namespaces=namespaces)

    updated = False
    for data in data_elements:
        if data.text and old_path in data.text:
            data.text = data.text.replace(old_path, new_path)
            updated = True

    if updated:
        tree.write(graphml_file_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
        print(f"Updated GraphML file: {graphml_file_path}")


def process_folder(old_path, new_path, base_folder):
    """处理主文件夹"""
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file == "kv_store_image_data.json":
                json_file_path = os.path.join(root, file)
                update_json_file(old_path, new_path, json_file_path)
            elif file.endswith(".graphml"):
                graphml_file_path = os.path.join(root, file)
                update_graphml_file(old_path, new_path, graphml_file_path)


if __name__ == "__main__":
    old_path = "./fusion_research/fusion_dataset"
    # 新路径,base_folder为存储路径，通常与new_path一致
    new_path = "./fusion_research/fusion_dataset"
    base_folder = "./fusion_research/fusion_dataset"

    # 定义子目录
    subfolders = ["paper", "novel", "news"]

    # 遍历子目录，逐一处理
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        process_folder(old_path, new_path, subfolder_path)