import os
from PyPDF2 import PdfReader

def count_pdf_pages_in_folder(folder_path):
    # 创建一个字典，键是页数，值是该页数的PDF文件数量
    page_count_list = []

    # 获取文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理PDF文件
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 打开并读取PDF文件
                with open(file_path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    page_count = len(pdf_reader.pages)
                    page_count_list.append(page_count)

            except Exception as e:
                print(f"无法读取文件 {filename}: {e}")

    return page_count_list

# 调用函数，指定文件夹路径
folder_path = '/Users/xueyaowan/Documents/硕士毕业设计/newproject/fusion_research/fusion_dataset/fusion_dataset_pdf/novel'
page_count_list = count_pdf_pages_in_folder(folder_path)
# 打印每种页数的PDF文件数量
print(page_count_list)