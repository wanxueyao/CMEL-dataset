import PyPDF2
import math

def split_pdf(input_pdf_path, num_parts):
    # 打开 PDF 文件
    with open(input_pdf_path, 'rb') as infile:
        reader = PyPDF2.PdfReader(infile)
        total_pages = len(reader.pages)
        
        # 计算每个新 PDF 的页数
        pages_per_part = math.ceil(total_pages / num_parts)

        # 获取文件名并去掉扩展名
        base_name = input_pdf_path.rsplit('.', 1)[0]

        for i in range(num_parts):
            writer = PyPDF2.PdfWriter()

            # 计算当前部分的起始页和结束页
            start_page = i * pages_per_part
            end_page = min((i + 1) * pages_per_part, total_pages)

            # 将指定页数的页面添加到新 PDF 中
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            # 创建新的输出文件名
            output_pdf_path = f"{base_name}({i + 1}).pdf"

            # 写入新文件
            with open(output_pdf_path, 'wb') as output_pdf:
                writer.write(output_pdf)

            print(f"创建了新的 PDF 文件: {output_pdf_path}")

# 示例：拆分 PDF 文件
input_pdf_path = "./fusion_dataset_pdf/Charlottes Web (E. B. White) .pdf"
num_parts = 3  # 拆分成的文件数
split_pdf(input_pdf_path, num_parts)