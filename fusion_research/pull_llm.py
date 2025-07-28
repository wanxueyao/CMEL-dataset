import re
import json
from openai import OpenAI

API_KEY = ""
MODEL = ""
URL = ""
MM_API_KEY = ""
MM_MODEL = ""
MM_URL = ""

# 正则化处理函数
def normalize_to_json(output):
    # 使用正则提取JSON部分
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # 验证JSON格式是否正确
            json_obj = json.loads(json_str)
            return json_obj  # 返回标准化的JSON对象
        except json.JSONDecodeError as e:
            print(f"JSON解码失败: {e}")
            return None
    else:
        print("未找到有效的JSON部分")
        return None

def normalize_to_json_list(output):
    """
    提取并验证JSON列表格式的字符串，返回解析后的JSON对象列表。
    即使JSON不完整，也尝试提取尽可能多的内容。
    """
    # 去除转义符和多余空白符
    cleaned_output = output.replace('\\"', '"').strip()
    
    # 使用宽松的正则表达式提取可能的JSON片段
    match = re.search(r"\[\s*(\{.*?\})*?\s*]", cleaned_output, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        
        # 移除多余逗号（可能由于截断导致多余逗号）
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = re.sub(r",\s*}$", "}", json_str)

        try:
            # 尝试完整解析
            json_obj = json.loads(json_str)
            if isinstance(json_obj, list):
                return json_obj
        except json.JSONDecodeError:
            # 如果完整解析失败，尝试逐项解析
            print("完整解析失败，尝试逐项解析...")
            items = []
            for partial_match in re.finditer(r"\{.*?\}", json_str, re.DOTALL):
                try:
                    item = json.loads(partial_match.group(0))
                    items.append(item)
                except json.JSONDecodeError:
                    print("跳过无效的JSON片段")
            return items if items else []
    else:
        print("未找到有效的JSON片段")
        return []

# 用LLM进行回答
def get_llm_response(cur_prompt, system_content):
    client = OpenAI(
        base_url=URL, api_key=API_KEY
    )

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": cur_prompt},
        ],
    )

    response = completion.choices[0].message.content
    return response

# 调用多模态LLM
def get_mmllm_response(cur_prompt, system_content, img_base):
    client = OpenAI(
        base_url=MM_URL, api_key=MM_API_KEY
    )

    completion = client.chat.completions.create(
        model=MM_MODEL,
        messages=[
            {"role": "system", "content": [
                    {
                        "type": "text",
                        "text": system_content
                    }
                    ]},
            {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base}"},
                    },
                    {
                        "type": "text",
                        "text": cur_prompt
                    }
                    ]},
        ],
    )

    response = completion.choices[0].message.content
    return response