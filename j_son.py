import json
import os

def save_to_json(file_path, data):
    """
    将数据保存到指定的 JSON 文件中。如果文件存在，则更新数据；如果文件不存在，则创建新的文件。
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = json.load(f)  # 读取现有内容
    else:
        content = []

    # 将新的数据添加到内容中
    content.append(data)

    with open(file_path, 'w') as f:
        json.dump(content, f, indent=4)
