import os
import re
import io
import json
import base64
from PIL import Image
from openai import OpenAI
from collections import OrderedDict

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_txt_string(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()  
    data = json.loads(content)
    return json.dumps(data, ensure_ascii=False)
    

def generate_jsonl_entry(image_base64, text):
    return{
        "messages": 
        [
            {
                "role": "user",
                "content":
                [
                    {
                        "type": "image_url",
                        "image_url": 
                        {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": text
            },
        ]
    }

train_jsonls = []
test_jsonls = []
train_jsonl_name = "train0.jsonl"
test_jsonl_name = "test0.jsonl"
train_pattern = re.compile(r"^(?!.*flip).*_(?:[1-9]|1[0-2])_rot0\.png$")

for root, dirs, files in os.walk(r"./origin_image_done_split_trans_label"):
    for file in files:
        if not file.endswith('.png'):
            # print(f'Skipping non-png file: {file}')
            continue

        img_source_path = os.path.join(root, file)
        # print(f'Processing file: {img_source_path}')
        txt_source_path = os.path.join(root, file.replace('.png', '.txt'))
        if not os.path.exists(txt_source_path):
            print(f'Skipping, no corresponding txt file: {txt_source_path}')
            continue
        img_encoded = encode_image(img_source_path)
        txt_string = get_txt_string(txt_source_path)
        json_entry = generate_jsonl_entry(img_encoded, txt_string)
        if train_pattern.match(file):
            train_jsonls.append(json_entry)
            print(f'Added to train set: {file}')
        else:
            test_jsonls.append(json_entry)

# Write the JSONL entries to a file
with open(train_jsonl_name, "w", encoding="utf-8") as f:
    for entry in train_jsonls:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
with open(test_jsonl_name, "w", encoding="utf-8") as f:
    for entry in test_jsonls:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")