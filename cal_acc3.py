import os
import re
import json
import base64
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict


load_dotenv()  # 讀取 .env 檔
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("model3")
client = OpenAI(api_key=api_key)
label_suffix = ".label3.txt"
output_path = "acc3.csv"
outout_path_tex = "acc3.tex"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_txt_string(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()  
    return content

def get_tag(file):
    first_tag = None
    second_tag = None
    third_tag = None
    fourth_tag = None
    fifth_tag = None
    train_pattern = re.compile(r"^(?!.*flip).*_(?:[1-9]|1[0-2])_rot[0-3]\.png$")
    if train_pattern.match(file):
        first_tag = "train"
    else:
        first_tag = "test"
    
    class_pattern = re.compile(r"(C\d{3})_")
    second_tag = class_pattern.match(file).group(1) 
    
    type_pattern = re.compile(r"C\d+_(\d+)")
    index = type_pattern.search(file).group(1)
    index = int(index)
    if index % 6 == 1 or index % 6 == 2:
        third_tag = "printed_only"
    elif index % 6 == 3 or index % 6 == 4:
        third_tag = "mixed_print_hand"
    elif index % 6 == 5 or index % 6 == 0:
        third_tag = "handwritten_only"
    
    if "flip" in file:
        fourth_tag = "flip"
    else:
        fourth_tag = "no_flip"

    if "rot0" in file:
        fifth_tag = "rot0"
    elif "rot1" in file:
        fifth_tag = "rot1"
    elif "rot2" in file:
        fifth_tag = "rot2"
    elif "rot3" in file:
        fifth_tag = "rot3"
    return first_tag, second_tag, third_tag, fourth_tag, fifth_tag


def generate_message(image_base64):
    return[
            {
                "role": "user",
                "content":
                [
                    {
                        "type": "image_url",
                        "image_url": 
                        {
                            "url": f"data:image/jpeg;base64,{full_image_descirbe_encode}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": 
                        {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please describe the image in detail using json format."
                    },
                ]
            }
        ]
# generate training data
full_image_descirbe_path = r'./full_image/full_image.png'
full_image_descirbe_encode = encode_image(full_image_descirbe_path)
    

for root, dirs, files in os.walk(r"./origin_image_done_split_trans_label_predict"):
    for file in files:
        if not file.endswith('.png'):
            # print(f'Skipping non-png file: {file}')
            continue
        img_source_path = os.path.join(root, file)
        txt_predcit_path =img_source_path.replace('.png', label_suffix)
        if os.path.exists(txt_predcit_path):
            print(f'Skipping, label file already exists: {txt_predcit_path}')
            continue
        image_base64 = encode_image(img_source_path)
        messages = generate_message(image_base64)
        
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            max_completion_tokens=1000,
            top_p=1,
            temperature=0,
        )
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            with open(txt_predcit_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'Successfully processed: {file}')
        else:
            print(f'No valid response for: {file}')

def default_metrics():
    return {
        "total_count": 0,
        "json_correct": 0,
        "structure_correct": 0,
        "type_correct": 0,
        "edge_correct": 0,
        "all_correct": 0
    }

correct_cal_dict = defaultdict(default_metrics)
for root, dirs, files in os.walk(r"./origin_image_done_split_trans_label_predict"):
    for file in files:
        if not file.endswith('.png'):
            # print(f'Skipping non-png file: {file}')
            continue
        img_source_path = os.path.join(root, file)
        txt_predcit_path =img_source_path.replace('.png', label_suffix)
        txt_label_path = img_source_path.replace('.png', '.txt')
        txt_predcit = get_txt_string(txt_predcit_path)
        txt_predcit_clean = txt_predcit.replace("```json", "").replace("```", "").strip()
        txt_label = get_txt_string(txt_label_path)
        first_tag, second_tag, third_tag, fourth_tag, fifth_tag = get_tag(file)
        try:
            label = json.loads(txt_label)
        except json.JSONDecodeError:
            print(f'Error decoding label JSON for: {txt_label_path}')
            continue  # 假設 label 是正確的
        
        correct_cal_dict[frozenset([first_tag, second_tag, third_tag, fourth_tag, fifth_tag])]["total_count"] += 1
        correct_cal_dict[frozenset([first_tag])]["total_count"] += 1
        correct_cal_dict[frozenset([second_tag])]["total_count"] += 1
        correct_cal_dict[frozenset([third_tag])]["total_count"] += 1
        correct_cal_dict[frozenset([fourth_tag])]["total_count"] += 1
        correct_cal_dict[frozenset([fifth_tag])]["total_count"] += 1
        correct_cal_dict['summary']["total_count"] += 1
        try:
            prediction = json.loads(txt_predcit_clean)
            correct_cal_dict[frozenset([first_tag, second_tag, third_tag, fourth_tag, fifth_tag])]["json_correct"] += 1
            correct_cal_dict[frozenset([first_tag])]["json_correct"] += 1
            correct_cal_dict[frozenset([second_tag])]["json_correct"] += 1
            correct_cal_dict[frozenset([third_tag])]["json_correct"] += 1
            correct_cal_dict[frozenset([fourth_tag])]["json_correct"] += 1
            correct_cal_dict[frozenset([fifth_tag])]["json_correct"] += 1
            correct_cal_dict['summary']["json_correct"] += 1
        except json.JSONDecodeError:
            continue

        required_keys = label.keys()
        if not all(k in prediction for k in required_keys):
            continue
        else:
            correct_cal_dict[frozenset([first_tag, second_tag, third_tag, fourth_tag, fifth_tag])]["structure_correct"] += 1
            correct_cal_dict[frozenset([first_tag])]["structure_correct"] += 1
            correct_cal_dict[frozenset([second_tag])]["structure_correct"] += 1
            correct_cal_dict[frozenset([third_tag])]["structure_correct"] += 1
            correct_cal_dict[frozenset([fourth_tag])]["structure_correct"] += 1
            correct_cal_dict[frozenset([fifth_tag])]["structure_correct"] += 1
            correct_cal_dict['summary']["structure_correct"] += 1

        type_correct = False
        if label["type"] == prediction["type"]:
            correct_cal_dict[frozenset([first_tag, second_tag, third_tag, fourth_tag, fifth_tag])]["type_correct"] += 1
            correct_cal_dict[frozenset([first_tag])]["type_correct"] += 1
            correct_cal_dict[frozenset([second_tag])]["type_correct"] += 1
            correct_cal_dict[frozenset([third_tag])]["type_correct"] += 1
            correct_cal_dict[frozenset([fourth_tag])]["type_correct"] += 1
            correct_cal_dict[frozenset([fifth_tag])]["type_correct"] += 1
            correct_cal_dict['summary']["type_correct"] += 1
            type_correct = True

        edge_correct = False
        for key, value in label.items():
            if key not in prediction:
                continue 
            if isinstance(value, str) and isinstance(prediction[key], str):
                if value != prediction[key]:
                    break
        else:
            edge_correct = True
            correct_cal_dict[frozenset([first_tag, second_tag, third_tag, fourth_tag, fifth_tag])]["edge_correct"] += 1
            correct_cal_dict[frozenset([first_tag])]["edge_correct"] += 1
            correct_cal_dict[frozenset([second_tag])]["edge_correct"] += 1
            correct_cal_dict[frozenset([third_tag])]["edge_correct"] += 1
            correct_cal_dict[frozenset([fourth_tag])]["edge_correct"] += 1
            correct_cal_dict[frozenset([fifth_tag])]["edge_correct"] += 1
            correct_cal_dict['summary']["edge_correct"] += 1

        if type_correct and edge_correct:
            correct_cal_dict[frozenset([first_tag, second_tag, third_tag, fourth_tag, fifth_tag])]["all_correct"] += 1
            correct_cal_dict[frozenset([first_tag])]["all_correct"] += 1
            correct_cal_dict[frozenset([second_tag])]["all_correct"] += 1
            correct_cal_dict[frozenset([third_tag])]["all_correct"] += 1
            correct_cal_dict[frozenset([fourth_tag])]["all_correct"] += 1
            correct_cal_dict[frozenset([fifth_tag])]["all_correct"] += 1
            correct_cal_dict['summary']["all_correct"] += 1        
data = []
for key, metrics in correct_cal_dict.items():
    if not isinstance(metrics, dict):
        continue  # 排除像 'summary': 0 這樣的錯誤初始化情況

    label = '|'.join(sorted(map(str, key))) if isinstance(key, frozenset) else key
    total = metrics["total_count"]
    row = {
        "tag": label,
        "total_count": total,
        "json_correct": f'{metrics["json_correct"]} ({metrics["json_correct"] / total * 100:.2f}%)' if total else "",
        "structure_correct": f'{metrics["structure_correct"]} ({metrics["structure_correct"] / total * 100:.2f}%)' if total else "",
        "type_correct": f'{metrics["type_correct"]} ({metrics["type_correct"] / total * 100:.2f}%)' if total else "",
        "edge_correct": f'{metrics["edge_correct"]} ({metrics["edge_correct"] / total * 100:.2f}%)' if total else "",
        "all_correct": f'{metrics["all_correct"]} ({metrics["all_correct"] / total * 100:.2f}%)' if total else ""
    }
    data.append(row)

df = pd.DataFrame(data)

# 自訂排序邏輯
hand_types = ["handwritten_only", "mixed_print_hand", "printed_only"]
classes = ["C105", "C106", "C413", "C445", "C505", "C603"]
flip = ["flip", "no_flip"]
rot = ["rot0", "rot1", "rot2", "rot3"]
groups = ["train", "test"]
special = ["summary"]

def sort_key(tag):
    parts = tag.split("|")
    if len(parts) == 5:
        return (0, tag)
    elif tag in hand_types:
        return (1, hand_types.index(tag))
    elif tag in classes:
        return (2, classes.index(tag))
    elif tag in flip:
        return (3, flip.index(tag))
    elif tag in rot:
        return (4, rot.index(tag))
    elif tag in groups:
        return (5, groups.index(tag))
    elif tag in special:
        return (6, special.index(tag))
    else:
        return (7, tag)

df = df.sort_values(by="tag", key=lambda x: x.map(sort_key)).reset_index(drop=True)
df.to_csv(output_path, index=False)
df.to_latex(outout_path_tex,index=False,escape=True,longtable=True,column_format=r">{\raggedright\arraybackslash}p{5cm}rrrrrr")
