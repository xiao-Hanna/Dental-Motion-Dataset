import os
import json
from tqdm import tqdm

# 输入路径
data_root = "data1/data24"
output_json = "annotations/person_keypoints_train.json"

# 创建 COCO 格式的空字典
coco_dict = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "keypoints": [],
            "skeleton": []
        }
    ]
}

ann_id = 1
img_id = 1

# 遍历所有图片
for fname in tqdm(os.listdir(data_root)):
    if fname.endswith(".jpg") or fname.endswith(".png"):
        base = os.path.splitext(fname)[0]
        img_file = os.path.join(data_root, fname)
        txt_file = os.path.join(data_root, base + ".txt")

        if not os.path.exists(txt_file):
            continue

        # 读取关键点
        keypoints = []
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                # 跳过空行
                if not line:
                    continue
                # 跳过注释行（以#开头的行）
                if line.startswith("#"):
                    continue
                try:
                    # 尝试分割并转换为浮点数
                    parts = line.split(",")
                    # 确保有两个数值
                    if len(parts) == 2:
                        x, y = map(float, parts)
                        keypoints.extend([x, y, 2])  # (x,y,可见度)
                    else:
                        print(f"警告: {txt_file} 中的行 '{line}' 格式不正确，已跳过")
                except ValueError:
                    # 处理无法转换为浮点数的情况
                    print(f"警告: {txt_file} 中的行 '{line}' 包含非数值数据，已跳过")

        # 如果没有有效的关键点数据，跳过此文件
        if not keypoints:
            print(f"警告: {txt_file} 中没有有效的关键点数据，已跳过")
            continue

        # 更新 category 里的关键点名字（p1, p2...）
        if not coco_dict["categories"][0]["keypoints"]:
            num_kpt = len(keypoints) // 3
            coco_dict["categories"][0]["keypoints"] = [f"p{i}" for i in range(1, num_kpt + 1)]

        # 构造 annotation
        x_coords = keypoints[0::3]
        y_coords = keypoints[1::3]
        bbox = [min(x_coords), min(y_coords),
                max(x_coords) - min(x_coords),
                max(y_coords) - min(y_coords)]

        # 获取真实图片尺寸（需要安装OpenCV）
        try:
            import cv2

            img = cv2.imread(img_file)
            height, width = img.shape[:2]
        except:
            # 如果无法读取，使用默认值
            height, width = 1080, 1920

        coco_dict["images"].append({
            "id": img_id,
            "file_name": fname,
            "height": height,
            "width": width
        })

        coco_dict["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": len(keypoints) // 3,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })

        ann_id += 1
        img_id += 1

# 保存 JSON
os.makedirs("annotations", exist_ok=True)
with open(output_json, "w") as f:
    json.dump(coco_dict, f, indent=4)

print(f"✅ 转换完成，标注文件已保存: {output_json}")
