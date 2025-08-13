import cv2
import mrcnn.model as modellib
import pandas as pd
from mrcnn.visualize import InferenceConfig, get_mask_contours, random_colors, draw_mask
import numpy as np
import diplib as dip
import glob
import os
import time

# 设置输入和输出路径
input_path = "images/"  # 输入文件夹路径
output_path = "output/"

# 日志文件路径
log_file_path = os.path.join(output_path, 'processing_log.xlsx')

# 加载 Mask-RCNN
config = InferenceConfig(num_classes=2, image_size=1024)
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
model.load_weights(filepath="dnn_model/mask_rcnn_pollen_detect.h5", by_name=True)

# 加载类别
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# 初始化日志数据字典和总表格数据
log_data = []  # 用于存储所有文件夹的详细数据
summary_data = []

total_images = 0  # 初始化总图片数量
total_folders = 0  # 初始化总文件夹数量


def apply_mask_black(image, mask):
    # 对图像应用黑色掩码
    for i in range(3):
        image[:, :, i] = np.where(mask == 0, 0, 255)
    return image


def display_instances(image, boxes, masks, ids, names, scores, alpha=50):
    n_instances = boxes.shape[0]

    if not n_instances:
        print('没有检测到实例')
        return image

    mask = np.zeros(image[:, :, 0].shape)
    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        label = names[ids[i]]
        if label == 'tube':
            mask += masks[:, :, i]

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    image_with_mask = apply_mask_black(image.copy(), mask)  # 保留原图的颜色，生成带黑色掩码的图像

    # 调整透明度
    overlay = image.copy()
    overlay[mask == 255] = (0, 255, 0)  # 使用绿色
    cv2.addWeighted(overlay, alpha / 100, image_with_mask, 1 - alpha / 100, 0, image_with_mask)

    return image_with_mask if mask.sum() != 0 else mask


def process_image(image_path, subfolder_name, alpha):
    print(f"正在处理图像: {image_path}")
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"读取图像失败: {image_path}")
            return
        start_time = time.time()  # 开始计时
        results = model.detect([image], verbose=0)
        r = results[0]
        frame = display_instances(image, r['rois'], r['masks'], r['class_ids'], classes, r['scores'], alpha)

        if frame.sum() != 0:
            name = os.path.basename(image_path)
            subfolder_output_path = os.path.join(output_path, subfolder_name)
            os.makedirs(os.path.join(subfolder_output_path, 'black'), exist_ok=True)
            os.makedirs(os.path.join(subfolder_output_path, 'color'), exist_ok=True)
            os.makedirs(os.path.join(subfolder_output_path, 'skeletonization'), exist_ok=True)

            cv2.imwrite(os.path.join(subfolder_output_path, 'black', name), frame)
            result = model.detect([image])[0]
            class_ids = result["class_ids"]
            number = pd.value_counts(class_ids).rename(index={1: "花粉粒:", 2: "花粉管:"})
            print(number)
            colors = random_colors(len(class_ids))

            # 在原始图像上添加标记
            color_image = image.copy()
            for i in range(len(class_ids)):
                class_id = result["class_ids"][i]
                box = result["rois"][i]
                y1, x1, y2, x2 = box
                class_name = classes[class_id]
                cv2.rectangle(color_image, (x1, y1), (x2, y2), colors[i], 2)
                cv2.putText(color_image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, colors[i], 1)
                contours = get_mask_contours(result["masks"][:, :, i])
                for cnt in contours:
                    cv2.polylines(color_image, [cnt], True, colors[i], 2)
                    color_image = draw_mask(color_image, [cnt], colors[i], alpha=0.5)

            # 保存带有彩色标记的图像

            
            cv2.imwrite(os.path.join(subfolder_output_path, 'color', name), color_image)

            img_grey = cv2.imread(os.path.join(subfolder_output_path, 'black', name), cv2.IMREAD_GRAYSCALE)
            after_median = cv2.medianBlur(img_grey, 3)
            thresh = 140
            bin_image = after_median > thresh
            sk = dip.EuclideanSkeleton(bin_image, endPixelCondition='two neighbors')
            sk = (np.array(sk) * 255).astype(np.uint8)

            cv2.imwrite(os.path.join(subfolder_output_path, 'skeletonization', name), sk)
            number_of_white_pix = np.sum(sk == 255)
            print(f'{name} 白色像素数:', number_of_white_pix)

            # 计算平均花粉管长度
            if number.get("花粉管:", 0) > 0:
                avg_pollen_tube_length = number_of_white_pix / number.get("花粉管:", 1)
            else:
                avg_pollen_tube_length = 0

            # 计算萌发率
            pollen_grain_count = number.get("花粉粒:", 0)
            pollen_tube_count = number.get("花粉管:", 0)
            if pollen_grain_count > 0:
                germination_rate = (pollen_tube_count / pollen_grain_count) * 100
            else:
                germination_rate = 0

            # 添加到日志数据中，插入文件夹名称
            log_data.append({
                "文件夹名称": subfolder_name,
                "图像名称": name,
                "花粉粒数量": pollen_grain_count,
                "花粉管数量": pollen_tube_count,
                "花粉管长度（白色像素）": number_of_white_pix,
                "平均花粉管长度": avg_pollen_tube_length,
                "萌发率（%）": germination_rate,
                "处理时间（秒）": time.time() - start_time  # 计算处理时间
            })
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")


def process_all_images(alpha=50):
    global total_images, total_folders
    start_time = time.time()  # 开始计时
    # 遍历所有子文件夹
    for subfolder_name in os.listdir(input_path):
        subfolder_path = os.path.join(input_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            total_folders += 1
            print(f"正在处理文件夹: {subfolder_name}")
            # 读取子文件夹中的所有图片文件，包括.bmp格式
            image_paths = glob.glob(os.path.join(subfolder_path, '*.jpg')) + \
                          glob.glob(os.path.join(subfolder_path, '*.jpeg')) + \
                          glob.glob(os.path.join(subfolder_path, '*.png')) + \
                          glob.glob(os.path.join(subfolder_path, '*.bmp'))

            total_images += len(image_paths)  # 统计总图片数量
            print(f"在 {subfolder_name} 中找到 {len(image_paths)} 张图像。")

            # 单线程顺序处理子文件夹中的所有图像
            for image_path in image_paths:
                process_image(image_path, subfolder_name, alpha)

            # 计算该子文件夹的平均萌发率和平均花粉管长度
            folder_data = [entry for entry in log_data if entry["文件夹名称"] == subfolder_name]
            avg_germination_rate = np.mean([entry["萌发率（%）"] for entry in folder_data])
            avg_pollen_tube_length = np.mean([entry["平均花粉管长度"] for entry in folder_data])
            summary_data.append({
                "文件夹名称": subfolder_name,
                "平均萌发率（%）": avg_germination_rate,
                "平均花粉管长度": avg_pollen_tube_length
            })
    # 将日志数据保存到 Excel 文件中
    with pd.ExcelWriter(log_file_path, engine='openpyxl') as writer:
        # 保存Summary总表格信息到第一个工作表
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # 保存所有详细数据到第二个工作表
        log_df = pd.DataFrame(log_data)
        log_df.to_excel(writer, sheet_name='Log Data', index=False)

    print(f"Total Images Processed: {total_images}")
    print(f"Total Folders Processed: {total_folders}")
    print(f"Processing complete. Time taken: {time.time() - start_time:.2f} seconds.")

# 调用函数处理所有图片
process_all_images()
