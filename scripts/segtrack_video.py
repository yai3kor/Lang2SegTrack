import logging
import queue

import cv2
import numpy as np
import torch
import gc
import imageio
import threading
from PIL import Image

from models.gdino.models.gdino import GDINO
from models.sam2.sam2.build_sam import build_sam2_video_predictor

color = [
    (128, 0, 0),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (64, 0, 0),
    (64, 128, 0),
    (64, 0, 128),
    (0, 64, 128),
    (128, 64, 0),
    (128, 0, 64),
    (0, 128, 64),
]

# 用于存储手动绘制的 bbox 和点
start = []
latest = None
drawing = False
add_new = False
ix, iy = -1, -1

# 键盘输入相关
input_queue = queue.Queue()
def input_thread():
    while True:
        user_input = input()
        input_queue.put(user_input)

def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, start, latest, add_new
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            start.append((x, y))
            latest = (x, y)
            add_new = True
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Video Tracking", param)
        else:
            drawing = True
            ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = param.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Video Tracking", img)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            min_drag_threshold = 2
            if abs(x - ix) > min_drag_threshold and abs(y - iy) > min_drag_threshold:
                start.append([ix, iy, x, y])
                latest = [ix, iy, x, y]
                add_new = True
                cv2.rectangle(param, (ix, iy), (x, y), (0, 255, 0), 2)
            drawing = False
            cv2.imshow("Video Tracking", param)

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def main(first_list=None):
    global start, latest, add_new
    if first_list is not None:
        start = first_list

    #
    model_path = "/home/jj/JKW/samurai/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    save_output = True
    output_path = "../models/sam2/output.mp4"
    model_cfg = determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
    logging.basicConfig(level=logging.CRITICAL + 1)

    # 加载视频文件
    video_path = "/home/jj/JKW/samurai/realsense_output.mp4"  # 改成你的 MP4 路径
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Init gdino
    gdino = GDINO()
    gdino.build_model()

    history = []

    try:
        print("Wait for first frame...")
        ret, color_image = cap.read()
        if not ret:
            print("Failed to read the first frame.")
            return
        height, width = color_image.shape[:2]

        # 视频写入器
        if save_output:
            writer = imageio.get_writer(output_path, fps=30)

        cv2.namedWindow("Video Tracking")
        threading.Thread(target=input_thread, daemon=True).start()

        # 初始化跟踪器
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state_from_numpy_frames([color_image], offload_video_to_cpu=True)
            exit_flag = True
            reset = None
            while exit_flag:
                ret, frame = cap.read()
                if not ret:
                    break  # 播完了
                frame_display = frame.copy()

                cv2.setMouseCallback("Video Tracking", draw_bbox, param=frame_display)

                # 键盘输入检测
                if not input_queue.empty():
                    user_input = input_queue.get()
                    print(f"The goal：{user_input}\n")
                    out = gdino.predict(
                        [Image.fromarray(frame)],
                        [user_input],
                        0.3,
                        0.25,
                    )
                    gdino_boxes = out[0]["boxes"].cpu().numpy().tolist()
                    gdino_boxes = [[int(num) for num in sublist] for sublist in gdino_boxes]
                    if len(history) == 0:
                        start.extend(gdino_boxes)
                        history.extend(gdino_boxes)
                    else:
                        history.extend(gdino_boxes)
                        reset = True

                if latest is not None and add_new:
                    history.append(latest)
                    add_new = False
                    reset = True

                if len(history) > len(start) and reset:
                    predictor.reset_state(state)
                    for id, item in enumerate(history):
                        if len(item) == 4:
                            x1, y1, x2, y2 = item
                            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            _, _, masks = predictor.add_new_points_or_box(state, box=item,
                                                                          frame_idx=state["num_frames"] - 1,
                                                                          obj_id=id)
                        else:
                            x, y = item
                            cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)
                            points_tensor = torch.tensor([[x, y]], dtype=torch.float32)
                            labels_tensor = torch.tensor([1], dtype=torch.int32)
                            _, _, masks = predictor.add_new_points_or_box(state, points=points_tensor,
                                                                          labels=labels_tensor,
                                                                          frame_idx=state["num_frames"] - 1,
                                                                          obj_id=id)
                    reset = False
                else:
                    for id, item in enumerate(start):
                        if len(item) == 4:
                            x1, y1, x2, y2 = item
                            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            _, _, masks = predictor.add_new_points_or_box(state, box=item,
                                                                          frame_idx=state["num_frames"] - 1,
                                                                          obj_id=id)
                        else:
                            x, y = item
                            cv2.circle(frame_display, (x, y), 5, (0, 255, 0), -1)
                            points_tensor = torch.tensor([[x, y]], dtype=torch.float32)
                            labels_tensor = torch.tensor([1], dtype=torch.int32)
                            _, _, masks = predictor.add_new_points_or_box(state, points=points_tensor,
                                                                          labels=labels_tensor,
                                                                          frame_idx=state["num_frames"] - 1,
                                                                          obj_id=id)

                start = []

                predictor.append_frame_to_inference_state(state, frame)

                new_frame_idx = state["num_frames"] - 1

                has_points_or_boxes = any(len(state["point_inputs_per_obj"][obj_idx]) > 0
                                          for obj_idx in range(len(state["point_inputs_per_obj"])))

                if has_points_or_boxes:
                    for frame_idx, object_ids, masks in predictor.propagate_in_video(state,
                                                                                     start_frame_idx=new_frame_idx,
                                                                                     max_frame_num_to_track=1):
                        mask_to_vis = {}
                        bbox_to_vis = {}
                        history.clear()
                        for obj_id, mask in zip(object_ids, masks):
                            mask = mask[0].cpu().numpy()
                            mask = mask > 0.0
                            non_zero_indices = np.argwhere(mask)
                            if len(non_zero_indices) == 0:
                                item = [0, 0, 0, 0]
                            else:
                                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                                item = [x_min, y_min, x_max - x_min, y_max - y_min]
                            bbox_to_vis[obj_id] = item
                            mask_to_vis[obj_id] = mask

                        for obj_id, mask in mask_to_vis.items():
                            mask_img = np.zeros((height, width, 3), np.uint8)
                            mask_img[mask] = color[obj_id % len(color)]
                            frame = cv2.addWeighted(frame, 1, mask_img, 0.6, 0)

                        for obj_id, item in bbox_to_vis.items():
                            label = f"obj_{obj_id}"
                            cv2.rectangle(frame, (item[0], item[1]),
                                          (item[0] + item[2], item[1] + item[3]),
                                          color[obj_id % len(color)], 2)
                            cv2.putText(frame, label, (item[0], item[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        color[obj_id % len(color)], 2)
                            history.append([item[0], item[1], item[0] + item[2], item[1] + item[3]])

                        if save_output:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            writer.append_data(rgb_frame)

                        cv2.imshow("Video Tracking", frame)
                else:
                    cv2.imshow("Video Tracking", frame_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit_flag = False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if save_output:
            writer.close()
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
