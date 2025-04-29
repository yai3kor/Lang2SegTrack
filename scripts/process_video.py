import argparse
import os
import os.path as osp

import imageio
import numpy as np
import cv2
import torch
import gc
import sys

from utils.color import COLOR
from utils.utils import determine_model_cfg, bbox_process, prepare_frames_or_path
from models.sam2.sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
import imageio.v3 as iio

def main(args, bbox_list:list[list[float]]):
    device = args.device
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=device)
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = bbox_process(bbox_list)

    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith(".jpg")])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

    if args.save_to_video:
        writer = imageio.get_writer(args.video_output_path, fps=30)

    mask_dir = args.mask_dir
    if args.mask_dir is not None:
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(exist_ok=True, parents=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        all_masks = []
        for idx, (bbox, track_label) in enumerate(prompts.values()):
            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=idx)
            all_masks.append(masks)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state, disable_display=False):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask
                if mask_dir is not None:
                    mask_path = mask_dir / f'OBJ{obj_id:02}_{frame_idx:04}.png'
                    iio.imwrite(mask_path, mask.astype(np.uint8) * 255)

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = COLOR[obj_id % len(COLOR)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.5, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    label = f"obj_{obj_id}"
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), COLOR[obj_id % len(COLOR)], 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                COLOR[obj_id % len(COLOR)], 2)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                writer.append_data(img)

        if args.save_to_video:
            writer.close()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="assets/02_cups.mp4")
    parser.add_argument("--model_path", default="models/sam2/checkpoints/sam2.1_hiera_tiny.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="processed_video.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    parser.add_argument("--mask_dir", help="If provided, save mask images to the given directory")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args, bbox_list=[[364.53216552734375, 426.52178955078125, 437.9630126953125, 500.3838195800781],
                          [260.3464050292969, 0.1987624168395996, 445.94500732421875, 198.92283630371094],
                          [829.1287841796875, 292.64874267578125, 999.095703125, 544.427978515625],
                          [568.9965209960938, 291.2940673828125, 738.1513671875, 544.0554809570312]])