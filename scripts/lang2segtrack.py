import base64
import threading
import queue
from io import BytesIO

import cv2
import torch
import gc
import numpy as np
import imageio
from PIL import Image

from models.gdino.models.gdino import GDINO
from models.gdino.utils import display_image_with_boxes
from models.sam2.sam import SAM
from utils.color import COLOR
import pyrealsense2 as rs


class Lang2SegTrack:
    def __init__(self, sam_type: str, model_path: str, video_path: str, output_path: str,
                 first_boxes:list[list] | None = None, save_video=True, interactive_mode=True,
                 gdino_16=False, device="cuda:0", mode="realtime"):
        self.sam_type = sam_type # the type of SAM model to use
        self.model_path = model_path # the path to the SAM model checkpoint
        self.video_path = video_path # the path to the video to track. If mode="video", this param is required.
        self.output_path = output_path # the path to save the output video. If save_video=False, this param is ignored.
        self.save_video = save_video # whether to save the output video
        self.interactive_mode = interactive_mode # whether to use interactive mode
        self.device = device
        self.mode = mode # the mode to run the tracker. "video" or "realtime"

        self.sam = SAM()
        self.sam.build_model(self.sam_type, self.model_path, device=device)
        self.gdino = GDINO()
        self.gdino_16 = gdino_16
        if not self.gdino_16:
            print("Building GroundingDINO model...")
            self.gdino.build_model(device=device)

        self.input_queue = queue.Queue()
        if first_boxes is not None:
            self.start = first_boxes # the initial bounding boxes to track.
        else:
            self.start = []
        self.history = []
        self.latest = None
        self.drawing = False
        self.add_new = False
        self.reset = False
        self.ix, self.iy = -1, -1
        self.frame_display = None
        self.height, self.width = None, None

    def input_thread(self):
        while True:
            user_input = input()
            self.input_queue.put(user_input)

    def draw_bbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self.start.append((x, y))
                self.latest = (x, y)
                self.add_new = True
                cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            else:
                self.drawing = True
                self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img = param.copy()
            cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Video Tracking", img)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            if abs(x - self.ix) > 2 and abs(y - self.iy) > 2:
                bbox = [self.ix, self.iy, x, y]
                self.start.append(bbox)
                self.latest = bbox
                self.add_new = True
                cv2.rectangle(param, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.drawing = False

    def add_to_state(self, predictor, state, list):
        for id, item in enumerate(list):
            if len(item) == 4:
                x1, y1, x2, y2 = item
                cv2.rectangle(self.frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                predictor.add_new_points_or_box(state, box=item, frame_idx=state["num_frames"] - 1, obj_id=id)
            else:
                x, y = item
                cv2.circle(self.frame_display, (x, y), 5, (0, 255, 0), -1)
                pt = torch.tensor([[x, y]], dtype=torch.float32)
                lbl = torch.tensor([1], dtype=torch.int32)
                predictor.add_new_points_or_box(state, points=pt, labels=lbl, frame_idx=state["num_frames"] - 1, obj_id=id)

    def track_and_visualize(self, predictor, state, frame, writer):
        has_input = any(len(state["point_inputs_per_obj"][i]) > 0 for i in range(len(state["point_inputs_per_obj"])))
        if has_input:
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, state["num_frames"] - 1, 1):
                self.history.clear()
                for obj_id, mask in zip(obj_ids, masks):
                    mask = mask[0].cpu().numpy() > 0.0
                    nonzero = np.argwhere(mask)
                    if nonzero.size == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = nonzero.min(axis=0)
                        y_max, x_max = nonzero.max(axis=0)
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    self.draw_mask_and_bbox(frame, mask, bbox, obj_id)
                    self.history.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        else:
            cv2.imshow("Video Tracking", self.frame_display)

        if writer:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)
        cv2.imshow("Video Tracking", frame)

    def draw_mask_and_bbox(self, frame, mask, bbox, obj_id):
        mask_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_img[mask] = COLOR[obj_id % len(COLOR)]
        frame[:] = cv2.addWeighted(frame, 1, mask_img, 0.6, 0)
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR[obj_id % len(COLOR)], 2)
        cv2.putText(frame, f"obj_{obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[obj_id % len(COLOR)], 2)

    def track(self):

        predictor = self.sam.video_predictor

        if self.mode == "realtime":
            print("Start with realtime mode.")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
        elif self.mode == "video":
            print("Start with video mode.")
            cap = cv2.VideoCapture(self.video_path)
            ret, color_image = cap.read()

        self.height, self.width = color_image.shape[:2]

        if self.save_video:
            writer = imageio.get_writer(self.output_path, fps=30)
        else:
            writer = None

        cv2.namedWindow("Video Tracking")

        if self.interactive_mode:
            threading.Thread(target=self.input_thread, daemon=True).start()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state_from_numpy_frames([color_image], offload_state_to_cpu=False)
            while True:
                if self.mode == "realtime":
                    frames = pipeline.wait_for_frames()
                    ret = frames.get_color_frame()
                    frame = np.asanyarray(ret.get_data())
                elif self.mode == "video":
                    ret, frame = cap.read()
                if not ret:
                    continue
                self.frame_display = frame.copy()

                if self.interactive_mode:
                    cv2.setMouseCallback("Video Tracking", self.draw_bbox, param=self.frame_display)

                # Handle text input to trigger GDINO
                if not self.input_queue.empty() and self.interactive_mode:
                    text = self.input_queue.get()
                    print(f"Goal: {text}")
                    out = self.gdino.predict([Image.fromarray(frame)], [text], 0.3, 0.25)
                    boxes = [[int(v) for v in box] for box in out[0]["boxes"].cpu().numpy().tolist()]
                    if len(self.history) == 0:
                        self.start.extend(boxes)
                    else:
                        self.reset = True
                    self.history.extend(boxes)

                if self.latest and self.add_new:
                    self.history.append(self.latest)
                    self.add_new = False
                    self.reset = True

                # Add new objects for tracking
                if len(self.history) > len(self.start) and self.reset:
                    predictor.reset_state(state)
                    self.add_to_state(predictor, state, self.history)
                    self.reset = False
                else:
                    self.add_to_state(predictor, state, self.start)

                self.start.clear()
                predictor.append_frame_to_inference_state(state, frame)
                self.track_and_visualize(predictor, state, frame, writer)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if self.mode == "realtime":
            pipeline.stop()
        else:
            cap.release()
        pipeline.stop()
        if writer:
            writer.close()
        cv2.destroyAllWindows()
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()

    def predict_img(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """Predicts masks for given images and text prompts using GDINO and SAM models.

        Parameters:
            images_pil (list[Image.Image]): List of input images.
            texts_prompt (list[str]): List of text prompts corresponding to the images.
            box_threshold (float): Threshold for box predictions.
            text_threshold (float): Threshold for text predictions.

        Returns:
            list[dict]: List of results containing masks and other outputs for each image.
            Output format:
            [{
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "masks": np.ndarray,
                "mask_scores": np.ndarray,
            }, ...]
        """
        if self.gdino_16:
            if len(images_pil) > 1:
                raise ValueError("GroundingDINO_16 only support single image")
            byte_io = BytesIO()
            images_pil[0].save(byte_io, format='PNG')
            image_bytes = byte_io.getvalue()
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            texts_prompt = texts_prompt[0]
            gdino_results = self.gdino.predict_dino_1_6_pro(base64_encoded, texts_prompt, box_threshold, text_threshold)
        else:
            gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        for idx, result in enumerate(gdino_results):
            result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            if result["labels"]:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)
        if sam_images:
            # print(f"Predicting {len(sam_boxes)} masks")
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
            # print(f"Predicted {len(all_results)} masks")
        return all_results


if __name__ == "__main__":

    tracker = Lang2SegTrack(sam_type="sam2.1_hiera_tiny",
                            model_path="models/sam2/checkpoints/sam2.1_hiera_tiny.pt",
                            video_path="assets/05_default_juggle.mp4",
                            output_path="processed_video.mp4",
                            mode="video",)
    tracker.track()

    # out = tracker.predict_img(
    #     [Image.open("assets/img_01.jpg")],
    #     ["cup.ball"],
    # )
    # print(out)
    # img = cv2.imread("assets/img_01.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # display_image_with_boxes(img, list(out[0]["boxes"]), out[0]["scores"], list(out[0]["labels"]))
