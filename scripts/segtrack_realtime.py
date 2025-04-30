import threading
import queue
import cv2
import torch
import gc
import numpy as np
import imageio
from PIL import Image

from models.gdino.models.gdino import GDINO
from models.sam2.sam2.build_sam import build_sam2_video_predictor
from utils.color import COLOR
from utils.utils import determine_model_cfg
import pyrealsense2 as rs


class Lang2SegTrack:
    def __init__(self, model_path: str, video_path: str, output_path: str,
                 first_boxes:list[list] | None = None, save_video=True, device="cuda:0", mode="realtime"):
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.save_video = save_video
        self.device = device
        self.mode = mode

        self.input_queue = queue.Queue()
        if first_boxes is not None:
            self.start = first_boxes
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

    def run(self):
        model_cfg = determine_model_cfg(self.model_path)
        predictor = build_sam2_video_predictor(model_cfg, self.model_path, device=self.device)

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

        gdino = GDINO()
        print("Building Grounding-DINO ......")
        gdino.build_model()

        self.height, self.width = color_image.shape[:2]

        if self.save_video:
            writer = imageio.get_writer(self.output_path, fps=30)
        else:
            writer = None

        cv2.namedWindow("Video Tracking")
        threading.Thread(target=self.input_thread, daemon=True).start()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state_from_numpy_frames([color_image], offload_video_to_cpu=True)
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
                cv2.setMouseCallback("Video Tracking", self.draw_bbox, param=self.frame_display)

                # Handle text input to trigger GDINO
                if not self.input_queue.empty():
                    text = self.input_queue.get()
                    print(f"Goal: {text}")
                    out = gdino.predict([Image.fromarray(frame)], [text], 0.3, 0.25)
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

        cap.release()
        pipeline.stop()
        if writer:
            writer.close()
        cv2.destroyAllWindows()
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":

    tracker = Lang2SegTrack(model_path="models/sam2/checkpoints/sam2.1_hiera_tiny.pt",
                            video_path="assets/05_default_juggle.mp4",
                            output_path="processed_video.mp4",
                            first_boxes=[[27,64,59,225]],
                            mode="realtime")
    tracker.run()
