
---

# Language-to-Segment&Track

Language-driven visual segmentation and object tracking system based on [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) and [SAMURAI](https://github.com/yangchris11/samurai).

---
### 🔥 News

- **`2025/05/08`**: Optimize memory usage and code logic.
- **`2025/04/29`**: Initial version submission.

---

### 🔍 What Can It Do?

#### For Images:
- Accepts a text prompt and returns:
  - Label
  - Score
  - Bounding Box
  - Segmentation Mask
- Supports **batch processing** of images

![](assets/Figure_1.png)

#### For Video Files:
- Allows initializing multiple object tracks via a bounding box list in the first frame
- Enables **interactive tracking and segmentation** on any frame using:
  - Mouse click
  - Manual box drawing
  - Text prompt

#### For Real-Time Video (Camera):
> Default camera supported is Intel RealSense. For others, please modify the video capture method manually.

- Interactive tracking and segmentation with:
  - Mouse click
  - Manual box
  - Text prompt

https://github.com/user-attachments/assets/40a340c7-d818-493f-b86a-bb8ed5ca517c

---

### ❓ Why Use [SAMURAI](https://github.com/yangchris11/samurai)? How is it Different from [SAM2](https://github.com/facebookresearch/sam2)?

- **SAMURAI** is a zero-shot visual tracking model based on SAM2
- It outperforms SAM2 in **visual tracking capabilities**

https://github.com/user-attachments/assets/9d368ca7-2e9b-4fed-9da0-d2efbf620d88

This demo comes from [SAMURAI](https://github.com/yangchris11/samurai) and shows the comparison of video tracking performance between [SAM2](https://github.com/facebookresearch/sam2) and [SAMURAI](https://github.com/yangchris11/samurai). All rights are reserved to the copyright owners (TM & © Universal (2019)). This clip is not intended for commercial use and is solely for academic demonstration in a research paper. Original source can be found [here](https://www.youtube.com/watch?v=cwUzUzpG8aM&t=4s).)

---

## 🚀 Get Started

We recommend using **Anaconda** for environment management.

### ✅ Prerequisites

- OS: Ubuntu >= 18.04  
- NVIDIA Driver (`nvidia-smi`) >= **550**

---

### 🛠 Installation

```bash
git clone https://github.com/wngkj/Lang2SegTrack.git
cd Lang2SegTrack

conda create -n lang2segtrack python=3.10
conda activate lang2segtrack

pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124

cd models/sam2
pip install -e .
pip install -e ".[notebooks]"

cd ../gdino
pip install -e .
```

---

### 📦 Download SAM2.1 Checkpoints

Place them into `sam2/checkpoints/`:

- [sam2.1_hiera_tiny](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

---

## Usage


  ```bash
  python scripts/lang2segtrack.py
  ```
- **`track`**: Track objects in video files or real-time video streams by using first-frame prompts, text-prompts, mouse click or manual boxes drawing.


- **`predict_img`**: Predict objects in images by using text-prompts.


- **`track_realtime_fast`**: Track objects in real-time video streams faster. Only supports first-frame prompts.
---

## 🙏 Acknowledgments

This project builds upon outstanding prior work:

- [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAMURAI](https://github.com/yangchris11/samurai)
- [SAM2](https://github.com/facebookresearch/sam2)
- [Language-Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)
