# Salient Object Segmentation using ViNet and Detectron2

## Pipeline

The pipeline consists of 3 stages:
- **Getting ViNet inferences** on the videos (the code needs to be provided with the path for the inferences)

- Main Code
    - **Running VOS model** on the videos separately.

    - **Getting the top n (here n=2) salient objects**. For this, the background segmentations are removed from the results of detectron2, and then the vinet mask is overlayed on the remaining segmentation results, and the segmentations with the highest scores are picked and shown.

### 1. Install Detectron2 (Change path as required)
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
mkdir -p /ssd_scratch/cvit/bhavberi
git clone https://github.com/facebookresearch/detectron2.git /ssd_scratch/cvit/bhavberi/detectron2
```

### 2. Setup the ViNet model
[https://github.com/samyak0210/ViNet](https://github.com/samyak0210/ViNet)

### 3. Get the ViNet inferences on the video/s you want to run

### 4. Run the code. Change the paths below as required (output, input, config)
```bash
python3 run.py --config-file /ssd_scratch/cvit/bhavberi/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input /home2/bhavberi/varun_videos/badminton/video.mp4 --confidence-threshold 0.6 --output /home2/bhavberi/detectron2_videos/ --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl
```

This will save the output video in the output directory specified above.

The configurations for the run.py file can be found in the below code block.

```python
parser.add_argument(
        "--config-file",
        default="/ssd_scratch/cvit/bhavberi/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of Salient objects to show",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
```

> To run for multiple videos, use the below command
```bash
videos_dir="/home2/bhavberi/varun_videos"

# Find all video files in the specified directory and its subdirectories
video_files=$(find "$videos_dir" -type f -name "video.mp4" | sort)

for video_file in $video_files; do
    echo "Processing $video_file"
    python3 run.py --config-file /ssd_scratch/cvit/bhavberi/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input "$video_file" --confidence-threshold 0.6 --output /home2/bhavberi/detectron2_videos/ --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl
done

```