# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
# from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo, SaliencyDemo

# constants
WINDOW_NAME = "COCO detections"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/ssd_scratch/cvit/bhavberi/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
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
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    print("Started Program")
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = SaliencyDemo(cfg, parallel=True)

    saliency_path = '/ssd_scratch/cvit/bhavberi/masks'

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            # logger.info(
            #     "{}: {} in {:.2f}s".format(
            #         path,
            #         "detected {} instances".format(len(predictions["instances"]))
            #         if "instances" in predictions
            #         else "finished",
            #         time.time() - start_time,
            #     )
            # )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(os.path.dirname(args.video_input))
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                # os.makedirs(output_fname, exist_ok=True)
                output_fname = output_fname + '_video' + file_ext

                add_frame = False
                
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            else:
                output_fname = args.output
                add_frame = True
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
        assert os.path.isfile(args.video_input)
        i = 0
        saliency_base_path = '/ssd_scratch/cvit/bhavberi/masks/' + basename + '/masks/'
        for frame, predictions, info in tqdm.tqdm(demo.run_on_video(video, add_frame=add_frame), total=num_frames):
            if args.output:
                if add_frame:
                    output_file.write(vis_frame)
                else:
                    final_image_name = "frame_{:04d}.jpg".format(i + 1)
                    saliency_path = os.path.join(saliency_base_path, final_image_name)

                    saliency_mask = cv2.imread(saliency_path)
                    saliency_mask = cv2.resize(saliency_mask, (width, height))
                    saliency_mask = cv2.cvtColor(saliency_mask, cv2.COLOR_BGR2GRAY)

                    vis_frame, mask = demo.overlay_saliency(frame, saliency_mask, predictions, info, args.n)

                    output_file.write(vis_frame)
                    
                    # Save the vis_frame to a file
                    os.makedirs('/ssd_scratch/cvit/bhavberi/detectron2_out/{}'.format(basename), exist_ok=True)
                    cv2.imwrite('/ssd_scratch/cvit/bhavberi/detectron2_out/{}/{:04d}.jpg'.format(basename, i), vis_frame)
                    os.makedirs('/ssd_scratch/cvit/bhavberi/detectron2_out/{}/masks'.format(basename), exist_ok=True)
                    cv2.imwrite('/ssd_scratch/cvit/bhavberi/detectron2_out/{}/masks/{:04d}.jpg'.format(basename, i), mask)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
            i+=1
        
        video.release()
        if args.output:
            output_file.release()
            pass
        else:
            cv2.destroyAllWindows()


# -- Setup detectron2
"""
pip install 'git+https://github.com/facebookresearch/detectron2.git'
mkdir -p /ssd_scratch/cvit/bhavberi
git clone https://github.com/facebookresearch/detectron2.git /ssd_scratch/cvit/bhavberi/detectron2
"""
# -- Get ViNet Saliency maps (Currently in 048, 051, 062)
"""
rsync -avzWP -e 'ssh -J bhavberi@ada.iiit.ac.in' bhavberi@gnode048:/ssd_scratch/cvit/bhavberi/masks /ssd_scratch/cvit/bhavberi/
rsync -avzWP -e 'ssh -J bhavberi@ada.iiit.ac.in' bhavberi@gnode051:/ssd_scratch/cvit/bhavberi/masks /ssd_scratch/cvit/bhavberi/
"""
# -- Change the paths below as required (output, input, config)
"""
python3 ~/detectron2/run.py --config-file /ssd_scratch/cvit/bhavberi/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input /home2/bhavberi/varun_videos/badminton/video.mp4 --confidence-threshold 0.6 --output /home2/bhavberi/detectron2_videos/ --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl
"""