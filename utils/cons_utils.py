import copy
import json
import re
import numpy as np
import argparse
import pandas as pd
import random
import torch
import os
import logging


# ANSI escape codes for colors
class Formatter(logging.Formatter):
    COLOR_CODES = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[91m',  # Red
        'ERROR': '\033[93m',  # Yellow
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET_CODE = '\033[0m'  # Reset color

    def format(self, record):
        log_color = self.COLOR_CODES.get(record.levelname, self.RESET_CODE)
        message = super().format(record)
        return f"{log_color}{message}{self.RESET_CODE}"


def load_logger(name):
    # Custom logger setup
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler with color formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define formatter with color
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)
    return logger


class BaseOptions(object):
    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_type", type=str, default='TimeChat',
                            choices=['Video-ChatGPT', 'Video-LLaMA', 'Video-LLaMA2', 'Video-LLaVA', 'VTimeLLM', 'TimeChat', 'VTG-LLM', 'VideoChat2', 'GPT4', 'Gemini'],
                            help="A list of Video-LLMs.")
        parser.add_argument("--dset_name", type=str, default="charades", choices=['activitynet', 'charades'],
                            help="Dataset name.")
        parser.add_argument("--task", type=str, default="consistency", choices=['grounding', 'consistency'],
                            help="Type of task.")
        parser.add_argument("--grounding_prompt", type=int, default=None)
        parser.add_argument('--description', action="store_true",
                            help="Prompt the model to generate a video description before performing target tasks.")
        parser.add_argument('--CoT', action="store_true", help="Utilizes Chain-of-Thought Reasoning.")
        parser.add_argument('--fine_tuned', action="store_true")
        parser.add_argument('--iou_thd', type=float, default=0.5)
        parser.add_argument('--no_skip', action="store_true",
                            help="Test the probes even if the initial prediction is in accurate.")
        parser.add_argument('--overwrite', action="store_true")
        parser.add_argument("--video_root", type=str, default="/data/video_datasets/", help="path to video files")
        parser.add_argument("--output_dir", type=str, default=None, help="path to output files")
        parser.add_argument("--exp_id", type=str, default=None, help="ID of this run.")
        parser.add_argument("--seed", type=int, default=1000)
        parser.add_argument('--debug', action="store_true", help="Debug mode.")
        self.parser = parser

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()

        if not opt.video_root:
            opt.video_root = f"/data/video_datasets/{opt.dset_name}"
        else:
            opt.video_root = os.path.join(opt.video_root, opt.dset_name)

        opt.test_path = f"data/{opt.dset_name}_consistency_test.json"

        self.opt = opt
        return opt


def generate_question(task, prompt, query, duration, st=None, ed=None):
    choice = random.choice(["pos", "neg"])
    if st and ed:
        st, ed = min(st, duration), min(ed, duration)

    add_detail = prompt["add_detail"]
    if task in ["grounding"]:
        question = prompt[task].format(event=query)
        add_detail = None

    elif task in ["description"]:
        question = prompt[task]
        add_detail = None

    elif task in ["occurrence"]:
        question = random.choice(prompt[choice]).format(event=query, st=st, ed=ed)

    elif task in ["compositional"]:
        query = query.replace("?", "")
        question = prompt[task].format(question=query, st=st, ed=ed)

    else:
        raise NotImplementedError(f"Not implemented task: {task}")

    return question, add_detail, choice


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.replace("'","").strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_iou(A, B):
    try:
        max0 = max((A[0]), (B[0]))
        min0 = min((A[0]), (B[0]))
        max1 = max((A[1]), (B[1]))
        min1 = min((A[1]), (B[1]))

        return round(max(min1 - max0, 0) / (max1 - min0), 2)

    except:
        return 0


def shifting_video_moment(video_features, org_timestamp, new_timestamp, duration):
    """
    Shifts frames of a video between the original timestamp and new timestamp.

    video_features: The input video features (either list or torch.Tensor)
    org_timestamp: The original start and end time (in seconds) for the part of the video to be moved.
    new_timestamp: The new start and end time (in seconds) where the original frames should be shifted.
    duration: Total duration of the video (in seconds).

    The format of video_features in Video-LLaVA and TimeChat is list, containing tensor features.
    """
    if isinstance(video_features, list):
        # Handle list of frames
        n_frames = len(video_features)
        if not isinstance(video_features[0], torch.Tensor):
            _img_embeds = copy.deepcopy(video_features)
            org_frame = second_to_frame(n_frames, org_timestamp, duration)
            new_frame = second_to_frame(n_frames, new_timestamp, duration)

            # Perform the shift
            _img_embeds[org_frame[0]: org_frame[1] + 1] = video_features[new_frame[0]: new_frame[1] + 1]
            _img_embeds[new_frame[0]: new_frame[1] + 1] = video_features[org_frame[0]: org_frame[1] + 1]
            # print("to", video_features[0].shape, len(video_features[0]))
            return _img_embeds
        else:
            img_embes = video_features[0]
            _img_embeds = img_embes.clone()

            org_frame = second_to_frame(n_frames, org_timestamp, duration)
            new_frame = second_to_frame(n_frames, new_timestamp, duration)

            # Perform the shift
            _img_embeds[org_frame[0]: org_frame[1]+1] = img_embes[new_frame[0]: new_frame[1]+1]
            _img_embeds[new_frame[0]: new_frame[1]+1] = img_embes[org_frame[0]: org_frame[1]+1]
            # print("to", video_features[0].shape, len(video_features[0]))
            return [_img_embeds]

    elif isinstance(video_features, torch.Tensor):
        n_frames = video_features.shape[0]
        org_frame = second_to_frame(n_frames, org_timestamp, duration)
        new_frame = second_to_frame(n_frames, new_timestamp, duration)

        # Calculate the number of frames in each range
        org_length = org_frame[1] - org_frame[0]
        new_length = new_frame[1] - new_frame[0]

        # Find the minimum length to avoid shape mismatch
        min_length = min(org_length, new_length)

        # Extract the original and new frame segments with adjusted lengths
        org_frame_feat = video_features[org_frame[0]: org_frame[0] + min_length, :]
        new_frame_feat = video_features[new_frame[0]: new_frame[0] + min_length, :]

        # Clone the tensor to avoid in-place overwriting
        shifted_video_features = video_features.clone()

        # Perform the swap, making sure both ranges are equal in length
        shifted_video_features[org_frame[0]: org_frame[0] + min_length, :] = new_frame_feat
        shifted_video_features[new_frame[0]: new_frame[0] + min_length, :] = org_frame_feat

        return shifted_video_features


def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()


def display(args):
    opt = args if isinstance(args, dict) else vars(args)
    # opt = vars(args)
    print(dict_to_markdown(opt, max_str_len=120))


def second_to_frame(n_frame, seconds, duration):
    return [int(seconds[0] / duration * n_frame), int(seconds[1] / duration * n_frame)]