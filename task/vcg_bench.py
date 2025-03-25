import json
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
import time
import os
from utils.cons_utils import (save_jsonl, save_json, load_json, get_iou,
                              BaseOptions, load_logger, display)
import random
import numpy as np

consistency_qa_path = 'data/consistency_qa.json'
generic_qa_path = 'data/generic_qa.json'
temporal_qa_path = 'data/temporal_qa.json'

gt_path = {'consistency': consistency_qa_path,
           'generic': generic_qa_path,
           'temporal': temporal_qa_path
           }

vcg_output = 'vcg_bench_results'
logger = load_logger("[VCG Bench]")


def find_video_path(args, vid):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    for fmt in video_formats:  # Added this line
        video_path = os.path.join(args.video_root, f"{vid}{fmt}")
        # Check if the video exists
        if os.path.exists(video_path):
            return video_path

    return None

def run_vgc_bench_consistency(args, model, gt_contents, type):
    output_list = []  # List to store the output results

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_contents, total=len(gt_contents)):
        video_name = sample['video_name']
        sample_set = sample
        question1 = sample['Q1']
        question2 = sample['Q2']

        # Load the video file
        video_path = find_video_path(args, video_name)
        if os.path.exists(video_path):
            video_features, msg = model.load_video_features(video_path)
            # Run inference on the video and add the output to the list
            output1 = model.naive_qa(video_features=video_features, query=question1, msg=msg)
            sample_set['pred1'] = output1

            output2 = model.naive_qa(video_features=video_features, query=question2, msg=msg)
            sample_set['pred2'] = output2

            output_list.append(sample_set)

            if args.debug:
                print(f"[Question1] {question1}")
                print(f"[Answer1] {output1}\n")
                print(f"[Question2] {question2}")
                print(f"[Answer2] {output2}\n")

    save_json(output_list, os.path.join(args.output_dir, f"{type}_qa.json"))


def _run_vgc_bench(args, model, gt_contents, type):
    output_list = []  # List to store the output results

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_contents, total=len(gt_contents)):
        video_name = sample['video_name']
        sample_set = sample
        question = sample['Q']

        # Load the video file
        video_path = find_video_path(args, video_name)
        if os.path.exists(video_path):
            video_features, msg = model.load_video_features(video_path)
            # Run inference on the video and add the output to the list
            output = model.naive_qa(video_features=video_features, query=question, msg=msg)
            sample_set['pred'] = output
            output_list.append(sample_set)

            if args.debug:
                print(f"[Question] {question}")
                print(f"[Answer] {output}")

    save_json(output_list, os.path.join(args.output_dir, f"{type}_qa.json"))


def run_vgc_bench(args, model):
    logger.info("Start VCG-Bench")
    if args.exp_id is None:
        args.exp_id = args.model_type

    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    if args.debug:
        output_dir = f"debug_{vcg_output}/{args.model_type}/{args.exp_id}_{args.dset_name}_{cur_time}"
    else:
        if args.fine_tuned:
            output_dir = f"{vcg_output}/{args.model_type}/fine_tuned_{args.exp_id}_{cur_time}"
        else:
            output_dir = f"{vcg_output}/{args.model_type}/{args.exp_id}_{cur_time}"

    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    display(args)
    save_json(vars(args), os.path.join(args.output_dir, "opt.json"), save_pretty=True)

    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    for type in ["consistency", "generic", "temporal"]:
        logger.info(f"Run {type} QA")
        gt_contents = load_json(gt_path[type])

        if args.debug:
            logger.info("Debug Mode. We will only use three samples.")
            gt_contents = gt_contents[:3]

        if type in ["generic", "temporal"]:
            _run_vgc_bench(args, model, gt_contents, type)
        else:
            run_vgc_bench_consistency(args, model, gt_contents, type)
