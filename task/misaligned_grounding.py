import torch
from tqdm import tqdm
from easydict import EasyDict as edict
import time
import os
import random
import numpy as np
from utils.cons_utils import (save_jsonl, save_json, load_json, get_iou,
                              BaseOptions, load_logger, display)
from eval.eval import calculate_r1
logger = load_logger("[Misaligned Grounding Evaluation]")

OPT_FILE_NAME = "opt.json"
PREDICTION_FILE_NAME = "predictions.jsonl"
EVALUATION_FILE_NAME = "eval_results.json"


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    print("Set seed ", seed)


def main(args, model):
    results = []

    test_data = load_json(args.test_path)
    target_vid_list = [file.split(".")[0] for file in os.listdir(args.video_root)]
    target_vid_list = [vid for vid in target_vid_list if vid in list(test_data.keys())]
    print(f"Total {len(target_vid_list)} videos in {args.video_root}")

    path_to_predictions = f"{args.task}_{PREDICTION_FILE_NAME}"
    path_to_eval_results = f"{args.task}_{EVALUATION_FILE_NAME}"
    total_ious = []

    for n_data, (vid, data) in tqdm(enumerate(test_data.items()), total=len(target_vid_list), desc="Evaluating.."):
        duration = data['duration']
        video_path = os.path.join(args.video_root, f"{vid}.mp4")

        # Load video frame features
        if os.path.exists(video_path):
            video_features, msg = model.load_video_features(video_path)
        else:
            print(f"Video {vid} not found")
            continue

        for i, (query, gt_moment) in enumerate(zip(data['sentences'], data['timestamps'])):
            misaligned_queries = data['consistency_annotations'][i]["M"]

            # Consistent moment grounding
            misaligned_grounding_results = [model.run(task="grounding",
                                                      video_features=video_features,
                                                      query=misaligned_query,
                                                      duration=duration,
                                                      msg=msg) for misaligned_query in misaligned_queries]

            # Save the results.
            result = edict(
                meta=edict(
                    vid=vid,
                    sentence=data['sentences'][i],
                    timestamp=data['timestamps'][i],
                    duration=data['duration']
                ),
                misaligned=edict(
                    pred_moments=misaligned_grounding_results,
                    ious=[get_iou(gt_moment, _pred["t"]) for _pred in misaligned_grounding_results],
                ),
            )
            total_ious.extend(result["misaligned"]["ious"])
            results.append(result)

        if args.debug and n_data == 1:
            break

        if n_data % 50 == 0:
            logger.info(f"{len(results)} results are saved")

            save_jsonl(results, os.path.join(args.output_dir, f"{path_to_predictions}"))

    logger.info(f"{len(results)} predictions will be saved at {path_to_predictions}")
    save_jsonl(results, os.path.join(args.output_dir, path_to_predictions))

    logger.info("============ Save Performance ============")
    grounding_results = calculate_r1(total_ious)
    print(grounding_results)
    save_json(grounding_results, os.path.join(args.output_dir, path_to_eval_results), save_pretty=True)
    logger.info("Done.")


def run_misaligned_grounding(model, args):
    logger.info("Run Misaligned Grounding")
    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    if args.exp_id is None:
        args.exp_id = args.model_type

    if args.debug:
        output_dir = f"debug_results/{args.model_type}/{args.exp_id}_{args.dset_name}_{cur_time}"
    else:
        if args.fine_tuned:
            output_dir = f"misaligned_grounding_results/{args.model_type}/fine_tuned_{args.exp_id}_{args.dset_name}_{cur_time}"
        else:
            output_dir = f"misaligned_grounding_results/{args.model_type}/{args.exp_id}_{args.dset_name}_{cur_time}"

    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    # display and save results
    display(args)
    save_json(vars(args), os.path.join(args.output_dir, OPT_FILE_NAME), save_pretty=True)

    main(args, model)

