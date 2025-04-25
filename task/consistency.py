import torch
from tqdm import tqdm
from easydict import EasyDict as edict
import time
import os
import random
import copy
import numpy as np
from utils.cons_utils import (save_jsonl, save_json, load_json, get_iou,
                              BaseOptions, load_logger,
                              display, load_jsonl)
from eval.eval import evaluate_predictions_for_consistency

logger = load_logger("[Consistency Evaluation]")

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
    path_to_predictions = f"{args.task}_{PREDICTION_FILE_NAME}"
    path_to_eval_results = f"{args.task}_{EVALUATION_FILE_NAME}"

    results = load_jsonl(os.path.join(args.output_dir, path_to_predictions)) if args.overwrite else []
    test_data = load_json(args.test_path)

    if len(results) != 0 and args.overwrite:
        _results = copy.deepcopy(results)
        filtered_results = []
        missed_vids = []
        for vid, data in test_data.items():
            _filtered = [_pred for _pred in _results if _pred['meta']['vid'] == vid]
            if len(_filtered) == len(data['sentences']):
                filtered_results.extend(_filtered)
            else:
                missed_vids.append(vid)

        print("=" * 150)
        print(f"{len(filtered_results)} predictions are loaded and {len(missed_vids)} videos missed in {args.output_dir}.")
        print("=" * 150)

        results = filtered_results
        test_data = {key: test_data[key] for key in missed_vids if key in test_data}

    for n_data, (vid, data) in tqdm(enumerate(test_data.items()), total=len(test_data), desc="Evaluating.."):
        duration = data['duration']
        video_path = os.path.join(args.video_root, f"{vid}.mp4")
        if os.path.exists(video_path):
            video_features, msg = model.load_video_features(video_path)
        else:
            print(f"Video {vid} not found")
            continue

        try:
            for i, (query, gt_moment) in enumerate(zip(data['sentences'], data['timestamps'])):
                gt_moment = [min(gt_moment[0], duration), min(gt_moment[1], duration)]
                aligned_sentences = data['consistency_annotations'][i]["A"]
                misaligned_sentences = data['consistency_annotations'][i]["M"]
                compositional_info = data['consistency_annotations'][i]["C"]

                # Comprehension Language in Video
                _chat_state = None
                if args.description:
                    _pred, _chat_state = model.run(task="description",
                                                   video_features=video_features,
                                                   query=query, duration=duration,
                                                   return_chat_state=True, msg=msg)
                    if args.debug:
                        print("Generated Descriptions:", _pred)

                # Grounding
                pred_moment = model.run(task="grounding",
                                        video_features=video_features,
                                        query=query, duration=duration,
                                        chat_state=_chat_state, msg=msg)

                iou_w_gt = get_iou(gt_moment, pred_moment["t"])

                # We will skip the rest probes if the prediction is in accurate (i.e., IoU < 0.5)
                if not args.no_skip and iou_w_gt < args.iou_thd and not args.debug:
                    # logger.info(f"""
                    #     Skip the video {vid} and its query '{query}' since the prediction is inaccurate.
                    #     Ground-truth: {gt_moment}   |
                    #     Prediction: {pred_moment["a"]}   |
                    #     Pred-moment: {pred_moment["t"]}   |
                    #     IoU: {iou_w_gt}             |
                    # """)

                    # Save the results.
                    result = edict(
                        meta=edict(
                            vid=vid,
                            sentence=data['sentences'][i],
                            timestamp=data['timestamps'][i],
                            duration=data['duration']
                        ),
                        prediction=edict(
                            qa=pred_moment,
                            iou=iou_w_gt,
                        ),
                    )
                    results.append(result)
                    continue

                # Grounding with shifted video
                shifted_moment = data['shifted_timestamps'][i]
                shifted_video_path = os.path.join(args.video_root, "shifted_videos", f"{vid}_{i}.mp4")

                if os.path.exists(shifted_video_path):
                    shifted_video_features, shifted_vid_msg = model.load_video_features(shifted_video_path)
                    pred_moment_for_shifted = model.run(task="grounding",
                                                        video_features=shifted_video_features,
                                                        query=query, duration=duration,
                                                        chat_state=_chat_state, msg=shifted_vid_msg)
                else:
                    pred_moment_for_shifted = None
                    logger.warning("Check the shifted video files are available.")

                # Consistent moment grounding
                rephrased_grounding_results = [model.run(task="grounding",
                                                         video_features=video_features,
                                                         query=aligned_sentence,
                                                         duration=duration,
                                                         chat_state=_chat_state,
                                                         msg=msg) for aligned_sentence in aligned_sentences]

                # Self-Answer Verification
                occur_ans = model.run(task="occurrence",
                                      video_features=video_features,
                                      query=query, duration=duration,
                                      st=pred_moment["t"][0],
                                      ed=pred_moment["t"][1],
                                      chat_state=_chat_state, msg=msg)

                occur_ans_aligned = [model.run(task="occurrence",
                                               video_features=video_features,
                                               query=aligned_sent,
                                               duration=duration,
                                               chat_state=_chat_state,
                                               st=pred_moment["t"][0],
                                               ed=pred_moment["t"][1],
                                               msg=msg) for aligned_sent in aligned_sentences]

                occur_ans_misaligned = [model.run(task="occurrence",
                                                  video_features=video_features,
                                                  query=misaligned_sent, duration=duration,
                                                  chat_state=_chat_state,
                                                  st=pred_moment["t"][0],
                                                  ed=pred_moment["t"][1],
                                                  msg=msg) for misaligned_sent in misaligned_sentences]

                # Compositional Understanding
                aigned_comp_q = compositional_info["Y"]
                aligned_comp_qa = [model.run(task="compositional",
                                             video_features=video_features,
                                             query=query, duration=duration,
                                             chat_state=_chat_state,
                                             st=pred_moment["t"][0],
                                             ed=pred_moment["t"][1],
                                             msg=msg) for query in aigned_comp_q]

                mislaigned_comp_q = compositional_info["N"]
                misaligned_comp_qa = [model.run(task="compositional",
                                                video_features=video_features,
                                                query=query, duration=duration,
                                                chat_state=_chat_state,
                                                st=pred_moment["t"][0],
                                                ed=pred_moment["t"][1],
                                                msg=msg) for query in mislaigned_comp_q]

                # Save the results.
                result = edict(
                    meta=edict(
                        vid=vid,
                        sentence=data['sentences'][i],
                        timestamp=data['timestamps'][i],
                        shifted_timestamp=shifted_moment,
                        duration=data['duration']
                    ),
                    prediction=edict(
                        qa=pred_moment,
                        iou=iou_w_gt,
                    ),
                    rephrased=edict(
                        pred_moments=rephrased_grounding_results,
                        ious=[get_iou(pred_moment["t"], _pred["t"]) for _pred in rephrased_grounding_results],
                    ),
                    holistic=edict(
                        original=[occur_ans],
                        aligned=occur_ans_aligned,
                        misaligned=occur_ans_misaligned,
                    ),
                    compositional=edict(
                        aligned=aligned_comp_qa,
                        misaligned=misaligned_comp_qa,
                    ),
                )

                if pred_moment_for_shifted:
                    result.shifted = edict(
                        qa=pred_moment_for_shifted,
                        iou=get_iou(shifted_moment, pred_moment_for_shifted["t"])
                    )

                results.append(result)

            if args.debug and n_data == 1:
                break

        except Exception as e:
            logger.exception(f"{e} with the video {video_path}")

        if n_data % 50 == 0:
            logger.info(f"{len(results)} predictions are saved")
            save_jsonl(results, os.path.join(args.output_dir, path_to_predictions))

    logger.info(f"{len(results)} predictions will be saved at {path_to_predictions}")
    logger.info("============ Save Predictions ============")
    save_jsonl(results, os.path.join(args.output_dir, path_to_predictions))

    # Remove the cuda memory since we don't need the model during scoring its answer.
    del model
    torch.cuda.empty_cache()

    logger.info("============ Save Performance ============")
    consistency_eval_results = evaluate_predictions_for_consistency(results, args=args, iou_thd=args.iou_thd, verbos=True)
    save_json(consistency_eval_results, os.path.join(args.output_dir, path_to_eval_results), save_pretty=True)
    logger.info("============ Done. ============")

def run_consistency(model, args):
    logger.info("Measuring Consistency")
    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    if args.exp_id is None:
        args.exp_id = args.model_type

    if args.overwrite and args.output_dir:
        if not os.path.exists(args.output_dir):
            raise FileNotFoundError(f"Failed to find Output directory {args.output_dir}")
        logger.info(f"Overwriting output dir from {args.output_dir}")
    else:
        if args.debug:
            output_dir = f"debug_results/{args.model_type}/{args.exp_id}_{args.dset_name}_{cur_time}"
        else:
            if args.fine_tuned:
                output_dir = f"results/{args.model_type}/fine_tuned_{args.exp_id}_{args.dset_name}_{cur_time}"
            else:
                output_dir = f"results/{args.model_type}/{args.exp_id}_{args.dset_name}_{cur_time}"

        os.makedirs(output_dir, exist_ok=True)
        args.output_dir = output_dir

    # display and save results
    display(args)
    save_json(vars(args), os.path.join(args.output_dir, OPT_FILE_NAME), save_pretty=True)

    main(args, model)
