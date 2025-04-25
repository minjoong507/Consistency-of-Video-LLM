from utils.cons_utils import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import glob
import numpy as np
from easydict import EasyDict as edict
from sklearn.metrics import auc
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    api_key="" # Put your GPT-API Key
)
logger = load_logger("Evaluation")
GPT_MODEL = "gpt-4o-mini"
OPT_FILE_NAME = "opt.json"
PREDICTION_FILE_NAME = "predictions.jsonl"
EVALUATION_FILE_NAME = "eval_results.json"

n_test_set = {"charades": 707,
              "activitynet": 1422}


def _validate_answer(question, answer, expected):
    """
        The implementation to determine whether the answer is correct.
        For efficiency, we first check if the answer contains 'yes' or 'no' keywords.
        If not, we employ GPT to assess the answer.
    """

    # If the answer only contains a single word, either "yes" or "no",
    if len(answer.split()) == 1 and answer.split()[0].lower() in ["yes", "no", "yes.", "no.", "yes,", "no,"]:
        answer = answer.split()[0].lower()
        if expected == "pos":
            if "yes" in answer:
                return 1
            else:
                return 0

        elif expected == "neg":
            if "no" in answer:
                return 1
            else:
                return 0

    else:
        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content":
                        """
                        You are an intelligent evaluator tasked with determining the correctness of an answer in response to a given question. 
                        Based on the expected response type (positive or negative), you need to decide whether the provided answer aligns correctly with both the content and timestamps in the question.
                        
                        INSTRUCTIONS:
                        1. Input Information:
                           - Question: A question about a specific moment or event in the video.
                           - Expected Answer:
                             - "pos" (Positive): The expected answer should agree with the content of the question. It may include phrases like "Yes," "Correct," or similar content that confirms the question's scenario.
                             - "neg" (Negative): The expected answer should disagree with the content of the question. It may include phrases like "No," "Incorrect," or similar content that refutes the question's scenario.
                           - Provided Answer: The actual answer given to the question. You must evaluate whether it is correct based on the question and the expected answer.
                        
                        2. Evaluation Process:
                           - If the expected answer type is "pos," the provided answer should be a positive affirmation of the question.
                           - If the expected answer type is "neg," the provided answer should refute or disagree with the question.
                           
                        3. Output format:
                        The output should be formatted as a dictionary format with the key "result", and the value being either 1(correct) or 0 (incorrect).
                        
                        EXAMPLES:
                        Input:
                        - Question: Does the event 'A young woman is standing and speaking with her friends' happen from 15 to 25 seconds in the video?
                        - Expected Answer: neg
                        - Provided Answer: No, we can not see a young woman standing from 15 to 25 seconds in the video.
                        
                        Output:
                        {
                          "result": 1
                        }
                        
                        Input:
                        - Question: Is the man reading a book under the tree from 0 to 30 seconds in the video?
                        - Expected Answer: pos
                        - Provided Answer: Yes.
                        
                        Output:
                        {
                          "result": 1
                        }
                        
                        Input:
                        - Question: Does the event 'A man play soccer' happens from 30 to 60 seconds in the video?
                        - Expected Answer: pos
                        - Provided Answer: The man is playing soccer from 80 to 90 seconds in the video.
                        
                        Output:
                        {
                          "result": 0
                        }
                    
                        
                        Input:
                        - Question: Does the event 'Person opens the window a few more times.' not happen from 0.0 to 5.0 seconds in the video? Please answer with 'Yes' or 'No'.
                        - Expected Answer: pos
                        - Provided Answer: The event 'Person opens the window a few more times' does not happen from 0.0 to 5.0 seconds.
                        
                        Output:
                        {
                          "result": 1
                        }
                        
                        Now it's your turn.
                        """
                },
                {
                    "role": "user",
                    "content":
                        f"""
                        Input:
                        - Question: {question}
                        - Expected Answer: {expected}
                        - Provided Answer: {answer}
                        
                        PLEASE DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. ONLY PROVIDE A DICTIONARY FORMAT.
                        Output:
                        """
                }
            ]
        )
        response_message = completion.choices[0].message.content
        try:
            response = ast.literal_eval(response_message)['result']
            return response

        except Exception as e:
            logger.warning(f"Error occurs: {e} with message: {response_message}")
            return None


def validate_answer(confusion_matrix, qa_pairs, query_type, key, debug=False):
    """
    confusion_matrix: The confusion matrix to report results for the given question and answer.
    qa_pairs: A list of tuples containing the question and answer pairs.
    expected: The expected answer type.
    key: The key of the task type.

    If the answer does not contain single word, either 'yes' or 'no,' we use GPT4-4.0 Mini to evaluate it.
    The value 1 for the response means the answer is correct. Otherwise, (i.e., response=0) it is incorrect.
    """

    for qa_pair in qa_pairs:
        prompt_choice, question, answer = qa_pair["c"], qa_pair["q"], qa_pair["a"]
        assert prompt_choice in ["pos", "neg"], f"We do not support the prompt template for the {prompt_choice}"
        case = None

        if query_type == "pos":
            expected_answer = "pos" if prompt_choice == "pos" else "neg"
            response = _validate_answer(question=question, answer=answer, expected=expected_answer)

            if response == 1:
                confusion_matrix['TP'] += 1
                case = 'TP'
            elif response == 0:
                confusion_matrix['FN'] += 1
                case = 'FN'
            else:
                logger.warning(f"failed to judge the '{key}' answer '{answer}'")

        else:
            expected_answer = "neg" if prompt_choice == "pos" else "pos"
            response = _validate_answer(question=question, answer=answer, expected=expected_answer)
            if response == 1:
                confusion_matrix['TN'] += 1
                case = 'TN'
            elif response == 0:
                confusion_matrix['FP'] += 1
                case = 'FP'
            else:
                logger.warning(f"failed to judge the '{key}' answer '{answer}'")

        # Update the result
        qa_pair["j"] = case

        if debug:
            print(f"Question: {question}, Answer: {answer}, Expected: {expected_answer} | {case}")

    return confusion_matrix, qa_pairs


# The codes for calculating scores
def calculate_auc_score(cm):
    try:
        TP = cm['TP']
        FP = cm['FP']
        FN = cm['FN']
        TN = cm['TN']
        # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
        tpr = TP / (TP + FN)  # True Positive Rate (Recall)
        fpr = FP / (FP + TN)  # False Positive Rate

        # Create ROC curve points
        tpr_values = [0, tpr, 1]
        fpr_values = [0, fpr, 1]

        # Calculate AUC
        roc_auc = auc(fpr_values, tpr_values)
        return roc_auc
    except:
        return 0


def calculate_r1(iou_values, thresholds=[0.3, 0.5, 0.7], n_preds=None):
    if n_preds is None:
        n_preds = len(iou_values)
    r1_scores = {}
    for threshold in thresholds:
        r1 = sum(1 for iou in iou_values if iou >= threshold) / n_preds if n_preds > 0 else 0
        r1_scores[f'R@1 IoU={threshold}'] = float("{:.2f}".format(r1 * 100))
    miou = round(sum(iou_values) / n_preds * 100, 3) if n_preds > 0 else 0
    r1_scores['mIoU'] = miou

    return r1_scores


def calculate_PRF(confusion_matrix, n_round=True):
    TP = confusion_matrix['TP']
    FP = confusion_matrix['FP']
    FN = confusion_matrix['FN']
    TN = confusion_matrix['TN']

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    scores = edict(Precision=precision,
                   Recall=recall,
                   F1_score=f1_score,
                   ROC_AUC=calculate_auc_score(confusion_matrix),
                   Accuracy=accuracy)

    if n_round:
        for k, v in scores.items():
            scores[k] = float("{:.2f}".format(v * 100))

    return scores


def evaluate_grounding(results, verbos=False):
    if isinstance(results, list):
        pass
    elif isinstance(results, str):
        results = load_jsonl(results)
    else:
        raise TypeError("Input file must be a list or a string")

    gt_moments = [result["meta"]["timestamp"] for result in results]
    try:
        pred_moments = [result["prediction"]["pred_moment"]["t"] for result in results]
    except:
        pred_moments = [result["prediction"]['qa']["t"] for result in results]

    print(f"{len(results)} predictions")

    ious = [get_iou(gt_moment, pred_moment) for gt_moment, pred_moment in zip(gt_moments, pred_moments)]

    grounding_results = calculate_r1(ious)
    if verbos:
        st = ""
        for key, val in grounding_results.items():
            print(f"{key}: {val}")
            st += f"{val} & "
        print(st)

    return calculate_r1(ious)

def evaluate_grounding_probes(results, key="shifted", verbos=False):
    ious =[]
    if key == "shifted":
        ious = [result['shifted']['iou'] for result in results if 'shifted' in result]

    elif key == "rephrased":
        for result in results:
            ious.extend(result['rephrased']['ious'])
    else:
        raise NotImplementedError

    metrics = calculate_r1(ious)

    if verbos:
        print(key)
        for k, v in metrics.items():
            print(f"{k}: {v}")

    return metrics


def evaluate_predictions_for_consistency(results, args, iou_thd=0.5, verbos=False):
    if isinstance(results, list):
        pass
    elif isinstance(results, str):
        results = load_jsonl(results)
    else:
        raise TypeError("Input file must be a list or a string")

    if args.debug:
        logger.info(f"During debug mode, we change the iou threshold {iou_thd} to 0.0 and evaluate a subset of predictions.")
        results = results[:3]
        iou_thd = 0

    performance_report = {
        "holistic": {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
        "compositional": {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    }

    org_ious = [get_iou(result['prediction']['qa']['t'], result['meta']['timestamp']) for result in results]

    if len(results) != n_test_set[args.dset_name]:
        total_videos = list(set([result['meta']['vid'] for result in results]))
        logger.warning(f"Number of predictions does not match the number of test set. "
                       f"Total Videos: {len(total_videos)},"
                       f"but we have {len(results)} predictions.")

    filtered_results = [result for result in results if iou_thd <= result['prediction']['iou']]
    logger.info(f"Applying {iou_thd} iou threshold, the {len(filtered_results)} predictions will be evaluated.")

    performance_report["org_grounding"] = calculate_r1(org_ious)
    performance_report["shifted_grounding"] = evaluate_grounding_probes(filtered_results, key='shifted', verbos=verbos)
    performance_report["rephrased_grounding"] = evaluate_grounding_probes(filtered_results, key='rephrased', verbos=verbos)

    # Evaluate each result and accumulate the scores
    for result in tqdm(filtered_results, desc="Evaluating Consistency..", total=len(filtered_results)):
        for key, data in result.items():
            if key in ['meta', 'prediction', 'rephrased', 'shifted']:
                pass

            elif key in ["holistic", "compositional"]:
                for type, qa_pairs in result[key].items():
                    if type in ["original", "aligned"]:
                        query_type = "pos"
                    elif type in ["misaligned"]:
                        query_type = "neg"
                    else:
                        raise NotImplementedError(f"This code not address the answer for {type} case")
                    performance_report[key], _qa_pairs = validate_answer(performance_report[key], qa_pairs, query_type, key, debug=args.debug)
                    result[key][type] = _qa_pairs
            else:
                raise NotImplementedError(f"This code not address the task {key}")

    for key, val in performance_report.items():
        if key in ["holistic", "compositional"]:
            performance_report[key]['scores'] = calculate_PRF(val)

    if args.output_dir:
        save_jsonl(filtered_results, os.path.join(args.output_dir, "predictions_w_judgement.jsonl"))

    # Absolute scores
    Ground = performance_report['org_grounding']['R@1 IoU=0.5']
    C_Ground = "{:.2f}".format(performance_report['rephrased_grounding']['R@1 IoU=0.5'] * Ground / 100)
    S_Ground = "{:.2f}".format(performance_report['shifted_grounding']['R@1 IoU=0.5'] * Ground / 100)
    H_verif_acc = "{:.2f}".format(performance_report['holistic']['scores']['Accuracy'] * Ground / 100)
    C_verif_acc = "{:.2f}".format(performance_report['compositional']['scores']['Accuracy'] * Ground / 100)

    performance_report["absolute_scores"] = edict(Ground=Ground,
                                                  C_Ground=C_Ground,
                                                  S_Ground=S_Ground,
                                                  H_verif_acc=H_verif_acc,
                                                  C_verif_acc=C_verif_acc)

    return performance_report


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description="Evaluate the Video-LLM's prediction results")
    args.add_argument("--iou_thd", type=float, default=0.5,
                      help="Considering the model's prediction which has a higher IoU than the threshold.")
    args.add_argument("--task", type=str, default="consistency", choices=["grounding", "consistency"],
                      help="Evaluation task")
    args.add_argument("--test_path", type=str, required=True,
                      help="path to the model's predictions")
    args.add_argument('--debug', action="store_true",
                      help="run in the debug mode.")
    args = args.parse_args()

    opt = None
    if ".jsonl" in args.test_path:
        file_path = args.test_path
    else:
        path_to_predictions = f"{args.task}_{PREDICTION_FILE_NAME}"
        file_path = os.path.join(args.test_path, path_to_predictions)
        opt_path = os.path.join(args.test_path, OPT_FILE_NAME)
        args.output_dir = args.test_path
        if os.path.exists(opt_path):
            opt = load_json(opt_path)

    if not os.path.exists(file_path):
        _file_path = glob.glob(os.path.join(args.test_path, f"*{PREDICTION_FILE_NAME}"))[0]
        if os.path.exists(_file_path):
            logger.info(f"Replace the file path {file_path} to {_file_path}")
            file_path = _file_path
        else:
            raise FileNotFoundError(f"File {file_path} does not exist!")

    logger.info(f"Loading predictions from {file_path}")
    if opt:
        logger.info(f"Configurations of {file_path}")
        display(opt)

    args.dset_name = opt["dset_name"]
    logger.info(f"Evaluating predictions for {args.task}")
    display(args)

    if args.task == "grounding":
        result = evaluate_grounding(file_path)
    elif args.task == "consistency":
        result = evaluate_predictions_for_consistency(file_path, iou_thd=args.iou_thd, args=args)
    else:
        raise NotImplementedError(f"This code does not handle the task {args.task}")

    logger.info("Done.")
    for k, v in result.items():
        print(f"{k}: {v}")

    if not args.debug:
        if PREDICTION_FILE_NAME in args.test_path:
            args.test_path = args.test_path.replace(PREDICTION_FILE_NAME, "")
        save_json(result, os.path.join(args.test_path, f"{args.task}_{EVALUATION_FILE_NAME}"), save_pretty=True)