"""
Trouble shootings:
   $  pip install pydantic==1.10.8

"""

import torch
import random
import numpy as np
from utils.cons_utils import BaseOptions
from task.grounding import run_grounding
from task.consistency import run_consistency

eval_func = {
    "grounding": run_grounding,
    "consistency": run_consistency,
}

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


if __name__ == "__main__":
    """ 
    Usage: python run.py --model_type TimeChat --dset_name charades --task grounding --debug
    """
    base_options = BaseOptions().parse()
    set_seed(base_options.seed)

    if base_options.model_type == "TimeChat":
        from timechat.utils import TimeChat, TimeChat_Options
        args = TimeChat_Options().parse()
        model = TimeChat(args)
        args.ckpt = model.model_config["ckpt"]

    # TODO: Add Custom Models

    else:
        raise NotImplementedError

    eval_func[args.task](model=model, args=args)

