# On the Consistency of Video Large Language Models in Temporal Comprehension

[![arXiv](https://img.shields.io/badge/arXiv-2411.12951-b31b1b.svg)](https://arxiv.org/abs/2411.12951)
<a href='https://huggingface.co/datasets/mjjung/Consistency-Evaluation-for-Video-LLMs'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
<a href='https://huggingface.co/collections/mjjung/vtune-6785f253479b8563af533ffa'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a>

## News
- [x] **[2025.03.25]** Evaluation Codes have been released.
- [x] **[2025.02.27]** Our paper has been accepted by CVPR 2025! 🎉
- [x] **[2025.01.15]** We are excited to share that our evaluation datasets, Charades-CON and ActivityNet-CON, are now available on Hugging Face! 🎉 Additionally, the training annotations for VTune have also been released. 
- [x] **[2025.01.14]** We have released our four checkpoints using VTune: [VideoLLaMA-7B-Charades-VTune](https://huggingface.co/mjjung/VideoLLaMA-7B-Charades-VTune), [VideoLLaMA-7B-ActvityNet-VTune](https://huggingface.co/mjjung/VideoLLaMA-7B-ActivityNet-VTune), [TimeChat-7B-Charades-VTune](https://huggingface.co/mjjung/TimeChat-7B-Charades-VTune), [TimeChat-7B-ActvityNet-VTune](https://huggingface.co/mjjung/TimeChat-7B-ActivityNet-VTune). Additionally, checkpoints with naive fine-tuning: [VideoLLaMA-7B-Charades-FT](https://huggingface.co/mjjung/VideoLLAMA-7B-Charades-FT), [VideoLLaMA-7B-ActvityNet-FT](https://huggingface.co/mjjung/VideoLLaMA-7B-ActivityNet-FT), [TimeChat-7B-ActivityNet-FT](https://huggingface.co/mjjung/TimeChat-7B-ActivityNet-FT) have been released.
- [x] **[2024.11.20]** Our paper has been released on arXiv.

## Introduction
![image](https://github.com/user-attachments/assets/cc7ba1a6-a7b5-4c87-88b5-471632fabbd1)
- We study the model’s consistency in temporal comprehension by assessing whether its responses align with the initial grounding, using dedicated probes and datasets. We specifically focus on video temporal grounding, where the task involves identifying timestamps in a video that correspond to language queries.

## Download
You can download the complete annotations for consistency evaluation from [Hugging Face](https://huggingface.co/datasets/mjjung/Consistency-Evaluation-for-Video-LLMs). The source videos are available via the following links:

- [Charades-STA](https://prior.allenai.org/projects/charades)
- [ActivityNet-Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/)

## Evaluation
Before starting the evaluation, make sure you have prepared the annotations and videos. You should also check the configuration of the Video-LLMs. Install the necessary dependencies using conda and pip for your model. Additionally, you may run `utils/shift_video.py` with the right paths to prepare shifted videos.
Here, we provide an example with the model `TimeChat`. We will include additional baseline models in the future.

To run the evaluation, use the following command:

```
python run.py --model_type TimeChat --dset_name activitynet --task consistency
````
`dset_name` refers to the test dataset, which can be either `charades` or `activitynet`. `task` refers to the evaluation task: either `consistency` or `grounding`. If set to `grounding`, the evaluation will be performed on the original test set.
You can also use the `--debug` flag before performing the actual evaluation to verify your configuration settings.

Once the evaluation is complete, the performance will be reported in `consistency_eval_results.json`, and you can check the model's output in `consistency_predictions.jsonl`.


## Training
For VTune, please download the training annotations for each dataset from Hugging Face. The hyperparameters should align with those specified in Appendix Table 11. 

For evaluation, please provide the checkpoints for each dataset using the links below:
- [Charades-STA](https://huggingface.co/mjjung/TimeChat-7B-Charades-VTune)
- [ActivityNet-Captions](https://huggingface.co/mjjung/TimeChat-7B-ActivityNet-VTune) 

Then, use the following command:
```
python run.py --model_type TimeChat --dset_name activitynet --fine_tuned --task consistency
```
Including the `fine_tuned` option will automatically switch the checkpoint path `ckpt` to `activitynet_ckpt` in `timechat/eval_configs/timechat.yaml`.

## Citation
If you find our paper useful, please consider citing our paper.
```BibTeX
@article{jung2024consistency,
  title={On the Consistency of Video Large Language Models in Temporal Comprehension},
  author={Jung, Minjoon and Xiao, Junbin and Zhang, Byoung-Tak and Yao, Angela},
  journal={arXiv preprint arXiv:2411.12951},
  year={2024}
}
```
## Acknowledgement
We appreciate for the following awesome Video-LLMs: 
- [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat) 
- [VTimeLLM](https://github.com/huangb23/VTimeLLM)
- [VTG-LLM](https://github.com/gyxxyg/VTG-LLM)
- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)
- [Video-LLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)
- [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)
