# On the Consistency of Video Large Language Models in Temporal Comprehension

[![arXiv](https://img.shields.io/badge/arXiv-2411.12951-b31b1b.svg)](https://arxiv.org/abs/2411.12951)
<a href='https://huggingface.co/collections/mjjung/vtune-6785f253479b8563af533ffa'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-brown'></a>
<a href='https://huggingface.co/collections/mjjung/vtune-6785f253479b8563af533ffa'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a>

## News
- [x] **[2025.01.15]** We are excited to share that our evaluation datasets, Charades-CON and ActivityNet-CON, are now available on Hugging Face! 🎉 Additionally, the training annotations for VTune have also been released. 
- [x] **[2025.01.14]** We have released our four checkpoints using VTune: [VideoLLaMA-7B-Charades-VTune](https://huggingface.co/mjjung/VideoLLaMA-7B-Charades-VTune), [VideoLLaMA-7B-ActvityNet-VTune](https://huggingface.co/mjjung/VideoLLaMA-7B-ActivityNet-VTune), [TimeChat-7B-Charades-VTune](https://huggingface.co/mjjung/TimeChat-7B-Charades-VTune), [TimeChat-7B-ActvityNet-VTune](https://huggingface.co/mjjung/TimeChat-7B-ActivityNet-VTune). Additionally, checkpoints with naive fine-tuning: [VideoLLaMA-7B-Charades-FT](https://huggingface.co/mjjung/VideoLLAMA-7B-Charades-FT), [VideoLLaMA-7B-ActvityNet-FT](https://huggingface.co/mjjung/VideoLLaMA-7B-ActivityNet-FT), [TimeChat-7B-ActivityNet-FT](https://huggingface.co/mjjung/TimeChat-7B-ActivityNet-FT) have been released.
- [x] **[2024.11.20]** Our paper has been released on arXiv.

## Introduction
![image](https://github.com/user-attachments/assets/cc7ba1a6-a7b5-4c87-88b5-471632fabbd1)
- We study the model’s consistency in temporal comprehension by assessing whether its responses align with the initial grounding, using dedicated probes and datasets. We specifically focus on video temporal grounding, where the task involves identifying timestamps in a video that correspond to language queries.

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
