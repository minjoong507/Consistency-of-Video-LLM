import argparse
import os
import random
import json
import numpy as np
import torch
import decord
import time
import subprocess
import re
from utils.cons_utils import BaseOptions, generate_question, load_logger
from utils.prompts import prompt, cot

from timechat.common.config import Config
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, conv_llava_llama_2
from timechat.processors.video_processor import load_video
decord.bridge.set_bridge('torch')
logger = load_logger("TimeChat")


class TimeChat_Options(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--cfg-path", default='timechat/eval_configs/timechat.yaml', help="path to configuration file.")
        self.parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        self.parser.add_argument("--options", nargs="+",
                                 help="override some settings in the used config, "
                                      "the key-value pair in xxx=yyy format will be merged into config file (deprecate), "
                                      "change to --cfg-options instead.",
                                 )


class TimeChat:
    def __init__(self, args):
        """
            Follow the official grounding prompt at https://github.com/RenShuhuai-Andy/TimeChat
        """

        cfg = Config(args)
        if args.fine_tuned:
            prompt["grounding"] = "Localize the visual content described by the given textual query '{event}' in the video, and output the start and end timestamps in seconds."
            cfg.model_cfg.ckpt = cfg.model_cfg.charades_ckpt if args.dset_name == "charades" else cfg.model_cfg.activitynet_ckpt
            if not os.path.exists(cfg.model_cfg.ckpt):
                raise FileNotFoundError(f"Check the checkpoint path: {cfg.model_cfg.ckpt}")
        else:
            prompt["grounding"] = "Please find the visual event described by a sentence in the video, determining its starting and ending times. The format should be: 'The event happens in the start time - end time'. For example, The event 'person turn a light on' happens in the 24.3 - 30.4 seconds. Now I will give you the textual sentence: {event}. Please return its start time and end time."
        logger.info(f"Load checkpoints from '{cfg.model_cfg.ckpt}'")

        self.model_config = cfg.model_cfg
        self.gpu_id = args.gpu_id
        self.model, self.vis_processor = self.load_model(cfg)
        self.chat = Chat(self.model, self.vis_processor, device='cuda:{}'.format(args.gpu_id))
        self.debug = args.debug
        self.n_frames = 96
        self.CoT = args.CoT

    def load_model(self, cfg):
        model_config = cfg.model_cfg
        model_config.device_8bit = self.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(self.gpu_id))
        model.eval()

        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        return model, vis_processor

    def load_video_features(self, video_path):
        video_features = []
        video, msg = load_video(
            video_path=video_path,
            n_frms=self.n_frames,
            height=224,
            width=224,
            sampling="uniform",
            return_msg=True
        )
        video = self.vis_processor.transform(video)
        video = video.unsqueeze(0).to(self.gpu_id)

        if self.model.qformer_text_input:
            # timestamp
            timestamps = msg.split('at')[1].replace('seconds.', '').strip().split(',')  # extract timestamps from msg
            timestamps = [f'This frame is sampled at {t.strip()} second.' for t in timestamps]
            timestamps = self.model.tokenizer(
                timestamps,
                return_tensors="pt",
                padding="longest",
                max_length=32,
                truncation=True,
            )

        if self.model.qformer_text_input:
            image_emb, _ = self.model.encode_videoQformer_visual(video, timestamp=timestamps)
        else:
            image_emb, _ = self.model.encode_videoQformer_visual(video)
        video_features.append(image_emb)

        return video_features, msg

    def inference(self, chat, chat_state, video_features):
        llm_message = chat.answer(conv=chat_state,
                                  img_list=video_features,
                                  num_beams=1,
                                  do_sample=False,
                                  temperature=0.05,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
        chat_state.messages[-1][-1] = llm_message

        return llm_message

    def extract_time(self, paragraph):
        prompt = 'A specific example is : 20.8 - 30.0 seconds'.lower()
        paragraph = paragraph.lower()
        paragraph.replace(prompt, '')
        # Split text into sentences based on common delimiters
        sentences = re.split(r'[!?\n]', paragraph)

        # Keywords that might indicate the presence of time information
        keywords = ["starts", "ends", "happens in", "start time", "end time", "start", "end", "happen"]
        # filter sentences by keywords
        candidates = []
        for sentence in sentences:
            # If sentence contains one of the keywords
            if any(keyword in sentence for keyword in keywords):
                candidates.append(sentence)

        timestamps = []
        # Check for The given query happens in m - n (seconds)
        patterns = [
            r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)"
        ]

        for time_pattern in patterns:
            time_matches = re.findall(time_pattern, paragraph)
            if time_matches:
                timestamps = [[float(start), float(end)] for start, end in time_matches]

        if len(sentences) == 0:
            return []
        # check for other formats e.g.:
        # 1 .Starting time: 0.8 seconds
        # Ending time: 1.1 seconds
        # 2. The start time for this event is 0 seconds, and the end time is 12 seconds.
        if len(timestamps) == 0:
            times = []
            time_regex = re.compile(r'\b(\d+\.\d+\b|\b\d+)\b')  # time formats (e.g., 18, 18.5)
            for sentence in candidates:
                time = re.findall(time_regex, sentence)
                if time:
                    time_in_sec = float(time[0])
                    times.append(time_in_sec)
            times = times[:len(times) // 2 * 2]
            timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
        # Check for  examples like:
        # 3. The event 'person flipped the light switch near the door' starts at 00:00:18 and ends at 00:00:23.
        if len(timestamps) == 0:
            times = []
            time_regex = re.compile(r'\b((\d{1,2}:\d{2}:\d{2}))\b')  # time formats (e.g., 18:00, 00:18:05)
            for sentence in candidates:
                time = re.findall(time_regex, sentence)
                if time:
                    t = time[0]
                else:
                    continue
                # If time is in HH:MM:SS format, convert to seconds
                if t.count(':') == 2:
                    h, m, s = map(int, t.split(':'))
                    time_in_sec = h * 3600 + m * 60 + s
                elif t.count(':') == 1:
                    m, s = map(int, t.split(':'))
                    time_in_sec = m * 60 + s
                times.append(time_in_sec)
            times = times[:len(times) // 2 * 2]
            timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]

        results = []
        for (start, end) in timestamps:
            if end > start:
                results.append([start, end])
            else:
                results.append([end, start])

        if len(results) == 0:
            return [0, 0]
        else:
            return results[0]

    def extract_time2(self, paragraph):
        pattern = r'(?:(?:from\s+|in\s+|between\s+|at\s+|—|to\s+|and\s+|–)\s*)?(\d+(?:\.\d+)?)\s*(?:to|and|–|—)\s*(?:(?:from\s+|in\s+|between\s+|at\s+|—|to\s+|and\s+|–)\s*)?(\d+(?:\.\d+)?)?\s*seconds?'
        match = re.search(pattern, paragraph)

        if match:
            start_timestamp = float(match.group(1))
            end_timestamp = float(match.group(2))

            # If only the start timestamp is found
            if end_timestamp is None:
                return [0, start_timestamp]
            else:
                return [float(start_timestamp), float(end_timestamp)]

        else:
            pattern = r'\d+\.\d+|\d+'

            # Find all matches in the input string
            matches = re.findall(pattern, paragraph)

            # Convert matched strings to float
            float_numbers = [float(match) for match in matches]
            if len(float_numbers) == 0:
                return [0, 0]
            elif len(float_numbers) == 1:
                return [0, float_numbers[0]]
            else:
                return float_numbers[:2]

    def initialize_chat(self, task, msg, add_detail=None):
        chat_state = conv_llava_llama_2.copy()
        chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

        if add_detail:
            chat_state.system += (" " + add_detail)
        if self.CoT:
            chat_state.system = cot[task] if task in cot else chat_state.system

        chat_state.append_message(chat_state.roles[0], "<Video><ImageHere></Video> " + msg)

        return chat_state

    def naive_qa(self, video_features, query, chat_state=None, msg=None):
        if not chat_state:
            chat_state = self.initialize_chat(task="qa", msg=msg)
        else:
            chat_state = chat_state.copy()
        self.chat.ask(query, chat_state)
        answer = self.inference(self.chat, chat_state, video_features)

        return answer

    def run(self, task, video_features, query, duration, chat_state=None, st=None, ed=None, msg=None, return_chat_state=False):
        question, add_detail, choice = generate_question(task, prompt, query, duration, st, ed)
        if not chat_state:
            chat_state = self.initialize_chat(task, msg, add_detail=add_detail)
        else:
            chat_state = chat_state.copy()

        question = " ".join([question, add_detail]) if add_detail else question
        self.chat.ask(question, chat_state)
        answer = self.inference(self.chat, chat_state, video_features)

        if self.debug:
            print(f"[{task}] Question: {question}")
            print(f"Answer: {answer}\n")

        if task in ["grounding"]:
            timestamps = self.extract_time(answer)
            if timestamps == [0,0]:
                timestamps = self.extract_time2(answer)
            return {"q": question, "a": answer, "t": timestamps}

        elif task in ["occurrence", "compositional"]:
            if task == "compositional":
                choice = "pos"
            return {"c": choice, "q": question, "a": answer}

        elif task in ["description"]:
            if return_chat_state:
                return answer, chat_state
            return answer
        else:
            raise NotImplementedError(f"Task {task} is not yet implemented.")
