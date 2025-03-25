neg_occurrence = [
    "Is the event '{event}' absent from {st} to {ed} seconds in the video?",
    "Is the event '{event}' not present from {st} to {ed} seconds in the video?",
    "Does the event '{event}' not happen from {st} to {ed} seconds in the video?",
    "Is the event '{event}' missing from {st} to {ed} seconds in the video?"
]

pos_occurrence = [
    "Is the event '{event}' present from {st} to {ed} seconds in the video?",
    "Is the event '{event}' occurring from {st} to {ed} seconds in the video?",
    "Does the event '{event}' happen from {st} to {ed} seconds in the video?",
    "Is the event '{event}' included from {st} to {ed} seconds in the video?"
]

grounding_prompts = [
    "When does the event '{event}' happen in the video? Please only return its start time and end time.",
    "Please find the visual contents in the video described by a given event, determining its starting and ending times. Now I will give you the event: '{event}'. Please only return its start time and end time.",
    "Please answer when the event '{event}' occurs in the video. The output format should be: start - end seconds’. Please return its start time and end time."
]

# default grounding prompt is third one in the list 'grounding_prompts'.
prompt = {
    "grounding": "Please answer when the event '{event}' occurs in the video. The output format should be: 'start - end seconds'. Please return its start time and end time.",
    "pos": pos_occurrence,
    "neg": neg_occurrence,
    "add_detail": "Please answer with 'Yes' or 'No'.",
    "description": "Please describe the given video in detail.",
    "compositional": "{question} from {st} to {ed} seconds in the video?",
}

cot = {
    "grounding": """Your task is to predict the start and end times of an action or event described by a query sentence based on the visual content of the video. Use Chain-of-Thought reasoning to break down the query, analyze key moments, and accurately identify the time range where the action occurs. 
    ### Chain-of-Thought Reasoning:
    1. **Step 1: Parse the Query**: Break down the query sentence to understand the key action or event that you need to locate.
    2. **Step 2: Analyze the Video Features**: Examine the sequence of video frames to detect patterns that match the key action described in the query.
    3. **Step 3: Identify the Temporal Boundaries**: Use temporal reasoning to find the start and end frames of the action based on the video features.
    4. **Step 4: Predict Start and End Times**: Map the identified frames to timestamps in the video, making sure the start and end times align with the query.
    5. **Step 5: Verify the Answer**: Check if the predicted time range accurately captures the action described in the query.
    """,
    "occurrence": """You are a model designed to predict when specific events occur in a video based on a query sentence. Your task is to verify whether the event described in the query occurs in the given moment of the video.
    ### Chain-of-Thought Reasoning:
    1. **Step 1: Verify the Event in the Predicted Time Range**: Analyze the video features from the predicted start time to the end time. Determine if the event described in the query occurs within this time range.
   - Example: For the query "The person is cooking," check for visual patterns such as a stove or kitchen utensils during the predicted moment.
    2. **Step 2: Answer the Verification Question**: Respond to the question:
   - **"Is the event '{event}' present from {start_time} to {end_time} seconds in the video?"**
   - Example: "Is the event 'The person is cooking' present from 30.0 to 40.0 seconds in the video?"
   - If find the event in the given moment, your answer should be "Yes.", if it does happen in the given moment, your answer should be "No.".
    """,
    "compositional": """You are a model designed to analyze the compositional elements of an event in a video. Your task is to verify whether each compositional element occurs during the given moment in the video based on the specific question you receive. Instead of analyzing the entire event at once, you will answer questions about individual components of the scene.
    ### Chain-of-Thought Reasoning:
    1. **Step 1: Analyze the Video Features for the Specific Element**: Analyze the video features from the start time to the end time. Look for visual evidence of the specific compositional element described in the question.
    2. **Step 2: Answer the Compositional Question**: Respond to the question:
   - Example: "Is there a young girl from 0.0 to 5.0 seconds in the video?"
   - If you find a young girl in the given video moment, your answer should be "Yes.". If it is not present, your answer should be "No.".
    """,
}