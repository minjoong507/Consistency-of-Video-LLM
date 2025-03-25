import re


def extract_time(paragraph):
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