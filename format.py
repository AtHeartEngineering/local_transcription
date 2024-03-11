def format_srt(data):
    srt_format = ''
    for i, segment in enumerate(data, start=1):
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        speaker = segment['speaker']
        text = f"{speaker}: {segment['text']}"
        srt_format += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_format

def format_time(seconds):
    """Convert seconds to SRT time format HH:MM:SS,MS."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(seconds % 1 * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"
