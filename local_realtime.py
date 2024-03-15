import logging
import traceback
import torch
import diart.operators as dops
import rich
import rx.operators as ops
from diart import SpeakerDiarization, PipelineConfig
from diart.sources import MicrophoneAudioSource
from whisper_transcriber import WhisperTranscriber
from utils import concat, colorize_transcription

# Suppress whisper-timestamped warnings for a clean output
logging.getLogger("whisper_timestamped").setLevel(logging.ERROR)

dia = SpeakerDiarization()
source = MicrophoneAudioSource()

asr = WhisperTranscriber(model="medium.en", device="cuda")

transcription_duration = 2
duration = 5
step = 0.5
latency = "min"
tau_activate = 0.5
rho_update = 0.1
delta_new = 0.57

batch_size = int(transcription_duration // step)
source.stream.pipe(
    dops.rearrange_audio_stream(
        duration, step),
    ops.buffer_with_count(count=batch_size),
    ops.map(dia),
    ops.map(concat),
    ops.filter(lambda ann_wav: ann_wav[0].get_timeline().duration() > 0),
    ops.starmap(asr),
    ops.map(colorize_transcription),
).subscribe(on_next=rich.print, on_error=lambda _: traceback.print_exc())

print("Listening...")
source.read()