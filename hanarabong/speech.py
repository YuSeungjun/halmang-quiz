"""Sentence speech from the selected AI Hub + JSS checkpoint and voice 03."""
import io
import os
import re
import textwrap
from functools import lru_cache

from catalog import AUDIO, CHECKPOINT, REFERENCE, RUN, audio_key, read_json


@lru_cache(maxsize=1)
def speaker():
    import torch
    from voxcpm import VoxCPM

    torch.set_num_threads(6)
    model = VoxCPM.from_pretrained(str(RUN / 'checkpoints' / CHECKPOINT), load_denoiser=False,
                                   optimize=False, device=os.getenv('HANARABONG_TTS_DEVICE', 'cuda:0'))
    return model, read_json(REFERENCE)['audio']


@lru_cache(maxsize=8)
def render_wav(text):
    import numpy as np
    import soundfile as sf

    model, reference = speaker()
    sr = model.tts_model.sample_rate
    pieces = []
    for sentence in re.split(r'(?<=[.!?。！？])\s+|\n+', text):
        for chunk in textwrap.wrap(sentence, width=64, break_on_hyphens=False):
            samples = model.generate(text=chunk, reference_wav_path=reference,
                                     normalize=False, denoise=False, max_len=512,
                                     inference_timesteps=10, cfg_value=2.0, retry_badcase=False,
                                     seed=int(audio_key(chunk)[:7], 16))
            if not len(samples) or not np.isfinite(samples).all() or np.max(np.abs(samples)) < .005:
                raise RuntimeError('합성 음성을 만들지 못했어요.')
            if pieces:
                pieces.append(np.zeros(sr // 6, dtype=np.float32))
            pieces.append(samples)
    output = io.BytesIO()
    sf.write(output, np.concatenate(pieces), sr, format='WAV', subtype='PCM_16')
    return output.getvalue()


def sentence_audio(text):
    recorded = AUDIO / f'{audio_key(text)}.wav'
    if recorded.is_file():
        return recorded.read_bytes()
    return render_wav(text)
