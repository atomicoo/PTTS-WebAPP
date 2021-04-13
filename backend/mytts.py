#!/usr/bin/env python
import os.path as osp
import librosa

import torch
from .hparams import HParam
from .transform import StandardNorm, TextProcessor
from .models import MelGenerator, ParallelText2Mel
from .synthesizer import Synthesizer

try:
    from .manager import GPUManager
except ImportError as err:
    print(err); gm = None
else:
    gm = GPUManager()


def select_device(device):
    cpu_request = device.lower() == 'cpu'
    # if device requested other than 'cpu'
    if device and not cpu_request:
        c = 1024 ** 2  # bytes to MB
        x = torch.cuda.get_device_properties(int(device))
        s = f'Using torch {torch.__version__} '
        print("%sCUDA:%s (%s, %dMB)" % (s, device, x.name, x.total_memory / c))
        return torch.device(f'cuda:{device}')
    else:
        print(f'Using torch {torch.__version__} CPU')
        return torch.device('cpu')


class MyTTS:
    def __init__(self, config=None, device=None):
        if torch.cuda.is_available():
            index = device if device else str(0 if gm is None else gm.auto_choice())
        else:
            index = 'cpu'
        self.device = device = select_device(index)

        self.hparams = hparams = HParam(config) \
            if config else HParam(osp.join(osp.dirname(osp.abspath(__file__)), "config", "default.yaml"))

        checkpoint = osp.join(osp.dirname(osp.abspath(__file__)), "pretrained", hparams.parallel.checkpoint)
        vocoder_checkpoint = osp.join(osp.dirname(osp.abspath(__file__)), "pretrained", hparams.vocoder.checkpoint)

        normalizer = StandardNorm(hparams.audio.spec_mean, hparams.audio.spec_std)
        processor = TextProcessor(hparams.text)
        text2mel = ParallelText2Mel(hparams.parallel)
        text2mel.eval()
        vocoder = MelGenerator(hparams.audio.n_mel_channels).to(device)
        vocoder.eval(inference=True)

        self.synthesizer = Synthesizer(
            model=text2mel,
            checkpoint=checkpoint,
            vocoder=vocoder,
            vocoder_checkpoint=vocoder_checkpoint,
            processor=processor,
            normalizer=normalizer,
            device=device
        )

    def __call__(self, texts, speed, volume, tone):
        rate = int(tone) / 3
        alpha = (4 / int(speed)) * rate
        beta = int(volume) / 3
        wave = self.synthesizer.inference(texts, alpha=alpha, beta=beta)
        wave = wave.cpu().detach().numpy()
        sr = self.hparams.audio.sampling_rate
        # use TSM + resample to change tone
        wave = librosa.core.resample(wave, int(sr*rate), sr)
        return wave, sr

