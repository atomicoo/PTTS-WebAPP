import time

import torch
import torch.nn as nn

from .functional import mask


class Synthesizer:
    def __init__(self, 
                 model=None, checkpoint=None, 
                 vocoder=None, vocoder_checkpoint=None, 
                 processor=None, normalizer=None, 
                 device='cuda'):
        # model
        self.model = model
        self.vocoder = vocoder
        self.processor = processor
        self.normalizer = normalizer

        # device
        self.device = device
        self.model.to(self.device)
        print(f'Model sent to {self.device}')

        # helper vars
        self.checkpoint = None
        self.epoch, self.step = 0, 0
        if checkpoint is not None:
            self.checkpoint = checkpoint
            self.load_checkpoint(checkpoint)

        self.vocoder_checkpoint = None
        if vocoder_checkpoint is not None:
            self.vocoder_checkpoint = vocoder_checkpoint
            self.load_voc_checkpoint(vocoder_checkpoint)

    def to_device(self, device):
        print(f'Sending network to {device}')
        self.device = device
        self.model.to(device)
        self.vocoder.to(device)
        return self

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Loaded checkpoint (e=%d s=%d) finished" % (self.epoch, self.step))

        self.checkpoint = None  # prevent overriding old checkpoint
        return self

    def load_voc_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.vocoder.load_state_dict(checkpoint)
        print("Loaded melgan checkpoint finished")

    def inference(self, texts, alpha=1.0, beta=1.0):
        print('Synthesizing...')
        since = time.time()
        texts, tlens = self.processor(texts)
        texts = torch.from_numpy(texts).long().to(self.device)
        texts = torch.cat((texts, torch.zeros(len(texts), 7).long().to(self.device)), dim=-1)
        tlens = torch.Tensor(tlens).to(self.device)
        with torch.no_grad():
            melspecs, prd_durans = self.model((texts, tlens, None, alpha))
        melspecs = self.normalizer.inverse(melspecs * beta)
        msk = mask(melspecs.shape, prd_durans.sum(dim=-1).long(), dim=1).to(self.device)
        melspecs = melspecs.masked_fill(~msk, -11.5129).permute(0, 2, 1)
        melspecs = torch.cat((melspecs, -11.5129*torch.ones(len(melspecs), melspecs.size(1), 3).to(self.device)), dim=-1)
        print(f"Inference {len(texts)} spectrograms, total elapsed {time.time()-since:.3f}s. Done.")
        waves = self.vocoder(melspecs).squeeze(1)
        print(f"Generate {len(texts)} audios, total elapsed {time.time()-since:.3f}s. Done.")
        return waves
