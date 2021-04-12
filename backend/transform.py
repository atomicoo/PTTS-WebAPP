import re
import numpy as np
from g2p_en import G2p

import torch.nn as nn
import torch.nn.functional as F
from torch import as_tensor, stack


class Pad:
    """Pad all tensors in first (length) dimension"""

    def __init__(self, pad_value=0, get_lens=False):
        self.pad_value = pad_value
        self.get_lens = get_lens

    def __call__(self, x):
        """Pad each tensor in x to the same length

        Pad tensors in the first dimension and stack them to form a batch

        :param x: list of tensors/lists/arrays
        :returns batch: (len_x, max_len_x, ...)
        """

        if self.get_lens:
            return self.pad_batch(x, self.pad_value), [len(xx) for xx in x]

        return self.pad_batch(x, self.pad_value)

    @staticmethod
    def pad_batch(items, pad_value=0):
        max_len = len(max(items, key=lambda x: len(x)))
        zeros = (2*as_tensor(items[0]).ndim -1) * [pad_value]
        return stack([F.pad(as_tensor(x), pad= zeros + [max_len - len(x)], value=pad_value)
                      for x in items])


class StandardNorm(nn.Module):
    def __init__(self, mean, std):
        super(StandardNorm, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean)/self.std

    def inverse(self, x):
        return x * self.std + self.mean



class TextProcessor:

    g2p = G2p()

    def __init__(self, hparams):
        self.units = self.graphemes = hparams.graphemes
        self.phonemes = hparams.phonemes
        self.phonemize = hparams.use_phonemes
        if self.phonemize:
            self.units = self.phonemes
        self.specials = hparams.specials
        self.punctuations = hparams.punctuations
        self.units = self.specials + self.units + self.punctuations
        self.txt2idx = {txt: idx for idx, txt in enumerate(self.units)}
        self.idx2txt = {idx: txt for idx, txt in enumerate(self.units)}

    def normalize(self, text):
        text = text.lower()
        text = re.sub("[ ]+", " ", text)
        # keep_re = "[^" + str(self.graphemes+self.punctuations) +"]"
        # text = re.sub(keep_re, " ", text)  # remove
        text = [ch if ch in self.graphemes+self.punctuations else ' ' for ch in text]
        text = list(text)
        if self.phonemize:
            text = self.g2p(''.join(text))
        return text

    def __call__(self, texts, max_n=None):
        if not isinstance(texts, (str, list)):
            raise TypeError("Inputs must be str or list(str)")
        if isinstance(texts, str):
            texts = [texts]
        normalized_texts = [self.normalize(line) for line in texts]  # text normalization
        tlens = [len(l) for l in normalized_texts]
        max_n = max_n or max(tlens)
        texts = np.zeros((len(normalized_texts), max_n), np.long)
        for i, text in enumerate(normalized_texts):
            texts[i, :len(text)] = [self.txt2idx.get(ch, 1) for ch in text]
        return texts, tlens
