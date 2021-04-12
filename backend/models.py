import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import mask, positional_encoding
from .transform import Pad


# ===============================================
# Parallel Text2Mel
# ===============================================

def expand_encodings(encodings, durations):
    """Expand phoneme encodings according to corresponding estimated durations

    Durations should be 0-masked, to prevent expanding of padded characters
    :param encodings:
    :param durations: (batch, time)
    :return:
    """
    encodings = [torch.repeat_interleave(e, d, dim=0)
                 for e, d in zip(encodings, durations.long())]

    return encodings


def expand_positional_encodings(durations, channels, repeat=False):
    """Expand positional encoding to align with phoneme durations

    Example:
        If repeat:
        phonemes a, b, c have durations 3,5,4
        The expanded encoding is
          a   a   a   b   b   b   b   b   c   c   c   c
        [e1, e2, e3, e1, e2, e3, e4, e5, e1, e2, e3, e4]

    Use Pad from transforms to get batched tensor.

    :param durations: (batch, time), 0-masked tensor
    :return: positional_encodings as list of tensors, (batch, time)
    """

    durations = durations.long()
    def rng(l): return list(range(l))

    if repeat:
        max_len = torch.max(durations)
        pe = positional_encoding(channels, max_len)
        idx = []
        for d in durations:
            idx.append(list(itertools.chain.from_iterable([rng(dd) for dd in d])))
        return [pe[i] for i in idx]
    else:
        max_len = torch.max(durations.sum(dim=-1))
        pe = positional_encoding(channels, max_len)
        return [pe[:s] for s in durations.sum(dim=-1)]


def round_and_mask(pred_durations, plen):
    pred_durations[pred_durations < 1] = 1  # we do not care about gradient outside training
    pred_durations = mask_durations(pred_durations, plen)  # the durations now expand only phonemes and not padded values
    pred_durations = torch.round(pred_durations)
    return pred_durations


def mask_durations(durations, plen):
    m = mask(durations.shape, plen, dim=-1).to(durations.device).float()
    return durations * m


def expand_enc(encodings, durations, mode=None):
    """Copy each phoneme encoding as many times as the duration predictor predicts"""
    encodings = Pad(0)(expand_encodings(encodings, durations))
    if mode:
        if mode == 'duration':
            encodings += Pad(0)(expand_positional_encodings(durations, encodings.shape[-1])).to(encodings.device)
        elif mode == 'standard':
            encodings += positional_encoding(encodings.shape[-1], encodings.shape[1]).to(encodings.device)
    return encodings


class ZeroTemporalPad(nn.ZeroPad2d):
    """Pad sequences to equal lentgh in the temporal dimension"""
    def __init__(self, kernel_size, dilation, causal=False):
        total_pad = (dilation * (kernel_size - 1))

        if causal:
            super(ZeroTemporalPad, self).__init__((0, 0, total_pad, 0))
        else:
            begin = total_pad // 2
            end = total_pad - begin
            super(ZeroTemporalPad, self).__init__((0, 0, begin, end))


class Conv1d(nn.Conv1d):
    """A wrapper around nn.Conv1d, that works on (batch, time, channels)"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, bias=True, padding=0):
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, dilation=dilation,
                                     groups=groups, bias=bias, padding=padding)

    def forward(self, x):
        return super().forward(x.transpose(2,1)).transpose(2,1)


class FreqNorm(nn.BatchNorm1d):
    """Normalize separately each frequency channel in spectrogram and batch,


    Examples:
        t = torch.arange(2*10*5).reshape(2, 10, 5).float()
        b1 = nn.BatchNorm1d(10, affine=False, momentum=None)
        b2 = (t - t.mean([0,2], keepdim=True))/torch.sqrt(t.var([0,2], unbiased=False, keepdim=True)+1e-05)
        -> b1 and b2 give the same results
        -> BatchNorm1D by default normalizes over channels and batch - not useful for differet length sequences
        If we transpose last two dims, we get normalizaton across batch and time
        -> normalization for each frequency channel over time and batch

        # compare to layer norm:
        Layer_norm: (t - t.mean(-1, keepdim=True))/torch.sqrt(t.var(-1, unbiased=False, keepdim=True)+1e-05)
        -> layer norm normalizes across all frequencies for each timestep independently of batch

        => LayerNorm: Normalize each freq. bin wrt to other freq bins in the same timestep -> time independent, batch independent, freq deendent
        => FreqNorm: Normalize each freq. bin wrt to the same freq bin across time and batch -> time dependent, other freq independent
    """
    def __init__(self, channels, affine=True, track_running_stats=True, momentum=0.1):
        super(FreqNorm, self).__init__(channels, affine=affine, track_running_stats=track_running_stats, momentum=momentum)

    def forward(self, x):
        return super().forward(x.transpose(2,1)).transpose(2,1)


class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, n=2, causal=False, norm=FreqNorm, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()

        self.blocks = [
            nn.Sequential(
                Conv1d(channels, channels, kernel_size, dilation=dilation),
                ZeroTemporalPad(kernel_size, dilation, causal=causal),
                activation(),
                norm(channels),  # Normalize after activation. if we used ReLU, half of our neurons would be dead!
            )
            for i in range(n)
        ]

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return x + self.blocks(x)


class TextEncoder(nn.Module):
    """Encodes input phonemes for the duration predictor and the decoder"""
    def __init__(self, hp):
        super(TextEncoder, self).__init__()
        self.kernel_size = hp.enc_kernel_size
        self.dilations = hp.enc_dilations

        self.prenet = nn.Sequential(
            nn.Embedding(hp.alphabet_size, hp.channels, padding_idx=0),
            Conv1d(hp.channels, hp.channels),
            eval(hp.activation)(),
        )

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hp.channels, self.kernel_size, d, n=2, norm=eval(hp.normalize), activation=eval(hp.activation))
            for d in self.dilations
        ])

        self.post_net1 = nn.Sequential(
            Conv1d(hp.channels, hp.channels),
        )

        self.post_net2 = nn.Sequential(
            eval(hp.activation)(),
            eval(hp.normalize)(hp.channels),
            Conv1d(hp.channels, hp.channels)
        )

    def forward(self, x):
        embedding = self.prenet(x)
        x = self.res_blocks(embedding)
        x = self.post_net1(x) + embedding
        return self.post_net2(x)


class SpecDecoder(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""
    def __init__(self, hp):
        super(SpecDecoder, self).__init__()
        self.kernel_size = hp.dec_kernel_size
        self.dilations = hp.dec_dilations

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hp.channels, self.kernel_size, d, n=2, norm=eval(hp.normalize), activation=eval(hp.activation))
            for d in self.dilations],
        )

        self.post_net1 = nn.Sequential(
            Conv1d(hp.channels, hp.channels),
        )

        self.post_net2 = nn.Sequential(
            ResidualBlock(hp.channels, self.kernel_size, 1, n=2),
            Conv1d(hp.channels, hp.out_channels),
            eval(hp.final_activation)()
        )

    def forward(self, x):
        xx = self.res_blocks(x)
        x = self.post_net1(xx) + x
        return self.post_net2(x)


class DurationPredictor(nn.Module):
    """Predicts phoneme log durations based on the encoder outputs"""
    def __init__(self, hp):
        super(DurationPredictor, self).__init__()

        self.layers = nn.Sequential(
            ResidualBlock(hp.channels, 4, 1, n=1, norm=eval(hp.normalize), activation=nn.ReLU),
            ResidualBlock(hp.channels, 3, 1, n=1, norm=eval(hp.normalize), activation=nn.ReLU),
            ResidualBlock(hp.channels, 1, 1, n=1, norm=eval(hp.normalize), activation=nn.ReLU),
            Conv1d(hp.channels, 1))

    def forward(self, x):
        """Outputs interpreted as log(durations)
        To get actual durations, do exp transformation
        :param x:
        :return:
        """
        return self.layers(x)


class VoiceEncoder(nn.Module):
    """Reference audio encoder"""
    def __init__(self, hp):
        super(VoiceEncoder, self).__init__()

        # Define the network
        self.lstm = nn.LSTM(hp.n_mel_channels, hp.channels, 3, batch_first=True)
        self.linear = nn.Linear(hp.channels, hp.speaker_dim)
        self.relu = nn.ReLU()

    def forward(self, mels):
        # Pass the input through the LSTM layers and retrieve the final hidden state of the last
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
        _, (hidden, _) = self.lstm(mels)
        # Take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)
        return embeds


class Interpolate(nn.Module):
    """Use multihead attention to increase variability in expanded phoneme encodings
    
    Not used in the final model, but used in reported experiments.
    """
    def __init__(self, hp):
        super(Interpolate, self).__init__()

        ch = hp.channels
        self.att = nn.MultiheadAttention(ch, num_heads=4)
        self.norm = FreqNorm(ch)
        self.conv = Conv1d(ch, ch, kernel_size=1)

    def forward(self, x):
        xx = x.permute(1, 0, 2)  # (batch, time, channels) -> (time, batch, channels)
        xx = self.att(xx, xx, xx)[0].permute(1, 0, 2)  # (batch, time, channels)
        xx = self.conv(xx)
        return self.norm(xx) + x


class ParallelText2Mel(nn.Module):
    def __init__(self, hp):
        """Text to melspectrogram network.
        Args:
            hp: hyper parameters
        Input:
            L: (B, N) text inputs
        Outputs:
            Y: (B, T, f) predicted melspectrograms
        """
        super(ParallelText2Mel, self).__init__()
        self.hparams = hp
        self.encoder = TextEncoder(hp)
        self.decoder = SpecDecoder(hp)
        self.duration_predictor = DurationPredictor(hp)

    def forward(self, inputs):
        texts, tlens, durations, alpha = inputs
        alpha = alpha or 1.0

        encodings = self.encoder(texts)  # batch, time, channels
        prd_durans = self.duration_predictor(encodings.detach() if self.hparams.separate_duration_grad 
                                   else encodings)[..., 0]  # batch, time

        # use exp(log(durations)) = durations
        if durations is None:
            prd_durans = (round_and_mask(torch.exp(prd_durans), tlens) * alpha).long()
            encodings = expand_enc(encodings, prd_durans, mode='duration')
        else:
            encodings = expand_enc(encodings, durations, mode='duration')

        melspecs = self.decoder(encodings)
        return melspecs, prd_durans


# ===============================================
# MelGAN Vocoder
# ===============================================

MAX_WAV_VALUE = 32768.0


class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(3)
        ])

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            for i in range(3)
        ])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)


class MelGenerator(nn.Module):
    def __init__(self, mel_channel):
        super(MelGenerator, self).__init__()
        self.mel_channel = mel_channel

        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channel, 512, kernel_size=7, stride=1)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4)),

            ResStack(256),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)),

            ResStack(128),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)),

            ResStack(64),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),

            ResStack(32),

            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=7, stride=1)),
            nn.Tanh(),
        )

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        return self.generator(mel)

    def eval(self, inference=False):
        super(MelGenerator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        hop_length = 256
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)

        audio = self.forward(mel)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()

        return audio
