import torch


def mask(shape, lengths, dim=-1):

    assert dim != 0, 'Masking not available for batch dimension'
    assert len(lengths) == shape[0], 'Lengths must contain as many elements as there are items in the batch'

    lengths = torch.as_tensor(lengths)

    to_expand = [1] * (len(shape)-1)+[-1]
    mask = torch.arange(shape[dim]).expand(to_expand).transpose(dim, -1).expand(shape).to(lengths.device)
    mask = mask < lengths.expand(to_expand).transpose(0, -1)
    return mask


def positional_encoding(channels, length, w=1):
    """The positional encoding from `Attention is all you need` paper

    :param channels: How many channels to use
    :param length: 
    :param w: Scaling factor
    :return:
    """
    enc = torch.FloatTensor(length, channels)
    rows = torch.arange(length, out=torch.FloatTensor())[:, None]
    cols = 2 * torch.arange(channels//2, out=torch.FloatTensor())

    enc[:, 0::2] = torch.sin(w * rows / (10.0**4 ** (cols / channels)))
    enc[:, 1::2] = torch.cos(w * rows / (10.0**4 ** (cols / channels)))
    return enc
