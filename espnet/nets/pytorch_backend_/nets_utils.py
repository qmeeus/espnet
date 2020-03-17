import torch


def lengths_to_padding_mask(lengths, batch_first=False):
    """
    convert lengths (a 1-D Long/Int tensor) to 2-D binary tensor

    Args:
        lengths: a (B, )-shaped tensor

    Return:
        max_length: maximum length of B sequences
        padding_mask: a (max_length, B) binary mask, where
        [t, b] = 0 for t < lengths[b] and 1 otherwise

    """

    if not torch.is_tensor(lengths):
        raise TypeError(f"lengths should be a tensor, got {type(lengths)}")

    max_length = torch.max(lengths).item()
    bsz = lengths.size(0)
    padding_mask = torch.arange(
        max_length     # a (T, ) tensor with [0, ..., T-1]
    ).type_as(lengths).view(
        1, max_length  # reshape to (1, T)-shaped tensor
    ).expand(
        bsz, -1         # expand to (B, T)-shaped tensor
    ) >= lengths.view(
        bsz, 1
    ).expand(
        -1, max_length
    )

    if not batch_first:
        padding_mask = padding_mask.t()

    return padding_mask, max_length
