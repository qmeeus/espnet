from editdistance import eval as editdistance

"""
This module provides helper functions to compute CER/WER
The problem with the method used by ESPnet in nets.e2e_asr is that
it is valid only when the target are letters, because it is converting 
the output/prediction to a string, then pass the strings to editdistance.eval.

The function editdistance.eval is perfectly able to manage lists as input.
Therefore, the whole dictionary lookup + string manipulation etc. is unnecessary

However, the conversion makes sense for words, as evaldistance uses hashing to 
compare elements. Since lists are not hashable, lists of lists are invalid inputs.
"""

def average_distance(seq_hat, seq_true):
    return editdistance(seq_hat, seq_true) / len(seq_true)


def error_rate(seq_hat, seq_true, reduction="none"):
    """
    Compute the error rate of a batch of predicted sequences vs groundtruth
    The error rate is computed using the editdistance
    :seq_hat: torch.Tensor,np.array: hard predictions, size = (batch_size, olen)
    :seq_true: torch.Tensor,np.array: tokens, size = (batch_size, ylen)
    :reduction: str: one of none, mean, sum, how to reduce the output
    returns: torch.Tensor,np.array: average error rate per sequence
        size = (1,) if reduction != none else (batch_size,)
    """
    assert reduction in ("mean", "sum")
    
    input_type = type(seq_hat)
    if input_type == torch.Tensor:
        # Conversion is necessary -> editdistance doesn't handle tensors
        seq_hat, seq_true = (seq.numpy() for seq in (seq_hat, seq_true))
    
    error_rates = input_type(list(map(average_distance, seq_hat, seq_true)))
    
    if reduction is "none":
        return error_rates

    return getattr(error_rates, reduction)()


cer = error_rate

def wer(seq_hat, seq_true, space_index, ignore_index=-1):
    
    if type(ignore_index) is int:
        ignore_index = [ignore_index]

    assert all(type(id) == int for id in ignore_index)

    def split_seq(seq):
        out = [[]]
        for el in seq:
            if el == space_index:
                out.append([])
            elif el in ignore_index:
                continue
            else:
                out[-1].append(el)
        return list(map("".join, out))

    seq_hat, seq_true = (split_seq(seq) for seq in (seq_hat, seq_true))
    return error_rate(seq_hat, seq_true)
