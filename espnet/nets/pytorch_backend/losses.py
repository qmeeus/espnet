import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class MaskedMSELoss(nn.Module):
    
    def __init__(self, reduction='sum'):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, target_lengths):
        target_mask = make_pad_mask(target_lengths).type_as(target).bool()
        mse = F.mse_loss(pred, target, reduction='none').mean(-1)
        mse.masked_fill_(target_mask, 0)
        sequence_loss = mse.sum(-1)
        if self.reduction == 'sum':
            return sequence_loss.sum()
        elif self.reduction == 'mean':
            return (sequence_loss / target_lengths).mean()
        else:
            return sequence_loss


if __name__ == "__main__":
    import numpy as np

    lengths = torch.tensor([10,8,5,2])
    pred = torch.rand(4, 10, 5).masked_fill_(make_pad_mask(lengths).unsqueeze(-1), 0)
    target = torch.rand(4, 10, 5)
    masked_mse = MaskedMSELoss()

    mse = nn.MSELoss()(pred, target).detach().numpy()
    mmse = masked_mse(pred, target, lengths).detach().numpy()

    pred, target, lengths = (t.numpy() for t in (pred, target, lengths))
    mmse_np = np.mean([
        ((pred[i,:l] - target[i,:l]) ** 2).mean(-1).sum() / l 
        for i, l in enumerate(lengths)
    ])

    print(f"MSE: {mse:.4f}")
    print(f"Masked MSE: {mmse:.4f}")
    print(f"Masked MSE (numpy): {mmse_np:.4f}")

    assert mse >= mmse
    assert np.isclose(mmse, mmse_np, 1e-9)
