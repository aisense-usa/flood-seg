import torch
import torch.nn as nn

def dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice.item()

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        preds = preds.contiguous()
        targets = targets.contiguous()

        bce_loss = self.bce(preds, targets)

        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (
            preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.smooth
        )
        dice_loss = 1 - dice.mean()

        return bce_loss + dice_loss


# split train test 

def split_flood_dataset(dataset, val_fraction=0.2):
    
    tiles_x, tiles_y = dataset.tiles_x, dataset.tiles_y
    
    # Number of rows for validation
    val_rows = int(tiles_y * val_fraction)
    
    # Assign tiles: last val_rows for validation, remaining for training
    train_indices = []
    val_indices = []
    
    for y in range(tiles_y):
        for x in range(tiles_x):
            idx = y * tiles_x + x
            if y >= tiles_y - val_rows:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
    
    return train_indices, val_indices

