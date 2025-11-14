from align_and_crop import align_and_crop_to_overlap
from model import SiameseUNetPP
from train_val import train_model
from dataloader import get_flood_dataloaders
import torch

align_and_crop_to_overlap(
    "./data/Hanumannagar_Preflood_Orthomosaic.tif",
    "./data/Hanumannagar_Postflood_Orthomosaic.tif",
    "./aligned_data/pre_aligned.tif",
    "./aligned_data/post_aligned.tif",
    block_size=1024  # tune for memory vs speed
)

train_loader, val_loader = get_flood_dataloaders(
    pre_path="./aligned_data/pre_aligned.tif",
    post_path="./aligned_data/post_aligned.tif",
    tile_size=512,
)


model = SiameseUNetPP(n_channels=3, n_classes=1, base_ch=32)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


train_model(model,train_loader, val_loader, device, 5)