import torch
import torch.optim as optim
from tqdm import tqdm
from utility import DiceBCELoss, dice_score

#  Train & Validate Epoch
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None):
    model.train()
    epoch_loss = 0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for pre_img, post_img, mask, _, _ in loop:  # <-- added mask + coords
        pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(pre_img, post_img)
            loss = loss_fn(outputs, mask)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    epoch_loss = 0
    epoch_dice = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validating", leave=False)
        for pre_img, post_img, mask, _, _ in loop:
            pre_img, post_img, mask = pre_img.to(device), post_img.to(device), mask.to(device)

            outputs = model(pre_img, post_img)
            loss = loss_fn(outputs, mask)
            epoch_loss += loss.item()
            epoch_dice += dice_score(outputs, mask)

    return epoch_loss / len(dataloader), epoch_dice / len(dataloader)


#  Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, path="best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f" Model improved — saved to {self.path}")
        else:
            self.counter += 1
            print(f" EarlyStopping counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience


# Training Loop
def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4, ckpt_path="best_model.pth"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = DiceBCELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    early_stopper = EarlyStopping(patience=3, path=ckpt_path)

    for epoch in range(epochs):
        print(f"\n🌊 Epoch [{epoch+1}/{epochs}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_dice = validate_one_epoch(model, val_loader, loss_fn, device)

        print(f" Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if early_stopper.step(val_loss, model):
            print(" Early stopping triggered.")
            break

    model.load_state_dict(torch.load(ckpt_path))
    return model
