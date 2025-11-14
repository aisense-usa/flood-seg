import torch
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, path="best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f" Model improved — saved to {self.path}")
        else:
            self.counter += 1
            print(f" EarlyStopping counter: {self.counter}/{self.patience}")

        return self.counter >= self.patience

# if __name__ == "__main__":
#     early_stopping = EarlyStopping(patience=5, delta=0.01, path="best_model.pth")
#     # Simulate validation losses
#     val_losses = [0.5, 0.4, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42]
#     if early_stopping(val_losses[0], model):
#         print("Stopping early")