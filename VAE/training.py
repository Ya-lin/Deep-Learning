

import torch

from tqdm import tqdm
use_tqdm = True
tqdm = tqdm if use_tqdm else lambda x:x


def trainer(model, loader, epochs, optimizer, loss_fn, device):
    history = {"train": [], "test": []}
    best_test_loss = float('inf')
    model.train()
    train_loader, test_loader = loader
    for e in tqdm(range(epochs)):
        train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        history["train"].append(train_loss/len(train_loader))
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_hat = model(x)
                loss = loss_fn(x_hat, x)
                test_loss += loss.item()
        history["test"].append(test_loss/len(test_loader))
        
        if history["test"][-1]<best_test_loss:
            best_test_loss = history["test"][-1]
            torch.save({"epoch": e, "model_dict": model.state_dict(),
                        "loss": best_test_loss}, "best_model.pth")

    return model, history

