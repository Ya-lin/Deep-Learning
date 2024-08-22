

import torch
from types import SimpleNamespace
from tqdm import tqdm
use_tqdm = True
tqdm = tqdm if use_tqdm else lambda x:x


def trainer(model, loader, epochs, optimizer):
    history = SimpleNamespace(train=[], test=[])
    best_test_loss = float('inf')
    model.train()
    for e in tqdm(range(epochs)):
        train_loss = 0.0
        for x, _ in loader.train:
            loss = model.total_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        history.train.append(train_loss/len(loader.train))
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, _ in loader.test:
                loss = model.total_loss(x)
                test_loss += loss.item()
        history.test.append(test_loss/len(loader.test))
        
        if history.test[-1]<best_test_loss:
            best_test_loss = history.test[-1]
            torch.save({"epoch": e, "model_dict": model.state_dict(),
                        "loss": best_test_loss}, "best_model.pth")

    return model, history

