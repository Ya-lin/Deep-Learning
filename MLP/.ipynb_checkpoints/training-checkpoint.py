

from tqdm import tqdm
use_tqdm = True
tqdm = tqdm if use_tqdm else lambda x:x


def trainer(model, train_loader, epochs, optimizer, loss_fn, device):
    train_loss = []
    model.train()
    for _ in tqdm(range(epochs)):
        running_loss = 0.0
        for x, y in train_loader:
            pred = model(x.to(device))
            loss = loss_fn(pred, y.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss.append(running_loss/len(train_loader))
    
    model.eval()
    
    return model, train_loss

