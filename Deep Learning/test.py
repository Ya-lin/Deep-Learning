

import torch
from torch.nn import functional as F

def predictive(pred_y):
    pred = F.softmax(pred_y, dim=1)
    pred = torch.argmax(pred, dim=1)
    return pred.cpu()

@torch.no_grad()
def tester(model, test_loader, device, predict=True):
    labels = []
    preds = []
    for x, y in test_loader:
        labels.append(y)
        x = x.to(device)
        if predict: 
            pred = model.predict(x)
        else:
            pred_y = model(x)
            pred = predictive(pred_y)
        preds.append(pred)
    labels = torch.cat(labels)
    preds = torch.cat(preds)
    return labels, preds

