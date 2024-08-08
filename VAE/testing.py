

from matplotlib import pyplot as plt
import torch

@torch.no_grad()
def reconstruction(model, test_loader, num, device):
    X, X_hat, Y = [], [], []
    for k, (x, y) in enumerate(test_loader):
        if k<num:
            X.append(x)
            Y.append(y)
            x_hat = model(x.to(device)).cpu()
            X_hat.append(x_hat)
    X = torch.cat(X, dim=0)
    X_hat = torch.cat(X_hat, dim=0)
    Y = torch.cat(Y, dim=0)
    return X, X_hat, Y


def display(images, n=10, size=(20, 3), cmap="gray_r", save_to=None):
    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()

