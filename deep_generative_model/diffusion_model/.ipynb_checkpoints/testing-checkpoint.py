

from matplotlib import pyplot as plt

def display(images, n=10, size=(20, 3), cmap="gray_r", save_to=None):
    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        image = images[i].permute(1,2,0).numpy()
        plt.imshow(image, cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()


    