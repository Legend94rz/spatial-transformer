from data_manager import ClutteredMNIST
import matplotlib.pyplot as plt
import numpy as np
from model import STN, learn


def plot_mnist_sample(mnist_sample):
    mnist_sample = np.squeeze(mnist_sample)
    plt.figure(figsize=(7, 7))
    plt.imshow(mnist_sample, cmap='gray', interpolation='none')
    plt.title('Cluttered MNIST sample', fontsize=20)
    plt.axis('off')
    plt.show()


dataset_path = "./dataset/mnist_cluttered_60x60_6distortions.npz"
batch_size = 256
num_epochs = 30

data_manager = ClutteredMNIST(dataset_path)
train_data, val_data, test_data = data_manager.load()
x_train, y_train = train_data

print(x_train.shape, y_train.shape)
learn(STN(input_shape=(60, 60, 1), num_classes=10), x_train, y_train, val_data[0], val_data[1])
