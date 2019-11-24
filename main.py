from data_manager import ClutteredMNIST
from model import STN, learn


dataset_path = "./dataset/mnist_cluttered_60x60_6distortions.npz"
batch_size = 256
num_epochs = 30

data_manager = ClutteredMNIST(dataset_path)
train_data, val_data, test_data = data_manager.load()
x_train, y_train = train_data

print(x_train.shape, y_train.shape)
learn(STN(input_shape=(60, 60, 1), num_classes=10), x_train, y_train, val_data[0], val_data[1])
