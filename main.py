from data_manager import ClutteredMNIST
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from model import STN, learn

from PIL import Image


def img2array(data_path, desired_size=None, expand=False, view=False):
    """Loads an RGB image as a 3D or 4D numpy array."""
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype='float32')
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """Converts a numpy array to a PIL img."""
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype('uint8'), 'RGB')


def deg2rad(x):
    """Converts an angle in degrees to radians."""
    return (x * np.pi) / 180



def plot_mnist_sample(mnist_sample):
    mnist_sample = np.squeeze(mnist_sample)
    plt.figure(figsize=(7, 7))
    plt.imshow(mnist_sample, cmap='gray', interpolation='none')
    plt.title('Cluttered MNIST sample', fontsize=20)
    plt.axis('off')
    plt.show()

def line(x1, y1, x2, y2):
    plt.plot([x1, x2], [y1, y2], 'o-')

dataset_path = "./dataset/mnist_cluttered_60x60_6distortions.npz"
batch_size = 256
num_epochs = 30

data_manager = ClutteredMNIST(dataset_path)
train_data, val_data, test_data = data_manager.load()
x_train, y_train = train_data
#plot_mnist_sample(x_train[7])


print(x_train.shape, y_train.shape)
#learn(STN(), x_train, y_train, val_data[0], val_data[1])


DIMS = (600, 600)
data_dir = './img/'

# load 4 cat images
img1 = img2array(data_dir + 'cat1.jpg', DIMS, expand=True)#, view=True)
img2 = img2array(data_dir + 'cat2.jpg', DIMS, expand=True)
img3 = img2array(data_dir + 'cat3.jpg', DIMS, expand=True)
img4 = img2array(data_dir + 'cat4.jpg', DIMS, expand=True)
input_img = np.concatenate([img1, img2, img3, img4], axis=0)
B, H, W, C = input_img.shape
print("Input Img Shape: {}".format(input_img.shape))

degree = 45
theta = np.array([
    [0.5*np.cos(deg2rad(degree)), -np.sin(deg2rad(degree)), -0.5],
    [np.sin(deg2rad(degree)), 0.5*np.cos(deg2rad(degree)), -0.5]
]).reshape(1, -1)
#theta = np.array([[1., 0, 0], [0, 1., 0]]).reshape(1, -1)

import tensorflow as tf

img_ph = tf.placeholder('float32', shape=(None, H, W, C))
theta = tf.constant(theta, dtype='float32')
gx, gy, out = STN.spatial_trans(img_ph, theta, (600, 600), 3)

session = tf.Session()
gx, gy, o = session.run([gx, gy, out], feed_dict={img_ph: input_img})
print(o.shape)

img_id = 1
plt.subplot(2, 1, 1)
plt.imshow(input_img[img_id])
xs = [gx[img_id, 0], gx[img_id, input_img.shape[2] - 1], gx[img_id, -input_img.shape[2]], gx[img_id, -1]]
ys = [gy[img_id, 0], gy[img_id, input_img.shape[1] - 1], gy[img_id, -input_img.shape[1]], gy[img_id, -1]]
print(xs, ys)
#line(xs[0], ys[0], xs[1], ys[1])
#line(xs[1], ys[1], xs[3], ys[3])
#line(xs[3], ys[3], xs[2], ys[2])
#line(xs[2], ys[2], xs[0], ys[0])
plt.scatter(gx, gy, s=2)

plt.subplot(2,1,2)
plt.imshow(o[img_id])
plt.show()
