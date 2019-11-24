import tensorflow as tf
from .layers import conv2d, flatten, fully_connected
from .metrics import categorical_accuracy
from .utils import processbar_print, get_session
import cv2
import matplotlib.pyplot as plt


# todo: 现在这个写法不能重入
class STN:
    def __init__(self, input_shape=(60, 60, 1), sampling_size=None, num_classes=10):
        # todo: support multiple type transform
        if sampling_size is None:
            sampling_size = (input_shape[0], input_shape[1])
        self.input_ph = tf.placeholder(tf.float32, shape=(None, *input_shape))
        self.label_ph = tf.placeholder(tf.float32, shape=(None, num_classes))
        locnet = tf.nn.max_pool2d(self.input_ph, ksize=(2, 2), strides=(2, 2), padding='SAME')
        locnet = conv2d(locnet, (5, 5, 1, 20), "locnet/conv1", padding='SAME')
        locnet = tf.nn.max_pool2d(locnet, ksize=(2, 2), strides=(2, 2), padding='SAME')
        locnet = conv2d(locnet, (5, 5, 20, 20), "locnet/conv2", padding='SAME')
        locnet = flatten(locnet)
        locnet = fully_connected(locnet, "locnet/fc", 50)
        locnet = tf.nn.relu(locnet)
        self.aff = affine_mat = fully_connected(locnet, "locnet/aff", 6, kernel_initializer=tf.initializers.zeros,
                                                bias_initializer=[1.0, 0, 0, 0, 1, 0])
        self.gx, self.gy, x = self.spatial_trans(self.input_ph, affine_mat, sampling_size, input_shape[-1])
        x = conv2d(x, (3, 3, 1, 32), "aff/conv1", padding='SAME')
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2), padding="SAME")
        x = conv2d(x, (3, 3, 32, 32), "aff/conv2", padding='SAME')
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2), padding="SAME")
        x = flatten(x)
        x = fully_connected(x, "aff/fc", 256)
        x = tf.nn.relu(x)
        self.logits = fully_connected(x, "fc", num_classes)
        self.loss = tf.losses.softmax_cross_entropy(self.label_ph, self.logits)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.accuracy = categorical_accuracy(self.label_ph, self.logits)
        self.output = tf.argmax(tf.nn.softmax(self.logits, axis=-1), axis=-1)

        # todo: exclude already initialized params
        self.session = get_session()
        self.session.run(tf.global_variables_initializer())
        #self.session.run(tf.local_variables_initializer())

    @staticmethod
    def spatial_trans(feature_map, affine_mat, output_size, channels):
        # feature map: [None, H, W, C]
        # affine_mat: [None, 6]
        # output_size: (H, W)
        num_batch, in_H, in_W = tf.shape(feature_map)[0], tf.shape(feature_map)[1], tf.shape(feature_map)[2]
        out_H, out_W = output_size

        # step 1. affine grid
        x = tf.linspace(-1.0, 1.0, out_W)
        y = tf.linspace(-1.0, 1.0, out_H)
        regular_x, regular_y = tf.meshgrid(x, y)
        reg_flatx, reg_flaty = tf.reshape(regular_x, [-1]), tf.reshape(regular_y, [-1])
        regular_grid = tf.stack([reg_flatx, reg_flaty, tf.ones_like(reg_flatx)])  # [3, HW]
        regular_grid = tf.expand_dims(regular_grid, axis=0)  # [1, 3, HW]
        regular_grid = tf.tile(regular_grid, [num_batch, 1, 1])  # [None, 3, HW]

        theta = tf.cast(tf.reshape(affine_mat, [-1, 2, 3]), tf.float32)
        regular_grid = tf.cast(regular_grid, tf.float32)
        sampled_grid = tf.matmul(theta, regular_grid)  # [None, 2, 3] x [None, 3, HW]=>[None, 2, HW]

        # step 2. sampler
        max_x = tf.cast(in_W - 1, 'int32')
        max_y = tf.cast(in_H - 1, 'int32')
        x = 0.5 * (1.0 + sampled_grid[:, 0, :]) * tf.cast(in_W, tf.float32)
        y = 0.5 * (1.0 + sampled_grid[:, 1, :]) * tf.cast(in_H, tf.float32)  # [None, HW], float32, range: [0, H]

        x0 = tf.cast(tf.floor(x), tf.int32)
        y0 = tf.cast(tf.floor(y), tf.int32)
        x0 = tf.clip_by_value(x0, 0, max_x-1)
        y0 = tf.clip_by_value(y0, 0, max_y-1)

        x1 = tf.clip_by_value(x0+1, 0, max_x)
        y1 = tf.clip_by_value(y0+1, 0, max_y)  # [None, HW], int32, range: [0, H]


        batch_idx = tf.reshape(tf.range(0, num_batch), [-1, 1])
        batch_idx = tf.tile(batch_idx, (1, out_H * out_W))  # [None, HW]
        #y0x0 = tf.gather_nd(feature_map, tf.stack([tf.reshape(batch_idx, [-1]), tf.reshape(y0, [-1]), tf.reshape(x0, [-1])], axis=-1))
        #y1x0 = tf.gather_nd(feature_map, tf.stack([tf.reshape(batch_idx, [-1]), tf.reshape(y1, [-1]), tf.reshape(x0, [-1])], axis=-1))
        #y0x1 = tf.gather_nd(feature_map, tf.stack([tf.reshape(batch_idx, [-1]), tf.reshape(y0, [-1]), tf.reshape(x1, [-1])], axis=-1))
        #y1x1 = tf.gather_nd(feature_map, tf.stack([tf.reshape(batch_idx, [-1]), tf.reshape(y1, [-1]), tf.reshape(x1, [-1])], axis=-1))
        y0x0 = tf.gather_nd(feature_map, tf.stack([batch_idx, y0, x0], axis=-1))
        y1x0 = tf.gather_nd(feature_map, tf.stack([batch_idx, y1, x0], axis=-1))
        y0x1 = tf.gather_nd(feature_map, tf.stack([batch_idx, y0, x1], axis=-1))
        y1x1 = tf.gather_nd(feature_map, tf.stack([batch_idx, y1, x1], axis=-1))  # [None*HW, C]

        x0, x1 = tf.cast(x0, tf.float32), tf.cast(x1, tf.float32)
        y0, y1 = tf.cast(y0, tf.float32), tf.cast(y1, tf.float32)

        w00 = (x1 - x) * (y1 - y)
        w10 = (x1 - x) * (y - y0)
        w01 = (x - x0) * (y1 - y)
        w11 = (x - x0) * (y - y0)  # [None, HW]

        # tf.reshape(y0x0, [-1, out_H, out_W, channels])
        return x0, y0, \
               tf.add_n([tf.reshape(w00, [-1, out_H, out_W, 1]) * tf.reshape(y0x0, [-1, out_H, out_W, channels]),
                         tf.reshape(w10, [-1, out_H, out_W, 1]) * tf.reshape(y1x0, [-1, out_H, out_W, channels]),
                         tf.reshape(w01, [-1, out_H, out_W, 1]) * tf.reshape(y0x1, [-1, out_H, out_W, channels]),
                         tf.reshape(w11, [-1, out_H, out_W, 1]) * tf.reshape(y1x1, [-1, out_H, out_W, channels])])

    def train(self, x_train, y_train, x_val, y_val):
        # todo: epochs, batch_size, time, etc...
        epochs = 10
        batch_size = 128
        #plt.ion()
        for e in range(epochs):
            print(f"========epoch: {e}=======")
            s = 0
            while s < len(x_train):
                _, loss, acc = self.session.run([self.train_op, self.loss, self.accuracy],
                                                feed_dict={self.input_ph: x_train[s:s + batch_size],
                                                           self.label_ph: y_train[s:s + batch_size]})
                s += batch_size
                processbar_print(s, len(x_train), "[Train] loss: %.4f, acc: %.4f" % (loss, acc))

                aff, gx, gy = self.session.run([self.aff, self.gx, self.gy],
                                               feed_dict={self.input_ph: x_val[:20], self.label_ph: y_val[:20]})
                plt.figure(figsize=(4, 4), dpi=300)
                for i in range(20):
                    plt.subplot(4, 5, i+1)

                    xs = [gx[i, 0], gx[i, x_val.shape[2] - 1], gx[i, -x_val.shape[2]], gx[i, -1]]
                    ys = [gy[i, 0], gy[i, x_val.shape[1] - 1], gy[i, -x_val.shape[1]], gy[i, -1]]
                    plt.imshow(x_val[i].squeeze(-1))
                    plt.scatter(xs, ys, c='red', s=8)
                plt.show()
                #plt.pause(0.01)

            self.evaluate(x_val, y_val)

    def evaluate(self, x_val, y_val):
        batch_size = 256
        s = 0
        while s < len(x_val):
            loss, acc, aff = self.session.run([self.loss, self.accuracy, self.aff],
                                              feed_dict={self.input_ph: x_val[s:s + batch_size],
                                                         self.label_ph: y_val[s:s + batch_size]})
            s += batch_size
            processbar_print(s, len(x_val), "[Valid] loss: %.4f, acc: %.4f" % (loss, acc))


def learn(model, x_train, y_train, x_val, y_val):
    model.train(x_train, y_train, x_val, y_val)
    #model.evaluate(x_val, y_val)
