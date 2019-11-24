import tensorflow as tf
from .layers import conv2d, flatten, fully_connected
from .metrics import categorical_accuracy
from .utils import progressbar_print, get_session, initilize as tf_initialize


class STN:
    def __init__(self, input_shape, num_classes, sampling_size=None):
        self.session = get_session()
        if sampling_size is None:
            sampling_size = (input_shape[0], input_shape[1])
        self.input_ph = tf.placeholder(tf.float32, shape=(None, *input_shape))
        self.label_ph = tf.placeholder(tf.float32, shape=(None, num_classes))
        locnet = flatten(self.input_ph)
        locnet = fully_connected(locnet, "fc1", 20, activation=tf.nn.relu)
        affine_mat = fully_connected(locnet, "aff", 6, kernel_initializer=tf.initializers.zeros,
                                     bias_initializer=[1.0, 0, 0, 0, 1, 0], activation=tf.tanh)
        x = self.spatial_trans(self.input_ph, affine_mat, sampling_size, input_shape[-1])
        x = conv2d(x, (3, 3, 1, 32), "conv1", padding='SAME', activation=tf.nn.relu)
        x = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2), padding="SAME")
        x = conv2d(x, (3, 3, 32, 32), "conv2", padding='SAME', activation=tf.nn.relu)
        x = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2), padding="SAME")
        x = flatten(x)
        x = fully_connected(x, "fc2", 256, activation=tf.nn.relu)
        self.logits = fully_connected(x, "fc3", num_classes)
        self.loss = tf.losses.softmax_cross_entropy(self.label_ph, self.logits)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.accuracy = categorical_accuracy(self.label_ph, self.logits)
        self.output = tf.argmax(tf.nn.softmax(self.logits, axis=-1), axis=-1)
        tf_initialize()

    @staticmethod
    def spatial_trans(feature_map, affine_mat, output_size, channels):
        # feature map: [None, H, W, C]
        # affine_mat: [None, 6]
        # output_size: (H, W)
        num_batch, in_h, in_w = tf.shape(feature_map)[0], tf.shape(feature_map)[1], tf.shape(feature_map)[2]
        out_h, out_w = output_size

        # step 1. affine grid
        x = tf.linspace(-1.0, 1.0, out_w)
        y = tf.linspace(-1.0, 1.0, out_h)
        regular_x, regular_y = tf.meshgrid(x, y)
        reg_flatx, reg_flaty = tf.reshape(regular_x, [-1]), tf.reshape(regular_y, [-1])
        regular_grid = tf.stack([reg_flatx, reg_flaty, tf.ones_like(reg_flatx)])  # [3, HW]
        regular_grid = tf.expand_dims(regular_grid, axis=0)  # [1, 3, HW]
        regular_grid = tf.tile(regular_grid, [num_batch, 1, 1])  # [None, 3, HW]

        theta = tf.cast(tf.reshape(affine_mat, [-1, 2, 3]), tf.float32)
        regular_grid = tf.cast(regular_grid, tf.float32)
        sampled_grid = tf.matmul(theta, regular_grid)  # [None, 2, 3] x [None, 3, HW]=>[None, 2, HW]

        # step 2. sampler
        max_x = tf.cast(in_w - 1, 'int32')
        max_y = tf.cast(in_h - 1, 'int32')
        x = 0.5 * (1.0 + sampled_grid[:, 0, :]) * tf.cast(in_w, tf.float32)
        y = 0.5 * (1.0 + sampled_grid[:, 1, :]) * tf.cast(in_h, tf.float32)  # [None, HW], float32, range: [0, H]

        x0 = tf.cast(tf.floor(x), tf.int32)
        y0 = tf.cast(tf.floor(y), tf.int32)
        x0 = tf.clip_by_value(x0, 0, max_x - 1)
        y0 = tf.clip_by_value(y0, 0, max_y - 1)

        x1 = tf.clip_by_value(x0 + 1, 0, max_x)
        y1 = tf.clip_by_value(y0 + 1, 0, max_y)  # [None, HW], int32, range: [0, H]

        batch_idx = tf.reshape(tf.range(0, num_batch), [-1, 1])
        batch_idx = tf.tile(batch_idx, (1, out_h * out_w))  # [None, HW]
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

        return tf.add_n([tf.reshape(w00, [-1, out_h, out_w, 1]) * tf.reshape(y0x0, [-1, out_h, out_w, channels]),
                         tf.reshape(w10, [-1, out_h, out_w, 1]) * tf.reshape(y1x0, [-1, out_h, out_w, channels]),
                         tf.reshape(w01, [-1, out_h, out_w, 1]) * tf.reshape(y0x1, [-1, out_h, out_w, channels]),
                         tf.reshape(w11, [-1, out_h, out_w, 1]) * tf.reshape(y1x1, [-1, out_h, out_w, channels])])

    def train(self, x_train, y_train, x_val, y_val):
        # todo: epochs, batch_size, time, etc...
        epochs = 50
        batch_size = 128
        for e in range(epochs):
            print(f"========epoch: {e}=======")
            s = 0
            while s < len(x_train):
                _, loss, acc = self.session.run([self.train_op, self.loss, self.accuracy],
                                                feed_dict={self.input_ph: x_train[s:s + batch_size],
                                                           self.label_ph: y_train[s:s + batch_size]})
                s += batch_size
                progressbar_print(s, len(x_train), "[Train] loss: %.4f, acc: %.4f" % (loss, acc))
            self.evaluate(x_val, y_val)

    def evaluate(self, x_val, y_val):
        batch_size = 256
        s = 0
        while s < len(x_val):
            loss, acc = self.session.run([self.loss, self.accuracy], feed_dict={self.input_ph: x_val[s:s + batch_size],
                                                                                self.label_ph: y_val[s:s + batch_size]})
            s += batch_size
            progressbar_print(s, len(x_val), "[Valid] loss: %.4f, acc: %.4f" % (loss, acc))


def learn(model, x_train, y_train, x_val, y_val):
    model.train(x_train, y_train, x_val, y_val)
