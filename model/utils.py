import tensorflow as tf
from tensorflow.python import debug as tf_debug
ALREADY_INITIALIZED = set()


def get_session(config=None, debug=False):
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.InteractiveSession(config=config)
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064")        # Failed on Windows
    assert sess is not None
    return sess


def initialize():
    new_variables = set(tf.global_variables())-ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def progressbar_print(cur_pos, max_pos, *args):
    e = '\n' if cur_pos >= max_pos else '\r'
    print(*args, end=e)
