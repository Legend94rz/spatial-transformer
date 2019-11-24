import tensorflow as tf


def get_session(config=None):
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.InteractiveSession(config=config)
    return sess


def progressbar_print(cur_pos, max_pos, *args):
    e = '\r'
    if cur_pos >= max_pos:
        e = '\n'
    print(*args, end=e)
