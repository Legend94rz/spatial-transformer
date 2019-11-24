import tensorflow as tf

def get_session(config=None):
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.InteractiveSession(config=config)
    return sess


def processbar_print(cur_pos, max_pos, *args):
    E = '\r'
    if cur_pos>=max_pos:
        E = '\n'
    print(*args, end=E)