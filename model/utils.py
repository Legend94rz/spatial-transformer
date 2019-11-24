import tensorflow as tf
ALREADY_INITIALIZED = set()


def get_session(config=None):
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.InteractiveSession(config=config)  # will set this session be the default session
    return sess


def initilize():
    new_variables = set(tf.global_variables())-ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def progressbar_print(cur_pos, max_pos, *args):
    e = '\r'
    if cur_pos >= max_pos:
        e = '\n'
    print(*args, end=e)
