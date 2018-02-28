import numpy as np

from ops import *


def prepare_descend_2d(FLAGS, make_target_fn):

    X = np.arange(FLAGS.target_fn_roi_start, FLAGS.target_fn_roi_end, FLAGS.target_fn_roi_step)
    Y = np.arange(FLAGS.target_fn_roi_start, FLAGS.target_fn_roi_end, FLAGS.target_fn_roi_step)
    Z = np.zeros(shape=(len(X), len(Y)))

    x_var = tf.Variable(FLAGS.x_initial, name="x", dtype=tf.float32)
    y_var = tf.Variable(FLAGS.y_initial, name="y", dtype=tf.float32)
    target_descendable_fn_output = make_target_fn(x_var, y_var)
    train_op = prepare_train_op(FLAGS, target_descendable_fn_output)

    x_plh = tf.placeholder(tf.float32, shape=[len(X)])
    y_plh = tf.placeholder(tf.float32)
    target_fn_output = make_target_fn(x_plh, y_plh)

    descent_history_x, descent_history_y = [], []
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        print("Preparing 2D mesh")
        for j, y in enumerate(Y):
            z_string = sess.run(target_fn_output, feed_dict={
                x_plh: X,
                y_plh: y
            })
            for i, z in enumerate(z_string):
                Z[i, j] = z

        print("Descending")
        for step_num in range(FLAGS.steps_to_descend):
            x_val, y_val = sess.run([x_var, y_var])
            if train_op:
                sess.run(train_op)
            descent_history_x.append(x_val)
            descent_history_y.append(y_val)

    return X, Y, Z, descent_history_x, descent_history_y

