import numpy as np

from ops import *


def prepare_descend_1d_vo_to_2d(FLAGS, make_target_fn):

    X = np.arange(FLAGS.target_fn_roi_start, FLAGS.target_fn_roi_end, FLAGS.target_fn_roi_step)
    Y = np.arange(FLAGS.sigma_step, FLAGS.sigma_roi_end, FLAGS.sigma_step)
    Z = np.zeros(shape=(len(Y), len(X)))

    x_var = tf.Variable(FLAGS.x_initial, name="x_var", dtype=tf.float32)
    y_sigma_var = tf.Variable(FLAGS.sigma_initial, name="y_sigma_var", dtype=tf.float32)
    y_sigma_restore_op = y_sigma_var.assign(FLAGS.sigma_step)
    probes_var = tf.random_normal([FLAGS.probe_num], mean=x_var, stddev=y_sigma_var)
    target_descendable_fn_output = tf.reduce_mean(make_target_fn(probes_var))
    train_op = prepare_train_op(FLAGS, target_descendable_fn_output)

    x_plh = tf.placeholder(tf.float32, shape=[len(X)])
    y_sigma_plh = tf.placeholder(tf.float32)
    probes_line_var = tf.random_normal([FLAGS.probe_num, len(X)], mean=x_plh, stddev=y_sigma_plh)
    target_fn_output = tf.reduce_mean(make_target_fn(probes_line_var), axis=0)

    descent_history_x, descent_history_y = [], []
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        print("Preparing 2D mesh")
        for j, y in enumerate(Y):
            z_string = sess.run(target_fn_output, feed_dict={
                x_plh: X,
                y_sigma_plh: y
            })
            for i, z in enumerate(z_string):
                Z[j, i] = z

        print("Descending")
        for step_num in range(FLAGS.steps_to_descend):
            x_val, y_val = sess.run([x_var, y_sigma_var])
            if train_op:
                sess.run(train_op)
            if y_val < FLAGS.sigma_step:
                sess.run(y_sigma_restore_op)
                y_val = FLAGS.sigma_step
            descent_history_x.append(x_val)
            descent_history_y.append(y_val)

    return X, Y, Z, descent_history_x, descent_history_y
