import numpy as np

from ops import *


def prepare_descend_1d_es(FLAGS, make_target_fn):

    X = np.arange(FLAGS.target_fn_roi_start, FLAGS.target_fn_roi_end, FLAGS.target_fn_roi_step)

    x_var = tf.Variable(FLAGS.x_initial, name="x", dtype=tf.float32)
    probes_var = tf.random_normal([FLAGS.probe_num], mean=x_var, stddev=FLAGS.es_sigma_1d)
    target_fn_output = tf.reduce_mean(make_target_fn(probes_var))
    train_op = prepare_train_op(FLAGS, target_fn_output)

    origin_fn_output = make_target_fn(x_var)

    descent_history_x, descent_history_y = [], []
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        print("Preparing plot")
        sess.run(init_op)
        Y = np.asarray([sess.run(target_fn_output, feed_dict={
                x_var: x
            }) for x in X])
        Y_outline = np.asarray([sess.run(origin_fn_output, feed_dict={
                x_var: x
            }) for x in X])

        print("Descending")
        for step_num in range(FLAGS.steps_to_descend):
            x_val, target_fn_value = sess.run([x_var, target_fn_output])
            if train_op:
                sess.run(train_op)
            descent_history_x.append(x_val)

    for x in descent_history_x:
        itemindex = np.where(X > x)[0]
        if itemindex is not None and len(itemindex):
            if itemindex[0] == 0:
                y = Y[0]
            else:
                y = (Y[itemindex[0]-1] + Y[itemindex[0]])/2.0
        else:
            y = Y[-1]
        descent_history_y.append(y)

    return X, Y, Y_outline, descent_history_x, descent_history_y
