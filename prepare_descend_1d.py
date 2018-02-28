import numpy as np

from ops import *


def prepare_descend_1d(FLAGS, make_target_fn):

    x_var = tf.Variable(FLAGS.x_initial, name="x_var", dtype=tf.float32)
    target_fn_output = make_target_fn(x_var)

    with_honest_es_train_op = False
    if with_honest_es_train_op:
        train_op = prepare_random_evolution_train_op_1d(FLAGS, x_var, make_target_fn, FLAGS.es_sigma_1d)
    else:
        train_op = prepare_train_op(FLAGS, target_fn_output)

    descent_history_x, descent_history_y = [], []
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        print("Preparing plot")
        X = np.arange(FLAGS.target_fn_roi_start, FLAGS.target_fn_roi_end, FLAGS.target_fn_roi_step)
        Y = [sess.run(target_fn_output, feed_dict={
                x_var: v
            }) for v in X]
        Y = np.asarray(Y)

        print("Descending")
        for step_num in range(FLAGS.steps_to_descend):
            x_val, target_fn_value = sess.run([x_var, target_fn_output])
            if train_op:
                sess.run(train_op)
            descent_history_x.append(x_val)
            descent_history_y.append(target_fn_value)

    return X, Y, descent_history_x, descent_history_y
