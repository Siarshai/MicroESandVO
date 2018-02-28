import tensorflow as tf


def prepare_train_op(FLAGS, target_fn_output):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
        grads_and_vars = optimizer.compute_gradients(target_fn_output, tvars)
        # capped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
        return optimizer.apply_gradients(grads_and_vars)


# Honest ES training op
def prepare_random_evolution_train_op_1d(FLAGS, input_variable, make_target_fn, sigma):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        accumulated_grad = tf.Variable(0.0, name="accumulated_grad", dtype=tf.float32)
        probes_var = tf.random_normal([FLAGS.probe_num], mean=0.0, stddev=sigma)
        approx_grad = tf.reduce_mean(probes_var*make_target_fn(input_variable + probes_var))/(sigma**2)
        update_grad_op = accumulated_grad.assign(FLAGS.momentum*accumulated_grad + (1.0 - FLAGS.momentum)*approx_grad)
        train_op = input_variable.assign(input_variable - FLAGS.learning_rate*accumulated_grad)
        return tf.group(update_grad_op, train_op)


def make_convex_1d(input_tensor):
    return tf.square(input_tensor)


def make_wobbly_square_1d(input_tensor):
    return -tf.cos(input_tensor * 6) / (1 + tf.square(input_tensor)) + tf.square(input_tensor) / 6.0


def make_clipped_wobbly_square_1d(input_tensor):
    return tf.clip_by_value(make_wobbly_square_1d(input_tensor), -1.0, 1.0)


def make_clipped_abs_1d(input_tensor):
    return tf.clip_by_value(tf.abs(input_tensor) - 1, 0.0, 1.0)


def make_wall_1d(input_tensor):
    return tf.sigmoid(input_tensor/2.0) + tf.clip_by_value(-tf.abs(input_tensor*15.0) + 1, 0.0, 1.0)


def make_two_canyons_1d(input_tensor):
    return 1 - 0.8*tf.exp(-tf.square(input_tensor + 1)) - tf.exp(-tf.square((input_tensor - 1)*9.0))


def make_convex_2d(input_tensor_x, input_tensor_y):
    r = tf.sqrt(tf.square(input_tensor_x - 1.0) + tf.square(input_tensor_y - 1.0))
    return r


def make_wobbly_square_2d(input_tensor_x, input_tensor_y):
    r = tf.sqrt(tf.square(input_tensor_x) + tf.square(input_tensor_y))
    return -tf.cos(6 * r) / (1 + tf.square(r)) + tf.square(r) / 18.0
