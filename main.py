from animate import animate_1d, animate_2d
from prepare_descend_1d_es import *
from prepare_descend_1d_vo_to_2d import *
from prepare_descend_1d import *
from prepare_descend_2d import *

Flags = tf.app.flags

Flags.DEFINE_string('plot_type', "1d_es", 'What simulation to run')
Flags.DEFINE_boolean('save_animation', True, 'Whether save mp4 file or just display it on screen')

Flags.DEFINE_integer('steps_to_descend', 150, 'Number of steps in animation')
Flags.DEFINE_float('learning_rate', 0.035, 'Learning rate of gradient descend')
Flags.DEFINE_float('momentum', 0.9, 'momentum of gradient descend')

Flags.DEFINE_float('target_fn_roi_start', -4.0, 'Spatial ROI start')
Flags.DEFINE_float('target_fn_roi_end', 4.0, 'Spatial ROI end')
Flags.DEFINE_float('sigma_roi_end', 3.0, 'Sigma ROI end (for VO)')
Flags.DEFINE_float('x_initial', 3.0, 'Starting x coordinate')
Flags.DEFINE_float('y_initial', 3.0, 'Starting y coordinate (for simple 2D optimization)')
Flags.DEFINE_float('sigma_initial', 1.9, 'Starting coordinate in sigma space (for VO)')
Flags.DEFINE_float('target_fn_roi_step', 0.025, 'Spatial grid step')
Flags.DEFINE_float('sigma_step', 0.005, 'Sigma grid step (for VO)')
Flags.DEFINE_float('es_sigma_1d', 0.35, 'Fixed smoothing sigma (for ES)')
Flags.DEFINE_integer('probe_num', 256, 'Number of samples in function estimation (for ES and VO)')

FLAGS = Flags.FLAGS

if FLAGS.plot_type == "1d":
    X, Y, descent_history_x, descent_history_y = prepare_descend_1d(
        FLAGS, make_convex_1d)
    animate_1d(FLAGS, X, Y, descent_history_x, descent_history_y)
elif FLAGS.plot_type == "2d":
    X, Y, Z, descent_history_x, descent_history_y = prepare_descend_2d(
        FLAGS, make_wobbly_square_2d)
    animate_2d(FLAGS, X, Y, Z, descent_history_x, descent_history_y, contour_plot=False)
elif FLAGS.plot_type == "1d_es":
    X, Y, Y_outline, descent_history_x, descent_history_y = prepare_descend_1d_es(
        FLAGS, make_two_canyons_1d)
    animate_1d(FLAGS, X, Y, descent_history_x, descent_history_y, Y_outline)
elif FLAGS.plot_type == "1d_vo_to_2d":
    X, Y, Z, descent_history_x, descent_history_y = prepare_descend_1d_vo_to_2d(
        FLAGS, make_two_canyons_1d)
    animate_2d(FLAGS, X, Y, Z, descent_history_x, descent_history_y, smoothing_sigma=1.25)
else:
    raise ValueError("Unknown plot type")
