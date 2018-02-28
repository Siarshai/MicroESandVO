import numpy as np
from matplotlib import pyplot as plt, animation
import scipy.ndimage as ndimage


def animate_1d(FLAGS, X, Y, descent_history_x, descent_history_y, Y_outline=None):
    print("Animating 1D")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot("111")
    if Y_outline is not None:
        ax.plot(X, Y_outline, lw=2, color="green")
    ax.plot(X, Y, lw=2, color="blue")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    clearlist = [plt.scatter([descent_history_x[0]], [descent_history_y[0]], c=(1.0, 0.0, 0.0))]

    def animate(i):
        trace_length = 8
        while clearlist:
            c = clearlist.pop()
            c.remove()
            del c
        for j in range(max(0, i - trace_length), i):
            clearlist.append(plt.scatter([descent_history_x[j]], [descent_history_y[j]],
                                         c=(1.0, 0.0, 0.0),
                                         alpha=(1.0 - (i - j) / (trace_length + 1) )))
        clearlist.append(plt.scatter([descent_history_x[i]], [descent_history_y[i]], c=(1.0, 0.0, 0.0)))
        return clearlist

    if FLAGS.save_animation:
        anim = animation.FuncAnimation(fig, animate, frames=FLAGS.steps_to_descend, interval=80, blit=True)
        anim.save('descent.mp4', writer="ffmpeg")
    else:
        anim = animation.FuncAnimation(fig, animate, frames=FLAGS.steps_to_descend, interval=80, blit=False)
        plt.show()


def animate_2d(FLAGS, X, Y, Z, descent_history_x, descent_history_y, smoothing_sigma=0.0, contour_plot=True):
    print("Animating 2D")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot("111")
    ax.set_xlabel("x")
    xmin, xmax, ymin, ymax = np.amin(X), np.amax(X), np.amin(Y), np.amax(Y)
    if smoothing_sigma > 0.0:
        Z = ndimage.gaussian_filter(Z, sigma=(smoothing_sigma, smoothing_sigma), order=0)
    if contour_plot:
        levels = np.arange(-4, 10, 0.25)
        CS = ax.contour(X, Y, Z, levels=levels)
        plt.clabel(CS, inline=1, fontsize=8)
        ax.set_ylabel("sigma")
    else:
        ax.imshow(Z, extent=(xmin, xmax, ymin, ymax),
                  origin="lower", cmap="ocean", aspect='auto')
        ax.set_ylabel("y")

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    clearlist = [plt.scatter([descent_history_x[0]], [descent_history_y[0]], c=(1.0, 0.0, 0.0))]

    def animate(i):
        trace_length = 8
        while clearlist:
            c = clearlist.pop()
            c.remove()
            del c
        for j in range(max(1, i - trace_length), i+1):
            clearlist.append(plt.plot([descent_history_x[j-1], descent_history_x[j]],
                                      [descent_history_y[j-1], descent_history_y[j]],
                                      c=(1.0, 0.0, 0.0),
                                      alpha=(1.0 - (i-j)/(trace_length+1)))[0])
        clearlist.append(plt.scatter([descent_history_x[i]], [descent_history_y[i]], c=(1.0, 0.0, 0.0)))
        return clearlist

    if FLAGS.save_animation:
        anim = animation.FuncAnimation(fig, animate, frames=FLAGS.steps_to_descend, interval=80, blit=True)
        anim.save('descent.mp4', writer="ffmpeg")
    else:
        anim = animation.FuncAnimation(fig, animate, frames=FLAGS.steps_to_descend, interval=80, blit=False)
        plt.show()
