import io
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})


def normalize(x):
    u, l = x.max(), x.min()
    return (x - l) / (u - l)


def plot_figure(tmp, max_size=4):
    _, _, xyz, wpqr, y, xx = tmp
    xx = normalize(xx)
    map_fn = lambda x: str(round(x, 2))
    plot_size = min(max_size, len(xyz))
    figure = plt.figure(figsize=(2 * plot_size, 4))
    for i in range(plot_size):
        xyzwpqr = list(map(map_fn, xyz[i].tolist() + wpqr[i].tolist()))
        plt.subplot(plot_size, 2, 2*i + 1, title=','.join(xyzwpqr))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(xx[i, 0])
        yy = list(map(map_fn, y[i].tolist()))
        plt.subplot(plot_size, 2, 2*i + 2, title=','.join(yy))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(xx[i, 1])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    return buf.getvalue()
