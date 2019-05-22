import io
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_figure(tmp, max_size=10):
    _, _, xyz, wpqr, y, xx = tmp
    plot_size = min(max_size, len(xyz))
    figure = plt.figure(figsize=(plot_size, 2))
    for i in range(plot_size):
        plt.subplot(plot_size, 2, 2*i + 1, title=','.join(xyz[i].tolist() + wpqr[i].tolist()))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(xx[i, 0])
        plt.subplot(plot_size, 2, 2*i + 2, title=','.join(y[i]))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x2_[i, 1])
    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image
