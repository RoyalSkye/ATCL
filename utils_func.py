import matplotlib.pyplot as plt
import numpy as np


def show(x, y, label, title, xdes, ydes, path, x_scale="linear", dpi=150):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 8))
    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:cyan',
              'tab:gray', 'tab:brown', 'tab:purple', 'tab:olive', 'tab:pink']

    assert len(x) == len(y)
    for i in range(len(x)):
        if i < len(label):
            plt.plot(x[i], y[i], color=colors[i], label=label[i])
        else:
            plt.plot(x[i], y[i], color=colors[i % len(label)])

    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes)
    plt.ylabel(ydes)

    plt.title(title)
    plt.legend(loc='lower right', fontsize=10)
    plt.xscale(x_scale)

    plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close("all")


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(nb_param, nb_param/1e6))


def stat(x_to_mcls, x_to_tls):
    # test how many data are given wrong cls
    wrong_cl_count, correct_cl_count = 0, 0
    for k, v in x_to_mcls.items():
        correct_cl_count += len(v)
        if x_to_tls[k] in v:
            wrong_cl_count += 1
    # test how many correct cls are given to each data
    correct_cl_count -= wrong_cl_count

    return wrong_cl_count/len(x_to_mcls), correct_cl_count/len(x_to_mcls)


def show_std(x, y, label, title, xdes, ydes, path, x_scale="linear", dpi=150):
    """
        input: x/y: e.g.: [[exp1, exp2, exp3], [at1, at2, at3], ...]
    """
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 8))
    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:cyan',
              'tab:gray', 'tab:brown', 'tab:purple', 'tab:olive', 'tab:pink']

    assert len(x) == len(y)
    for k in range(len(y)):
        xx, yy = x[k], y[k]
        epoch = len(xx[0])
        y_zip, y_est, y_err = [], np.array([0.] * epoch), np.array([0.] * epoch)
        for i in range(epoch):
            ll = []
            for j in range(len(yy)):
                ll.append(yy[j][i])
            y_zip.append(ll)
        for i in range(epoch):
            y_est[i] = np.mean(y_zip[i])
            y_err[i] = np.std(y_zip[i])
        plt.plot(xx[0], y_est, color=colors[k], label=label[k])
        plt.fill_between(xx[0], y_est - y_err, y_est + y_err, alpha=0.2, color=colors[k])

    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes)
    plt.ylabel(ydes)

    plt.title(title)
    plt.legend(loc='lower right', fontsize=10)
    plt.xscale(x_scale)

    plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close("all")
