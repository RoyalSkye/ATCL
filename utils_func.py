import matplotlib.pyplot as plt
import numpy as np


def show(x, y, label, title, xdes, ydes, path, x_scale="linear", dpi=300):
    plt.style.use('fivethirtyeight')  # bmh, fivethirtyeight, Solarize_Light2
    plt.figure(figsize=(10, 8))
    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:cyan',
              'tab:gray', 'tab:brown', 'tab:purple', 'tab:olive', 'tab:pink']
    # colors = ['tab:pink', 'tab:olive', 'tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:cyan',
              # 'tab:gray', 'tab:brown', 'tab:purple']

    assert len(x) == len(y)
    for i in range(len(x)):
        if i < len(label):
            plt.plot(x[i], y[i], color=colors[i], label=label[i], linewidth=1.5)  # linewidth=1.5
        else:
            plt.plot(x[i], y[i], color=colors[i % len(label)], linewidth=1.5)  # linewidth=1.5

    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes, fontsize=24)
    plt.ylabel(ydes, fontsize=24)

    plt.title(title, fontsize=24)
    # my_y_ticks = np.arange(0, 1.1, 0.2)
    # plt.yticks(my_y_ticks, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='lower right', fontsize=16)
    plt.xscale(x_scale)
    # plt.margins(x=0)

    # plt.grid(True)
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


def show_std(x, y, label, title, xdes, ydes, path, x_scale="linear", std=True, dpi=300):
    """
        input: x/y: e.g.: [[exp1, exp2, exp3], [at1, at2, at3], ...]
    """
    print(">> Plot with mean with std err here!")
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 8))
    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:cyan',
              'tab:gray', 'tab:brown', 'tab:purple', 'tab:olive', 'tab:pink']
    # colors = ['tab:pink', 'tab:olive', 'tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:cyan',
    #           'tab:gray', 'tab:brown', 'tab:purple']

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
            y_err[i] = np.std(y_zip[i]) / np.sqrt(3)  # len(y[0])
        plt.plot(xx[0], y_est, color=colors[k], label=label[k])  # linewidth=2.0 linestyle=":", marker='o', ms=15
        if std:
            plt.fill_between(xx[0], y_est - y_err, y_est + y_err, alpha=0.2, color=colors[k])

    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes, fontsize=24)
    plt.ylabel(ydes, fontsize=24)

    plt.title(title, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='lower right', fontsize=20)
    plt.xscale(x_scale)
    # plt.margins(x=0)

    # plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close("all")
