import matplotlib.pyplot as plt


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
    plt.legend(loc='upper right', fontsize=10)
    plt.xscale(x_scale)

    plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close("all")