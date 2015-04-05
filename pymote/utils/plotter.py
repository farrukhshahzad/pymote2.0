# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os
import scipy as sp
import matplotlib.pyplot as plt



# plot input data
def plots(x, y, fname, ymax=None, xmin=None, ymin=None,
          xlabel="X", ylabel="Y", title="Plot", labels="", more_plots=None):

    colors = ['g', 'k', 'b', 'm', 'r']
    line_styles = ['o-', '-.', '--', 'o:', 'x-']
    styles = ['go-', 'k-.', 'm.-', 'bx:', 'ro--']
    plt.figure(num=None, figsize=(9, 6))
    plt.clf()

    if more_plots and isinstance(more_plots, list):
        more_plots.insert(0, y)
        for yy in more_plots:
            #plt.plot(x, yy, linestyle=line_styles.pop(), linewidth=1, c=colors.pop())
            plt.plot(x, yy, styles.pop())
            #plt.scatter(x, yy, s=10)
        plt.legend(["%s" % m for m in labels], loc="upper left")
    else:
        plt.plot(x, y)
        plt.scatter(x, y, s=10)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.autoscale(tight=True)
    if ymin:
        plt.ylim(ymin=ymin)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)

    plt.grid(True, linestyle='-', color='0.75')
    #plt.show()
    plt.savefig(fname)


# plot bars input data
def plot_bars(x, y, fname, ymax=None, xmin=None, ymin=None,
          xlabel="X", ylabel="Y", title="Plot", color='r'):

    plt.figure(num=None, figsize=(9, 6))
    plt.clf()
    #plt.scatter(x, y, s=10)
    width = 0.5
    plt.bar(x-width/2, y, width, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.autoscale(tight=True)
    if ymin:
        plt.ylim(ymin=ymin)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)

    plt.grid(True, linestyle='-', color='0.75')
    #plt.show()
    plt.savefig(fname)

if __name__ == '__main__':
    plot_bars([1,2,3,4], [100,200,34,200], None, "test", title="Good Graph")