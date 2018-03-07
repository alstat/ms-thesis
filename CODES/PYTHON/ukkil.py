"""
UKKIL FUNCTIONS
"""

import matplotlib.pylab as plt

def xyplot(x, y, xlab = "", ylab  = "", xlab_fontsize = 15, ylab_fontsize = 15, color = "", 
    bgcol = "#FFFFFF", marker = "o", markersize = 5, grid_lty = "solid", linestyle = "None", 
    linewidth = 1.5, ticksize = 12, grid_color = "#F2F3F4", figsize = (6, 6), file_name = "", 
    save = False, add = False):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    
    """ X-Y PLOT
    --------------------------
    x - input for x-axis
    y - input for y-axis
    """

    if add == True:
        plt.gcf()
        plt.gca()
        if color == "":
            plt.plot(x, y, marker = marker, markersize = markersize, linestyle = linestyle, linewidth = linewidth)
            plt.show()
        else:
            plt.plot(x, y, color = color, marker = marker, markersize = markersize, linestyle = linestyle, linewidth = linewidth)
            plt.show()
    else:
        f, ax = plt.subplots(figsize = figsize)
        if color == "":
            plt.plot(x, y, marker = marker, markersize = markersize, linestyle = linestyle, linewidth = linewidth)
        else:
            plt.plot(x, y, color = color, marker = marker, markersize = markersize, linestyle = linestyle, linewidth = linewidth)
        plt.minorticks_on()
        ax.set_xlabel(xlab, fontsize = xlab_fontsize)
        ax.set_ylabel(ylab, fontsize = ylab_fontsize)
        ax.grid("on", which = "major", color = grid_color, linestyle = grid_lty)
        ax.set_axis_bgcolor(bgcol)
        ax.tick_params(axis = "both", which = "major", labelsize = ticksize, pad = 5)
        ax.tick_params(axis = "both", which = "minor", labelsize = ticksize, pad = 5)
        ax.set_axisbelow("on")
        ax.locator_params(axis = "x", nbins = 7, tight = True)
        ax.locator_params(axis = "y", nbins = 7, tight = True)
        ax.margins(.03)

    if save == True:
        if file_name == "":
            plt.savefig(joinpath(pwd(), "Current Plot.png"), bbox_inches = "tight", dpi = 200)
            plt.cla()
            plt.clf()
            plt.close()
        else:
            plt.savefig(file_name, bbox_inches = "tight", dpi = 200)
            plt.cla()
            plt.clf()
            plt.close()

