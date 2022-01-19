
#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Copyright by Artur K. Lidtke, Univ. of Southampton, UK, 2014

This code is distributed under GNU Lesser General Public Licence Agreement
version 3 or newer and without any warranty of merchantability or suitability
for any particular or general purpose.
Please see http://www.gnu.org/licenses/lgpl.html for details.

Created on Wed Jan 20 15:28:06 2016
"""

import matplotlib
import matplotlib.pyplot as plt

#matplotlib.style.use("classic")

def getColours(N, cmap="jet"):
    """ Returns a list of N colours spanning evenly across the colour map (default jet - blue-green-red) """
    cm = matplotlib.pyplot.cm.get_cmap(cmap)
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=N-1)
    colourMapRGB = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
    return [colourMapRGB.to_rgba(iCurve) for iCurve in range(N)]

markers = ['d', 'p', 's', 'v', 'o', '8', '^', 'D', '*', 'h', '<', 'H', '>', '+']*3

tickFontProperties = matplotlib.font_manager.FontProperties(family='serif',
                      style='normal',weight='normal',size=16)

def makeAxesNice(fig, ax, xlab='', ylab='', figtitle='', zlab="",
            marginL=False, marginR=False, marginT=False, marginB=False,
            xLabPad=False, yLabPad=False, zLabPad=False,
            rightAndTopBorder=False, axes3d=False):
    """ Apply nice formatting to the given figure and axes """
    # override default for a 3D plot
    # TODO this would be much better done with **kwargs, but hey, it works
    if axes3d:
        if not marginL:
            marginL = 0.05
        if not marginR:
            marginR = 0.95
        if not marginT:
            marginT = 0.95
        if not marginB:
            marginB = 0.05

        if not xLabPad:
            xLabPad = 20
        if not yLabPad:
            yLabPad = 20
        if not zLabPad:
            zLabPad = 20

    else:
        if not marginL:
            marginL = 0.125
        if not marginR:
            marginR = 0.96
        if not marginT:
            marginT = 0.91
        if not marginB:
            marginB = 0.16

    # set title
    if len(figtitle)>0:
        fig.canvas.set_window_title(figtitle)
    try:
        ax.tick_params(axis='both', reset=False, which='both', length=5, width=2)
    except KeyError:
        pass
    for spine in ['top',  'right', 'bottom', 'left', "polar"]:
        try:
            ax.spines[spine].set_linewidth(2)
        except KeyError:
            pass

    ax.tick_params(axis='y', direction='out', which="both")
    ax.tick_params(axis='x', direction='out', which="both")
    if axes3d:
        ax.tick_params(axis='z', direction='out', which="both")

    fig.subplots_adjust(left=marginL, right=marginR, top=marginT, bottom=marginB, hspace=0.2)
    fig.patch.set_facecolor("white")

    if len(xlab)>0:
        ax.set_xlabel(r'${}$'.format(xlab.replace(' ','\,')), fontsize=20)
        if xLabPad:
            ax.xaxis.labelpad = xLabPad
    if len(ylab)>0:
        ax.set_ylabel(r'${}$'.format(ylab.replace(' ','\,')), fontsize=20)
        if yLabPad:
            ax.yaxis.labelpad = yLabPad
    if len(zlab)>0:
        ax.set_zlabel(r'${}$'.format(zlab.replace(' ','\,')), fontsize=20)
        if zLabPad:
            ax.zaxis.labelpad = zLabPad

    for label in ax.get_xticklabels() :
        label.set_fontproperties(tickFontProperties)
    for label in ax.get_yticklabels() :
        label.set_fontproperties(tickFontProperties)
    if axes3d:
        for label in ax.get_zticklabels() :
            label.set_fontproperties(tickFontProperties)

    if (not rightAndTopBorder) and ("polar" not in ax.spines.keys()):
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        if axes3d:
            ax.yaxis.set_ticks_position('bottom')
            ax.xaxis.set_ticks_position('bottom')
            ax.zaxis.set_ticks_position('bottom')
        else:
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

    # set background for a 3D plot
    if axes3d:
        # remove the grey thing
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        # format grid lines
        ax.w_xaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 0.5), "linewidth": 0.5, 'linestyle': "--"}})
        ax.w_yaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 0.5), "linewidth": 0.5, 'linestyle': "--"}})
        ax.w_zaxis._axinfo.update({'grid' : {'color': (0, 0, 0, 0.5), "linewidth": 0.5, 'linestyle': "--"}})

def niceFig(xArrs, yArrs, xlab='', ylab='', figtitle='', labels=[], xlim=[], ylim=[],
            style='bw', marginL=0.125, marginR=0.96, marginT=0.91, marginB=0.16, ncol=1, figSize=None,
            fontsizeLabels=20, fontsizeTicks=16, fontsizeLegend=18, rightAndTopBorder=False,
            legbox=[], legloc="best", returnTwinAxes=False, twinYlabel='', returnCmap=False, colours=None):
    """ Create a nice figure """
    if colours == None:
        colours = getColours(len(xArrs))
    else:
        style = "colour"

    lineStyles = ['-','--','-.',':']

    if figSize is not None:
        fig, ax = plt.subplots(1, figsize=figSize)
    else:
        fig, ax = plt.subplots(1)
    if len(figtitle)>0:
        fig.canvas.set_window_title(figtitle)

    ax.tick_params(axis='both',reset=False,which='both',length=5,width=2)
    ax.tick_params(axis='y', direction='out', which="both")
    ax.tick_params(axis='x', direction='out', which="both")

    fig.patch.set_facecolor("white")

    for spine in ['top', 'right','bottom','left']:
        ax.spines[spine].set_linewidth(2)

    fig.subplots_adjust(left=marginL, right=marginR, top=marginT, bottom=marginB, hspace=0.2)

    ax.set_xlabel(r'${}$'.format(xlab.replace(' ','\,')),fontsize=fontsizeLabels)
    ax.set_ylabel(r'${}$'.format(ylab.replace(' ','\,')),fontsize=fontsizeLabels)

#    tickFontProperties = matplotlib.font_manager.FontProperties(family='serif',
#                      style='normal',weight='normal',size=fontsizeTicks)

    if len(xlim)>0:
        ax.set_xlim(xlim)
    if len(ylim)>0:
        ax.set_ylim(ylim)

    lns = []
    for i in range(len(xArrs)):
        if style == 'bw':
            lns += ax.plot(xArrs[i],yArrs[i],lw=2,c='k',ls=lineStyles[i])
        elif style in ["c", "color", "colour", "colours"]:
            lns += ax.plot(xArrs[i],yArrs[i],lw=2,c=colours[i],ls='-')
        else:
            lns += ax.plot(xArrs[i],yArrs[i],lw=2,c=colours[i],ls='-')

    if len(labels)>1:
        if len(legbox)==0:
            legend = ax.legend(lns,[r'${}$'.format(l.replace(' ','\,')) for l in labels],
                    ncol=ncol,prop={'size':fontsizeLegend},loc=legloc)
        else:
            legend = ax.legend(lns,[r'${}$'.format(l.replace(' ','\,')) for l in labels],
                    ncol=ncol,prop={'size':fontsizeLegend},bbox_to_anchor=legbox)

        legend.get_frame().set_linewidth(2)

    for label in ax.get_xticklabels():
        label.set_fontproperties(tickFontProperties)
    for label in ax.get_yticklabels():
        label.set_fontproperties(tickFontProperties)

    if returnTwinAxes:
        ax2 = plt.twinx()
        ax2.tick_params(axis='both',reset=False,which='both',length=5,width=2)
        ax2.tick_params(axis='y', direction='out', which="both")
        ax2.tick_params(axis='x', direction='out', which="both")

        for spine in ['top', 'right','bottom','left']:
            ax2.spines[spine].set_linewidth(2)
        for label in ax2.get_yticklabels() :
            label.set_fontproperties(tickFontProperties)

        ax2.set_ylabel(r'${}$'.format(twinYlabel.replace(' ','\,')),fontsize=fontsizeLabels)

    if not rightAndTopBorder:
        if returnTwinAxes:
            for a in [ax, ax2]:
                a.spines['top'].set_visible(False)
                a.xaxis.set_ticks_position('bottom')
        else:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

    # create the return tuple
    ret = [fig,ax]

    if returnCmap:
        ret.append(colours)

    if returnTwinAxes:
        ret.append(ax2)

    return ret

def addColourBar(fig, cs, cbarLabel, pos=[0.85, .25, 0.03, 0.5], fontsize=20, orientation="vertical"):
    """ Add a nice colour bar """
    position = fig.add_axes(pos)#[0.85, .25, 0.03, 0.5])
    cbar = fig.colorbar(cs, cax=position, orientation=orientation)
#        cbar.ax.tick_params(labelsize=16)
    for label in cbar.ax.get_xticklabels():
        label.set_fontproperties(tickFontProperties)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(tickFontProperties)
    cbar.set_label("${}$".format(cbarLabel), fontsize=fontsize)
    return cbar

# =======
# module test

if __name__ == "__main__":
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    # === UNIT TEST 0: 2D PLOTS ===
    # create some data
    x = np.linspace(0, 2*np.pi, 101)
    y0 = np.sin(x)
    y1 = np.sin(2.*x)

    # plot black and white
    fig, ax = niceFig([x, x], [y0, y1], "x [-]", "f(x) [-]", "Default figure")

    # plot in colour with a legend, custom margins and limits
    fig, ax = niceFig([x, x], [y0, y1], "x [-]", "f(x) [-]", "Colourful figure", ["sin(x)", "sin(2x)"],
            style="c", marginL=0.15, marginR=0.95, marginB=0.15, marginT=0.95)

    # plot on two axes
    fig, ax0, ax1 = niceFig([], [], "x [-]", "f(x) [-]", "Twin axes figure",
            returnTwinAxes=True, twinYlabel="sin^2(2x)", marginR=0.875)
    l0 = ax0.plot(x, y0, 'r-', lw=2, label=r"$sin(x)$")
    l1 = ax1.plot(x, y1**2, 'b-', lw=2, label=r"$sin^2(2x)$")
    legend = ax1.legend(l0+l1, [l.get_label() for l in l0+l1], loc="best", prop={"size":18})
    legend.get_frame().set_linewidth(2)

    # === UNIT TEST 1: SURFACE PLOT ===
    # create some data
    x = np.linspace(0, 2*np.pi, 101)
    y = np.linspace(-np.pi, np.pi, 101)
    x, y = np.meshgrid(x, y)
    z = np.sin(2.*x) * np.cos(y)

    # prepare axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # apply formatting
    makeAxesNice(fig, ax, "x/\pi [-]", "y/\pi [-]", "3D plot", zlab="f(x,y) [-]", axes3d=True, marginR=0.8)

    # plot with lines
#    ax.plot_wireframe(x/np.pi, y/np.pi, z, color="k", lw=2, rstride=5, cstride=5)
    # plot as a coloured surface
    cmap = plt.cm.jet
    cs = ax.plot_surface(x/np.pi, y/np.pi, z, linewidth=1, alpha=0.75,
                    cmap=cmap, norm=matplotlib.colors.BoundaryNorm(np.linspace(-1, 1, 25), cmap.N))
    # add colour bar
    cbar = addColourBar(fig, cs, "f(x,y)")

    plt.show()
