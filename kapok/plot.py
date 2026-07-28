# -*- coding: utf-8 -*-
"""Helper functions for easy histogram and density plots, as well as a
    function for linear regression using the covariance matrix of the data.
    The histogram and density plots are plotted using matplotlib's hist and
    hist2d functions.  The helper functions handle the axis labels,
    fitted line display, etc., in order to produce complete plots with a
    single function call.

    Authors: Guillaume Brigot (linear regression function), Michael Denbina
        (histogram and density plot helper functions).

    Copyright 2016 California Institute of Technology.  All rights reserved.
    United States Government Sponsorship acknowledged.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import collections

import numpy as np
import h5py

from numpy.linalg import eig
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



def covariance_regression(x,y):
    """Linear regression using the covariance matrix.  Code by Guillaume
    Brigot.

    Arguments:
        x (array): the x variable to perform the regression on.
        y (array): the y variable to perform the regression on.

    Returns:
        a: the slope of the line.
        b: the y-intercept of the line.
        rsq: the r-squared value of the line fit fit (1 - SSresidual/SStotal),
            where SSresidual is the sum of the squares of the residuals, and
            SStotal is the total sum of squares.
        corr: the Pearson correlation coefficient between the data's y
            values and the fitted line.

    """
    cov_matrix = np.cov(x, y)
    w, v = eig(cov_matrix)
    ind_max = np.argmax(w)
    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)
    a, b = np.polyfit([mean_x,mean_x - v[ind_max][0]],[mean_y,mean_y + v[ind_max][1]], 1)

    SSres = np.sum(np.square(y - (a*x + b)))
    SStot = np.sum(np.square(y - mean_y))
    rsq = (1 - SSres/SStot)
    corr = np.corrcoef((a*x + b), y)[0, 1]

    return a, b, rsq, corr


def density(x, y, mask=None, xlim=None, ylim=None, xname=None,
            yname=None, title=None, cmin=0, numbins=100, simline=True, fitline=True,
            showstats=True, showbinavg=False, minsamplebinavg=100, returnbinavg=False,
            units=None, cmap='gist_heat_r', lognorm=False, cov_regression=True,
            figsize=None, savefile=None, dpi=125, **kwargs):
    """Function for easy plotting of a 2D histogram (density plot) of the
    input x and y data (2D histogram).

    These plots provide a comparison of the distribution of two variables,
    often variables we would expect to be similar (e.g., radar and lidar
    forest heights).

    Arguments:
        x (array): the data to plot on the x axis.
        y (array): the data to plot on the y axis.
        mask (array): boolean array to choose which data is included in the
            plot.  Mask should be set to True for pixels to be plotted, False
            in pixels not to be plotted.
        xlim (tuple): the lower and upper x axis limits.  Default: (0, 50).
        ylim (tuple): the lower and upper y axis limits.  Default: Same as
            xlim.
        xname (str): the x axis label.
        yname (str): the y axis label.
        cmin: the minimum number of samples for a bin to be plotted on the
            histogram.  Default: 0 (all data).
        numbins (int): the number of histogram bins along each axis.  Default:
            100.
        simline (bool): True to display the line y=x.  False to not display.
            Default: False.
        fitline (bool): True to display a fitted line to the data.  False to
            not display.  Default: False.
        showstats (bool): True to display some stats (bias, std. dev., RMSE)
            in text on the plot.  False to not display.  Default: False.
        units (str): Units of the provided data, e.g., 'm' for height data.
            Default: '' (empty string).
        cmap: The colormap to use.  Default: 'gist_heat_r'.
        lognorm (bool): Set to True to use logarithmic scaling.  False for
            linear frequency scaling.  Default: False.
        cov_regression (bool): Set to True to use covariance matrix
            regression for the line fit, using the covariance_regression()
            function above.  Set to False to use standard least
            squares regression as implemented in SciPy.  Default: True.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        savefile (str): Path and filename to save the figure.  Default: None.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            200.

    """
    if mask is None:
        mask = np.ones(x.shape, dtype='bool')

    if xlim is None:
        xlim = (np.floor(np.nanmin(x[mask])),np.ceil(np.nanmax(x[mask])))

    xlim = np.array(xlim)

    if units is None:
        units = ''

    ind = mask & np.isfinite(x) & np.isfinite(y)

    if ylim is None:
        ylim = xlim
    else:
        ylim = np.array(ylim)

    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)

    ax = plt.gca()

    # 2D Histogram
    if lognorm == False:
        (counts, xedges, yedges, Image) = plt.hist2d(x[ind].flatten(),y[ind].flatten(),bins=(numbins,numbins),range=((xlim[0],xlim[1]),(ylim[0],ylim[1])),cmin=cmin,cmap=cmap,density=True,**kwargs)
    else:
        from matplotlib.colors import LogNorm
        (counts, xedges, yedges, Image) = plt.hist2d(x[ind].flatten(),y[ind].flatten(),bins=(numbins,numbins),range=((xlim[0],xlim[1]),(ylim[0],ylim[1])),cmin=cmin,cmap=cmap,norm=LogNorm(),**kwargs)

    # Lines:
    if simline:
        plt.plot(xlim,xlim,'k--')

    if cov_regression:
        slope, intercept, rsq, corr = covariance_regression(x[ind].flatten(),y[ind].flatten())
    else:
        from scipy.stats import linregress
        slope, intercept, rvalue, pval, stderr = linregress(x[ind].flatten(),y[ind].flatten())
        rsq = rvalue**2


    if fitline == True:
        plt.plot(xlim,xlim*slope + intercept,'r--')
        if np.sign(intercept) == 1:
            plt.text((xlim[1]-xlim[0])*0.025 + xlim[0],(ylim[1]-ylim[0])*0.94 + ylim[0], 'y = '+str(np.round(slope,decimals=2))+'x + '+str(np.round(intercept,decimals=2)), size=14)
        else:
            plt.text((xlim[1]-xlim[0])*0.025 + xlim[0],(ylim[1]-ylim[0])*0.94 + ylim[0], 'y = '+str(np.round(slope,decimals=2))+'x - '+str(np.round(np.abs(intercept),decimals=2)), size=14)

    if (fitline == True):
        plt.text((xlim[1]-xlim[0])*0.025 + xlim[0],(ylim[1]-ylim[0])*0.86 + ylim[0], r'$r^2$: '+str(np.round(rsq,decimals=2)),size=14)

    if (showbinavg == True):
        xcenters = (xedges[0:-1] + xedges[1:]) / 2
        ybinavg = np.ones(xcenters.shape[0], dtype='float64') * np.nan

        for num in range(len(xcenters)):
            ind_data = (x >= xedges[num]) & (x < xedges[num+1])
            if np.sum(ind_data) > minsamplebinavg:
                ybinavg[num] = np.nanmean(y[ind_data])

        plt.plot(xcenters, ybinavg, 'b--')


    bias = np.mean(y[ind] - x[ind])
    bias = np.round(bias, decimals=2)
    rms = np.sqrt(np.mean(np.square(y[ind] - x[ind])))
    rms = np.round(rms, decimals=2)
    stdev = np.std(y[ind] - x[ind])
    stdev = np.round(stdev, decimals=2)
    stdev_x = np.std(x[ind])
    stdev_x = np.round(stdev_x, decimals=2)
    n = len(y[ind])

    if showstats == True:
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.23 + ylim[0],'Bias: '+str(bias)+' '+units,size=14,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.18 + ylim[0],'Y-X Std.Dev.: '+str(stdev)+' '+units,size=14,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.13 + ylim[0],'X Std.Dev.: '+str(stdev_x)+' '+units,size=14,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.08 + ylim[0],'RMSE: '+str(rms)+' '+units,size=14,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.03 + ylim[0],'N: '+str(n),size=14,horizontalalignment='right')
    else:
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.03 + ylim[0],'N: '+str(n),size=14,horizontalalignment='right')

    if xname is not None:
        plt.xlabel(xname)

    if yname is not None:
        plt.ylabel(yname)

    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar()
    cbar.set_label('Frequency', rotation=90)

    if title is not None:
        plt.title(title)

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    
    if returnbinavg:
        return xcenters, ybinavg



def hist(x, mask=None, xlim=None, ylim=None, xname=None, yname=None,
         numbins=50, units=None, title=None, plotmean=False, showrms=False,
         figsize=None, norm=True, savefile=None, dpi=125):
    """Helper function for easy 1D histogram plotting.

    Arguments:
        x (array): data to plot.
        mask (array): boolean array to choose which data is included in the
            plot.  Mask should be set to True for pixels to be plotted, False
            in pixels not to be plotted.
        xlim (tuple): the lower and upper x axis limits.  Default: (0, 50).
        ylim (tuple): The lower and upper y axis limits.  Defaut: Determined
            by data.
        xname (str): the x axis label.
        numbins (int): the number of histogram bins.  Default: 50.
        units (str): Units of the provided data, e.g., 'm' for height data.
            Default: '' (empty string).
        title (str): Title for the plot.  Default: None.
        plotmean (bool): Plot the mean value as a vertical dashed line.
            Default: False.
        showrms (bool): Set to True to display the RMS value of x on the plot.
            (e.g., if the input data are height errors).  False: Do not
            display.  Default: False.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        norm (bool): Normalize histogram and show relative frequency.
            Default: True.
        savefile (str): Path and filename to save the figure.  Default: None.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            200.

    """
    if mask is None:
        mask = np.ones(x.shape, dtype='bool')

    if xlim is None:
        xlim = (np.floor(np.nanmin(x[mask])),np.ceil(np.nanmax(x[mask])))

    xlim = np.array(xlim)

    if units is None:
        units = ''

    ind = mask & np.isfinite(x)

    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)

    ax = plt.gca()

    if norm:
        plt.hist(x[ind].flatten(), bins=numbins, range=xlim, density=True, edgecolor='black', linewidth=1.2)
    else:
        plt.hist(x[ind].flatten(), bins=numbins, range=xlim, edgecolor='black', linewidth=1.2)

    bias = np.mean(x[ind])
    bias = np.round(bias, decimals=2)
    rms = np.sqrt(np.mean(np.square(x[ind])))
    rms = np.round(rms, decimals=2)
    stdev = np.std(x[ind])
    stdev = np.round(stdev, decimals=2)
    n = len(x[ind])

    if ylim is None:
        ylim = ax.get_ylim()

    if plotmean:
        plt.plot([bias,bias],[ylim[0],ylim[1]],'--r')

    plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.94 + ylim[0],'Mean: '+str(bias)+' '+units,horizontalalignment='right')
    plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.86 + ylim[0],'Std.Dev.: '+str(stdev)+' '+units,horizontalalignment='right')

    if showrms:
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.78 + ylim[0],'RMS: '+str(rms)+' '+units,horizontalalignment='right')
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.70 + ylim[0],'N: '+str(n),horizontalalignment='right')
    else:
        plt.text((xlim[1]-xlim[0])*0.97 + xlim[0],(ylim[1]-ylim[0])*0.78 + ylim[0],'N: '+str(n),horizontalalignment='right')

    if xname is not None:
        plt.xlabel(xname)

    if yname is not None:
        plt.ylabel(yname)
    elif norm:
        plt.ylabel('Frequency')
    else:
        plt.ylabel('Count')

    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])

    if title is not None:
        plt.title(title)

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)



def swath(inc, data, mask=None, xlim=None, ylim=None, yname=None, figsize=None, savefile=None, dpi=125):
    """Plot the average and standard deviation of an input quantity vs.
    incidence angle, to show the variation across the swath.

    The mean value of the data in each range bin is plotted with a solid
    line.  The mean plus and minus the standard deviation is shown as a
    shaded region.

    Arguments:
        inc (array): Incidence angles for each pixels, in radians.
        data (array): Data to plot.
        mask (array): Boolean array.  Where mask == True, the data will be
            included in the plot.  Where mask == False, the data will be
            excluded.
        xlim (tuple): X-axis limits.  Default: Full swath.
        ylim (tuple): Y-axis limits.  Default based on data.
        yname (str): The y-axis label.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        savefile (str): Path and filename to save the figure.  Default: None.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            200.

    """
    if mask is None:
        mask = np.ones(data.shape, dtype='bool')

    if xlim is None:
        xlim = (20,65)

    xlim = np.array(xlim)

    ind = mask & np.isfinite(data)
    data_masked = data.copy()
    data_masked[~ind] = np.nan
    inc_masked = inc.copy()
    inc_masked[~ind] = np.nan

    if figsize is None:
        fig, ax = plt.subplots(1)
    else:
        fig, ax = plt.subplots(1, figsize=figsize)

    inc_avg = np.degrees(np.nanmean(inc_masked, axis=0))
    data_avg = np.nanmean(data_masked, axis=0)
    data_std = np.nanstd(data_masked, axis=0)

    ax.fill_between(inc_avg, data_avg+data_std, data_avg-data_std, facecolor='blue', alpha=0.1)
    plt.plot(inc_avg, data_avg, linewidth=3)
    plt.xlim(xlim)
    plt.xlabel('Incidence Angle (degrees)')

    if ylim is not None:
        plt.ylim(ylim)

    if yname is not None:
        plt.ylabel(yname)

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)



def scatter_field(x, y, x_name='', y_name='', xlim=None, ylim=None,
                  simline=True, showfitlines=True, showstats=True,
                  cov_regression=True, savefile=None):
    """ Plot a scatter plot.
    """
    if xlim is not None:
        xlim = np.array(xlim)
        if ylim is None:
            ylim = xlim

    plt.figure()

    markersize = 8

    ind = np.isfinite(x) & np.isfinite(y)
    x = x[ind]
    y = y[ind]
    diff = y - x

    plt.plot(x, y, 'k+', markersize=markersize)

    if xlim is None:
        xlim = plt.gca().get_xlim()

    if ylim is None:
        ylim = plt.gca().get_ylim()

    if simline == True:
        plt.plot(xlim,ylim,'k--')

    if cov_regression:
        slope, intercept, rsq, corr = covariance_regression(x,y)
    else:
        from scipy.stats import linregress
        slope, intercept, rvalue, pval, stderr = linregress(x,y)
        rsq = rvalue**2


    if (showfitlines == True):
        plt.plot(xlim,xlim*slope+intercept,'r--')
        if np.sign(intercept) == 1:
            plt.text((xlim[1]-xlim[0])*0.025 + xlim[0],(ylim[1]-ylim[0])*0.94 + ylim[0], 'y = '+str(np.round(slope,decimals=2))+'x + '+str(np.round(intercept,decimals=2)),size=14)
        else:
            plt.text((xlim[1]-xlim[0])*0.025 + xlim[0],(ylim[1]-ylim[0])*0.94 + ylim[0], 'y = '+str(np.round(slope,decimals=2))+'x - '+str(np.round(np.abs(intercept),decimals=2)),size=14)
        plt.text((xlim[1]-xlim[0])*0.025 + xlim[0],(ylim[1]-ylim[0])*0.86 + ylim[0], '$r^2$: '+str(np.round(rsq,decimals=2)),size=14)


    bias = np.round(np.mean(diff), decimals=2)
    rms = np.round(np.sqrt(np.mean(np.square(diff))), decimals=2)
    mae = np.round(np.nanmean(np.abs(diff)), decimals=2)
    stdev = np.round(np.std(diff), decimals=2)
    n = len(y)

    y_fit = slope*x + intercept
    diff_fit = y - y_fit
    rms_fit = np.round(np.sqrt(np.mean(np.square(diff_fit))), decimals=2)
    mae_fit = np.round(np.nanmean(np.abs(diff_fit)), decimals=2)

    bias_str = 'Bias: '
    stdev_str = 'Std. Dev.: '
    rms_str = 'RMSE: '
    mae_str = 'MAE: '
    rms_fit_str = 'RMSE (vs. Fit): '
    mae_fit_str = 'MAE (vs. Fit): '

    if showstats:
        plt.text((xlim[1]-xlim[0])*0.60 + xlim[0],(ylim[1]-ylim[0])*0.33 + ylim[0],bias_str+str(bias)+'m',size=14)
        plt.text((xlim[1]-xlim[0])*0.60 + xlim[0],(ylim[1]-ylim[0])*0.28 + ylim[0],stdev_str+str(stdev)+'m',size=14)
        plt.text((xlim[1]-xlim[0])*0.60 + xlim[0],(ylim[1]-ylim[0])*0.23 + ylim[0],rms_str+str(rms)+'m',size=14)
        plt.text((xlim[1]-xlim[0])*0.60 + xlim[0],(ylim[1]-ylim[0])*0.18 + ylim[0],mae_str+str(mae)+'m',size=14)
        plt.text((xlim[1]-xlim[0])*0.60 + xlim[0],(ylim[1]-ylim[0])*0.13 + ylim[0],rms_fit_str+str(rms_fit)+'m',size=14)
        plt.text((xlim[1]-xlim[0])*0.60 + xlim[0],(ylim[1]-ylim[0])*0.08 + ylim[0],mae_fit_str+str(mae_fit)+'m',size=14)

    plt.text((xlim[1]-xlim[0])*0.60 + xlim[0],(ylim[1]-ylim[0])*0.03 + ylim[0],'N: '+str(n),size=14)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])

    if savefile is not None:
        plt.savefig(savefile, dpi=200, bbox_inches='tight', pad_inches=0.1)



def mbs(scene, az, rng, refheight=None, figsize=None, dpi=125, savefile=None):
    """Multi-baseline sensitivity plot.  Plots quantities such as coherence
    magnitude and phase height as a function of the kz value.

    Arguments:
        scene: Kapok Scene object.
        az (int): Azimuth index of pixel to plot.
        rng (int): Range index of pixel to plot.
        refheight: Reference forest height from other remote sensing
            or field data.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            125.
        savefile (str): Path and filename to save the figure.  Default: None.

    """
    import itertools
    # marker = itertools.cycle(('+','+','+','+','x','x','+','.','+','x')) # Pongara
    marker = itertools.cycle(('+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','+','x','+','+','+','+','+','x','.','+','+','+','+')) # Lope
    markersize = itertools.cycle((12,12,12,12,12,12,12,12,12,12))

    import kapok.rvog
    import kapok.topo
    from kapok.lib import mb_tr_index
    nptodb = 20/np.log(10)

    def calcmu(gamma, volume, ground):
        """ """
        mu = np.zeros(np.shape(volume),dtype='float32')
        p = np.zeros(np.shape(volume),dtype='complex64')

        l2 = np.square(np.abs(volume - ground))


        ind = (l2 == 0)
        if np.any(ind): # high and low coherences are equal -- usually happens with void data at edges of scene
            p[ind] = volume[ind]
            mu[ind] = np.nan


        ind = (l2 > 0)
        if np.any(ind):
            t = (np.real(gamma-ground)*np.real(volume-ground) + np.imag(gamma-ground)*np.imag(volume-ground))[ind] / l2[ind]
            t = np.clip(t, 0, 1)
            p[ind] = ground[ind] + t*(volume[ind]-ground[ind])
            mu[ind] = np.abs((volume[ind] - p[ind]) / (p[ind] - ground[ind]))

        return mu

    if isinstance(refheight, (collections.Sequence, np.ndarray, h5py.Dataset)):
        refheight = refheight[az,rng]

    if figsize is None:
        fig, ax = plt.subplots(2, 4)
    else:
        fig, ax = plt.subplots(2, 4, figsize=figsize)

    # Get vector of optimized coherences and kz values.
    high = np.zeros((scene.num_baselines),dtype='complex64')
    low = np.zeros((scene.num_baselines),dtype='complex64')
    whigh = np.zeros((scene.num_baselines,3),dtype='complex64')
    wlow = np.zeros((scene.num_baselines,3),dtype='complex64')
    gammahv = np.zeros((scene.num_baselines),dtype='complex64')
    gammahh = np.zeros((scene.num_baselines),dtype='complex64')
    kz = np.zeros((scene.num_baselines),dtype='float32')
    inc = np.ones((scene.num_baselines),dtype='float32')*scene.inc[az,rng]
    ground = np.zeros((scene.num_baselines),dtype='complex64')
    groundalt = np.zeros((scene.num_baselines),dtype='complex64')
    volindex = np.zeros((scene.num_baselines),dtype='bool')

    groundmag = 0.9

    for bl in range(scene.num_baselines):
        (t1,t2) = mb_tr_index(bl)
        kz[bl] = (scene.kza[t2,az,rng] - scene.kza[t1,az,rng])

        gammatemp = np.zeros((2,2,2),dtype='complex')
        gammatemp[0,:,:] = scene.pdcoh[bl,0,az,rng]
        gammatemp[1,:,:] = scene.pdcoh[bl,1,az,rng]
        groundtemp, groundalttemp, volindextemp = kapok.topo.groundsolver(gammatemp, kz=kz[bl], groundmag=groundmag, returnall=True, silent=True)
        ground[bl] = groundtemp[0,0]
        groundalt[bl] = groundalttemp[0,0]
        volindex[bl] = volindextemp[0,0]
        high[bl] = np.where(volindex[bl],scene.pdcoh[bl,1,az,rng],scene.pdcoh[bl,0,az,rng])
        low[bl] = np.where(volindex[bl],scene.pdcoh[bl,0,az,rng],scene.pdcoh[bl,1,az,rng])
        whigh[bl] = np.where(volindex[bl],scene.pdweights[bl,1,az,rng],scene.pdweights[bl,0,az,rng])
        wlow[bl] = np.where(volindex[bl],scene.pdweights[bl,0,az,rng],scene.pdweights[bl,1,az,rng])
        gammahv[bl] = scene.coh('hv',bl=bl,pix=(az,rng))
        gammahh[bl] = scene.coh('hh',bl=bl,pix=(az,rng))

    psep = np.abs(np.angle(high*np.conj(low))/kz)

    def remove_negatives(ph, kz):
        ind = (ph < 0)
        while np.any(ind):
            ph[ind] += np.abs(2*np.pi/kz[ind])
            ind = (ph < 0)
        return ph


    ph = remove_negatives(np.angle(high*np.conj(ground))/kz,kz)
    ph_alt = remove_negatives(np.angle(low*np.conj(groundalt))/kz,kz)
    ph_2pi = remove_negatives(ph + np.abs(2*np.pi/kz),kz)
    ph_alt_2pi = remove_negatives(ph_alt + np.abs(2*np.pi/kz),kz)
    ph_4pi = remove_negatives(ph + np.abs(4*np.pi/kz),kz)
    ph_alt_4pi = remove_negatives(ph_alt + np.abs(4*np.pi/kz),kz)


    # Top left plot is high coherence magnitude vs. kz.
    for n, kzv in enumerate(kz):
        ax[0,0].plot(np.abs(kz[n]), np.abs(high[n]),marker=next(marker),markersize=next(markersize),label='Observed')

    # RVoG Curve Fitting for Coherences.
    if refheight is not None:
        hvvec = np.linspace(refheight, refheight, num=2)
    else:
        hvvec = np.linspace(np.nanmin(ph), 80, num=101)
    extvec = np.linspace(1e-2, 3, num=101) / nptodb
    mincost = 1e10
    for hvtest in hvvec:
        for exttest in extvec:
            gv = kapok.rvog.rvogfwdvol(hvtest, exttest, inc, kz)
            tdf = np.abs(high / (gv*ground))
            tdf = np.clip(tdf, 0.0, 1.0)
            tdf = np.nanmean(tdf)

            cost = np.nanmean(np.square(np.abs(high - tdf*gv*ground)))
            # cost = np.nanmean(np.square(np.abs(high) - np.abs(tdf*gv*ground)))

            if cost < mincost:
                mincost = cost
                hvfit = hvtest
                extfit = exttest
                tdffit = tdf

    # print('ref height: '+str(refheight)+' m')
    # print('hv fitted: '+str(hvfit)+' m')
    # print('ext fitted: '+str(extfit*nptodb)+' dB/m')
    # print('tdf fitted: '+str(tdffit))
    gv = kapok.rvog.rvogfwdvol(hvfit, extfit, inc, kz)*tdffit
    phfit = remove_negatives(np.angle(gv)/kz,kz)

    indsort = np.argsort(np.abs(kz))

    ax[0,0].plot(np.abs(kz[indsort]), np.abs(gv[indsort]), '-', color='ForestGreen', label='Fitted RVoG')

    ax[0,0].set_xlabel(r'$|k_{z}|$ (rad/m)')
    ax[0,0].set_ylabel('High Coherence Magnitude')
    ax[0,0].set_ylim([0, 1])


    # Top middle plot is phase height vs. kz.
    for n, kzv in enumerate(kz):
        ax[0,1].plot(np.abs(kz[n]), ph[n], marker=next(marker), markersize=next(markersize), label='Standard PCH')
    for n, kzv in enumerate(kz):
        ax[0,1].plot(np.abs(kz[n]), ph_alt[n], color='#1F618D', marker=next(marker), markersize=next(markersize), label='Alt. Ground PCH')
    for n, kzv in enumerate(kz):
        ax[0,1].plot(np.abs(kz[n]), ph_2pi[n], color='#CB4335', marker=next(marker), markersize=next(markersize), label='S+2pi PCH')
    for n, kzv in enumerate(kz):
        ax[0,1].plot(np.abs(kz[n]), ph_alt_2pi[n], color='#3498DB', marker=next(marker), markersize=next(markersize), label='A+2pi PCH')
    for n, kzv in enumerate(kz):
        ax[0,1].plot(np.abs(kz[n]), ph_4pi[n], color='#F1948A', marker=next(marker), markersize=next(markersize), label='S+4pi PCH')
    for n, kzv in enumerate(kz):
        ax[0,1].plot(np.abs(kz[n]), ph_alt_4pi[n], color='#AED6F1', marker=next(marker), markersize=next(markersize), label='A+4pi PCH')

    ymax = np.max(ph[np.abs(kz) > 0.03]) + 15
    if refheight is not None:
        ax[0,1].plot([np.min(np.abs(kz)), np.max(np.abs(kz))], [refheight, refheight], '--', color='Green', label='Reference')
        if ymax < refheight+15:
            ymax = refheight+15

    ax[0,1].plot(np.abs(kz[indsort]), phfit[indsort], '-', color='ForestGreen', label='Fitted RVoG')

    ax[0,1].plot(np.abs(kz[indsort]), np.pi/np.abs(kz[indsort]), '--', color='Red', label=r'$h_{\pi}$')
    ax[0,1].plot(np.abs(kz[indsort]), 2*np.pi/np.abs(kz[indsort]), '--', color='Orange', label=r'$h_{2\pi}$')

    ax[0,1].set_xlabel(r'$|k_{z}|$ (rad/m)')
    ax[0,1].set_ylabel('Phase Center Height (m)')
    ax[0,1].set_ylim([0, ymax])


    # Top right plot is phase separation vs. kz.
    for n, kzv in enumerate(kz):
        ax[0,2].plot(np.abs(kz[n]),psep[n], marker=next(marker), markersize=next(markersize), color='k', label='Phase Separation (High-Low)')

    psep_hv_hh = np.abs(np.angle(gammahv*np.conj(gammahh))/kz)

    for n, kzv in enumerate(kz):
        ax[0,2].plot(np.abs(kz[n]),psep_hv_hh[n], marker=next(marker), markersize=next(markersize), color='b', label='Phase Separation (HV-HH)')
    ax[0,2].set_xlabel(r'$|k_{z}|$ (rad/m)')
    ax[0,2].set_ylabel('Phase Center Separation (m)')
    ax[0,2].set_ylim([0, np.max(psep[np.abs(kz) > 0.03]) + 5])


    # Bottom left plot is RVoG forest height vs. kz.
    import kapok.rvog
    inc = np.ones(high.shape, dtype='float32') * scene.inc[az,rng]

    high_rvog = np.concatenate((high, high, high, low, low, low))
    ground_rvog = np.concatenate((ground, ground, ground, groundalt, groundalt, groundalt))
    inc_rvog = np.concatenate((inc, inc, inc, inc, inc, inc))
    kz_rvog = np.concatenate((kz, kz, kz, kz, kz, kz))
    hv_min_rvog = np.concatenate((ph, ph_2pi, ph_4pi, ph_alt, ph_alt_2pi, ph_alt_4pi))
    vecmax = np.ones(scene.num_baselines,dtype='float32') * 1e6
    hv_max_rvog = np.concatenate((ph_2pi, ph_4pi, vecmax, ph_alt_2pi, ph_alt_4pi, vecmax)) - 1
    hv_max_rvog[hv_max_rvog > 70] = 70

    setext = False
    if setext:
        ext = np.ones(high_rvog.shape, dtype='float32') * 0.2/nptodb
        hv, tdf, conv = kapok.rvog.rvoginv(high_rvog, ground_rvog, inc_rvog, kz_rvog, limit2pi=False, hv_min=hv_min_rvog, hv_max=hv_max_rvog, ext=ext, silent=True)
    else:
        hv, ext, conv = kapok.rvog.rvoginv(high_rvog, ground_rvog, inc_rvog, kz_rvog, limit2pi=False, hv_min=hv_min_rvog, hv_max=hv_max_rvog, silent=True)
        voltemp = kapok.rvog.rvogfwdvol(hv, ext, inc_rvog, kz_rvog)
        tdf = np.abs(high_rvog / voltemp)


    for n, kzv in enumerate(kz):
        ax[1,0].plot(np.abs(kz[n]),hv[n],'k', marker=next(marker), markersize=next(markersize), label='Standard HV')
    for n, kzv in enumerate(kz):
        ax[1,0].plot(np.abs(kz[n]),hv[n+scene.num_baselines*3],color='#1F618D', marker=next(marker), markersize=next(markersize), label='Alt. Ground HV')
    for n, kzv in enumerate(kz):
        ax[1,0].plot(np.abs(kz[n]),hv[n+scene.num_baselines],color='#CB4335', marker=next(marker), markersize=next(markersize), label='S+2pi HV')
    for n, kzv in enumerate(kz):
        ax[1,0].plot(np.abs(kz[n]),hv[n+scene.num_baselines*4],color='#3498DB', marker=next(marker), markersize=next(markersize), label='A+2pi HV')
    for n, kzv in enumerate(kz):
        ax[1,0].plot(np.abs(kz[n]),hv[n+scene.num_baselines*2],color='#F1948A', marker=next(marker), markersize=next(markersize), label='S+4pi HV')
    for n, kzv in enumerate(kz):
        ax[1,0].plot(np.abs(kz[n]),hv[n+scene.num_baselines*5],color='#AED6F1', marker=next(marker), markersize=next(markersize), label='A+4pi HV')

    ymax = np.max(hv[0:scene.num_baselines][np.abs(kz) > 0.03]) + 15
    if refheight is not None:
        ax[1,0].plot([np.min(np.abs(kz)), np.max(np.abs(kz))], [refheight, refheight], '--', color='Green', label='Reference')
        if ymax < refheight+15:
            ymax = refheight+15

    ax[1,0].set_xlabel(r'$|k_{z}|$ (rad/m)')
    ax[1,0].set_ylabel(r'$h_{v}$ (m)')
    ax[1,0].set_ylim([0, ymax])


    # Bottom middle plot is RVoG extinction vs. kz.
    ax[1,1].plot(np.abs(kz),ext[0:scene.num_baselines]*nptodb,'+k',label=r'Standard $\sigma_{x}$')
    ax[1,1].plot(np.abs(kz),ext[scene.num_baselines*3:scene.num_baselines*4]*nptodb,'+',color='#1F618D',label=r'Alt. Ground $\sigma_{x}$')
    ax[1,1].plot(np.abs(kz),ext[scene.num_baselines:scene.num_baselines*2]*nptodb,'+',color='#CB4335',label=r'S+2pi $\sigma_{x}$')
    ax[1,1].plot(np.abs(kz),ext[scene.num_baselines*4:scene.num_baselines*5]*nptodb,'+',color='#3498DB',label=r'A+2pi $\sigma_{x}$')
    ax[1,1].plot(np.abs(kz),ext[scene.num_baselines*2:scene.num_baselines*3]*nptodb,'+',color='#F1948A',label=r'S+4pi $\sigma_{x}$')
    ax[1,1].plot(np.abs(kz),ext[scene.num_baselines*5:scene.num_baselines*6]*nptodb,'+',color='#AED6F1',label=r'A+4pi $\sigma_{x}$')

    ax[1,1].set_xlabel(r'$|k_{z}|$ (rad/m)')
    ax[1,1].set_ylabel(r'$\sigma_{x}$ (dB/m)')
    ax[1,1].set_ylim([0, 1.0])


    # Bottom right plot is RVoG extinction vs. kz.
    ax[1,2].plot(np.abs(kz),tdf[0:scene.num_baselines],'+k',label=r'Standard $\alpha_{vt}$')
    ax[1,2].plot(np.abs(kz),tdf[scene.num_baselines*3:scene.num_baselines*4],'+',color='#1F618D',label=r'Alt. Ground $\alpha_{vt}$')
    ax[1,2].plot(np.abs(kz),tdf[scene.num_baselines:scene.num_baselines*2],'+',color='#CB4335',label=r'S+2pi $\alpha_{vt}$')
    ax[1,2].plot(np.abs(kz),tdf[scene.num_baselines*4:scene.num_baselines*5],'+',color='#3498DB',label=r'A+2pi $\alpha_{vt}$')
    ax[1,2].plot(np.abs(kz),tdf[scene.num_baselines*2:scene.num_baselines*3],'+',color='#F1948A',label=r'S+4pi $\alpha_{vt}$')
    ax[1,2].plot(np.abs(kz),tdf[scene.num_baselines*5:scene.num_baselines*6],'+',color='#AED6F1',label=r'A+4pi $\alpha_{vt}$')

    ax[1,2].set_xlabel(r'$|k_{z}|$ (rad/m)')
    ax[1,2].set_ylabel(r'$\alpha_{vt}$')
    # ax[1,2].set_ylim([0, 1.0])



    # ax[0,3].plot(np.abs(kz), mu_hv, '+k', label='Mu')
    # ax[0,3].plot(np.abs(kz), mu_hv_alt, '+b', label='Mu')

    weights = np.abs(kz) / np.sqrt((1-np.square(np.abs(high)))/np.square(np.abs(high)))
    ecc = np.sqrt(1 - np.square(np.abs(scene.pdcohminor[:,0,az,rng] - scene.pdcohminor[:,1,az,rng]) / np.abs(scene.pdcoh[:,0,az,rng] - scene.pdcoh[:,1,az,rng])))

    # weights = np.abs(high) / np.sqrt(1 - np.square(np.abs(high))) * np.abs(kz)
    # ax[0,3].plot(np.abs(kz), weights, '+k', label='Weights')
    # ax[0,3].plot(np.abs(kz), np.angle(ground), '+k')
    # ax[0,3].plot(np.abs(kz), np.angle(groundalt), '+b')
#    groundz = np.angle(ground)/kz
#    groundaltz = np.angle(groundalt)/kz
#    ax[0,3].plot(np.abs(kz), groundz, '+k')
#    ax[0,3].plot(np.abs(kz), groundaltz, '+b')
#    ax[0,3].set_ylim([-50,50])
#    gz = np.angle(gammahh)/kz
#    ax[0,3].plot(np.abs(kz), gz, '+k')
#    ax[0,3].plot(np.abs(kz), gz + 2*np.pi/np.abs(kz), '+r')
#    ax[0,3].plot(np.abs(kz), gz - 2*np.pi/np.abs(kz), '+b')
#    ax[0,3].set_ylim([-np.pi/0.05, np.pi/0.05])


    mu_hv = calcmu(gammahv, high, ground)
    mu_hv_alt = calcmu(gammahv, low, groundalt)

    mu_low = calcmu(low, high, ground)
    mu_low_alt = calcmu(high, low, groundalt)


    def freemantwocomponent(cov):
        """ freemantwocomponent

        Performs the two component decomposition proposed by Freeman in:

        A. Freeman, "Fitting a Two-Component Scattering Model
        to Polarimetric SAR Data From Forests," IEEE TGARS, 2007.

        Attempts to decompose the covariance matrix into a ground component
        and a volume component.

        Input Arguments:

        - cov, a Corpa covariance matrix object to be decomposed.
        - tr, the track number to decompose.

        Returns:

        - pg, the estimated power of the ground scattering component.
        - pv, the estimated power of the volume scattering component.

        """
        z1 = cov[:,:,0,0] - cov[:,:,2,2]
        z1[np.abs(z1) <= 0] = 1e-10

        z2 = 2*cov[:,:,1,1] + cov[:,:,0,2] - cov[:,:,0,0]

        z3 = z2 / z1
        z3[z3 == 0] = 1e-10

        y = -1 * np.imag(z3) * (1 + 2*np.real(z3)) / np.square(np.abs(z3))
        x = 1 + (y * np.real(z3) / np.imag(z3))

        alpha = x + 1j*y
        fg = z1 / (1 - np.square(np.abs(alpha)))
        fv = cov[:,:,0,0] - fg
        rho = 1 - (2 * cov[:,:,1,2] / fv)

        pg = fg * (1 + np.square(np.abs(alpha)))
        pv = fv * (3 - rho)

        return pg, pv


#    mu_low = 20*np.log10(calcmu(low, high, ground))
#    mu_low_alt = 20*np.log10(calcmu(high, low, groundalt))
#    mu_hv = 20*np.log10(calcmu(gammahv, high, ground))
#    mu_hv_alt = 20*np.log10(calcmu(gammahv, low, groundalt))
#    mu_hh = 20*np.log10(calcmu(gammahh, high, ground))
#    mu_hh_alt = 20*np.log10(calcmu(gammahh, low, groundalt))

#    delta = np.zeros((kz.shape[0],3),dtype='float32')
#
#    for n, kzv in enumerate(kz):
#        # ax[0,3].plot(np.abs(kz[n]),mu_low[n],'k', marker=next(marker), markersize=next(markersize), label='Standard Mu Low')
#        # ax[0,3].plot(np.abs(kz[n]),mu_hv[n],'k', marker=next(marker), markersize=next(markersize), label='Standard Mu Low')
#
#        # BNMAS
#        from kapok.lib import mb_cov_index, makehermitian
#        import scipy.linalg
#        row, col = mb_cov_index(n)
#        t11 = scene.cov[az,rng,row:row+scene.num_pol,row:row+scene.num_pol]
#        t22 = scene.cov[az,rng,col:col+scene.num_pol,col:col+scene.num_pol]
#        om12 = scene.cov[az,rng,row:row+scene.num_pol,col:col+scene.num_pol]
#
#        t = 0.5*(t11 + t22)
#        t = makehermitian(t)
#        del t11, t22
#
#        t = np.matrix(t)
#        om12 = np.matrix(om12)
#
#        cinvsqrt = np.matrix(scipy.linalg.fractional_matrix_power(t, -0.5))
#
#        # Whitened interferometric coherence matrix
#        om12w = cinvsqrt*om12*cinvsqrt
#
#        h1 = (om12w + om12w.H)/2
#        h2 = (om12w - om12w.H)/2j
#
#        a = 4*np.trace(h1**2)
#        b = -8*np.trace(h1*h2)
#        c = 4*np.trace(h2**2)
#        d = 4*np.trace(h1)
#        e = -4*np.trace(h2)
#        f = 3
#
#        x = (4*a*f - np.square(d) - 4*c*f + np.square(e))/2
#        y = 2*b*f - d*e
#
#        x = np.real(x)
#        y = np.real(y)
#
#        theta = (np.arctan2(y,x) + np.pi)/2
#
#        rho = -2*f / (d*np.cos(theta) + e*np.sin(theta))
#
#        z = rho*(np.cos(theta) + 1j*np.sin(theta))
#
#        om12double = (z*om12w - np.conj(z)*om12w.H - np.eye(scene.num_pol))/(2*z)
#
#
#        # Single phase centre case:
#        gammasingle = np.trace(om12w) / 3
#        om12single = np.matrix(np.eye(scene.num_pol) * gammasingle)
#
#
#        # Triple phase centre case:
#        from corpa.polinsar.cohopt import joint_diag
##        w, v = scipy.linalg.eig(om12w)
##        w = np.matrix(w)
##        v = np.matrix(v)
##        d = np.matrix(np.eye(scene.num_pol))
##        d[0,0] = w[0,0]
##        d[1,1] = w[0,1]
##        d[2,2] = w[0,2]
##        om12triple = v * d * v.H
#        V, D = joint_diag(om12w)
#        D = np.matrix(np.diagflat(np.diag(D)))
#        om12triple = V * D * V.H
#
#
#        def frobnorm(m):
#            return np.real(np.trace(m.H * m))
#
#        om12wnorm = frobnorm(om12w)
#
#        delta[n,0] = frobnorm(om12single-om12w)# / om12wnorm
#        delta[n,1] = frobnorm(om12double-om12w)# / om12wnorm
#        delta[n,2] = frobnorm(om12triple-om12w)# / om12wnorm
#
#        ax[0,3].plot(np.abs(kz[n]),delta[n,1],'k', marker=next(marker), markersize=next(markersize), label='Double PC Matrix Distance')
#
#
#    for n, kzv in enumerate(kz):
#        ax[0,3].plot(np.abs(kz[n]),delta[n,0],'b', marker=next(marker), markersize=next(markersize), label='Single PC Matrix Distance')
#
#    for n, kzv in enumerate(kz):
#        ax[0,3].plot(np.abs(kz[n]),delta[n,2],'r', marker=next(marker), markersize=next(markersize), label='Triple PC Matrix Distance')

    showmulow = False
    showmuhv = False

    if showmulow:
        for n, kzv in enumerate(kz):
            ax[0,3].plot(np.abs(kz[n]),mu_low[n],'k', marker=next(marker), markersize=next(markersize), label='Mu HV')

        for n, kzv in enumerate(kz):
            ax[0,3].plot(np.abs(kz[n]),mu_low_alt[n],'b', marker=next(marker), markersize=next(markersize), label='Alternate Mu HV')

    if showmuhv:
        for n, kzv in enumerate(kz):
            ax[0,3].plot(np.abs(kz[n]),mu_hv[n],'k', marker=next(marker), markersize=next(markersize), label='Mu HV')

        for n, kzv in enumerate(kz):
            ax[0,3].plot(np.abs(kz[n]),mu_hv_alt[n],'b', marker=next(marker), markersize=next(markersize), label='Alternate Mu HV')


    if False:
        for n, kzv in enumerate(kz):
            #ax[0,3].plot(np.abs(kz[n]),np.angle(gammahv[n])/kz[n],'k',marker=next(marker),markersize=next(markersize))
            ax[0,3].plot(np.abs(kz[n]),np.angle(ground[n])/kz[n],'k',marker=next(marker),markersize=next(markersize))

    if False:
        for n, kzv in enumerate(kz):
            ax[0,3].plot(np.abs(kz[n]),ecc[n],marker=next(marker),markersize=next(markersize))



    # ax[0,3].plot(np.abs(kz), mu_low, '+k')
    # ax[0,3].plot(np.abs(kz), mu_low_alt, '+b')
#    pg = np.zeros((scene.num_tracks+1), dtype='complex64')
#    pv = np.zeros((scene.num_tracks+1), dtype='complex64')
#    for tr in range(scene.num_tracks+1):
#        cov = scene.cov[az-1:az+1, rng-1:rng+1, tr*3:tr*3 + 3, tr*3:tr*3 + 3]
#        if tr == 0:
#            covsum = cov.copy()
#        elif tr < scene.num_tracks:
#            covsum += cov
#        else:
#            cov = covsum / scene.num_tracks
#        pgtemp, pvtemp = freemantwocomponent(cov)
#        pg[tr] = pgtemp[1,1]
#        pv[tr] = pvtemp[1,1]

    # ax[0,3].plot(range(scene.num_tracks+1), pg, '+b')
    # ax[0,3].plot(range(scene.num_tracks+1), pv, '+g')



    #ax[1,3].plot(np.abs(kz), ecc, '+k',label='Eccentricity')

    #ax[1,3].scatter(ecc,weights,s=50,c=np.abs(kz),cmap='viridis')

    # DEVELOP MULTI-BASELINE SEQUENTIAL RVOG
    hv_max = 70
    hv_min = 1.0
    hpi_tol = 0.90

    coh_seq = np.ones(kz.shape,dtype='complex64') * -99
    hv_seq = np.ones(kz.shape,dtype='float32') * -99
    kz_seq = np.ones(kz.shape,dtype='float32') * -99
    whigh_seq = np.ones((kz.shape[0],3),dtype='complex64') * -99
    hv_options = np.zeros((6),dtype='float32')

    for n, bl in enumerate(np.argsort(np.abs(kz))):
        kz_seq[n] = kz[bl]
        pch = ph[bl]

        if np.abs(kz[bl]) < (np.pi/100): # Baseline too small.
            coh_seq[n] = -1
            hv_seq[n] = -1
        else:
            ind = (hv_seq > 0) # compare to previous solutions, if available
            if np.any(ind):
                if np.pi/kz[bl] < hv_max:
                    # Use the polarimetric weight vectors to help choose the ground.
                    wtemp = np.nanmean(whigh_seq[ind])
                    diststandard = np.sum(np.square(np.abs(whigh[bl] - wtemp)))
                    distalt = np.sum(np.square(np.abs(wlow[bl] - wtemp)))

                    if diststandard < distalt:
                        # Primary ground phase solution.
                        hv_options[3:6] = 0
                        hv_options[0] = hv[bl] # standard solution
                        hv_options[1] = hv[bl+(scene.num_baselines)] # standard + 2pi
                        hv_options[2] = hv[bl+(scene.num_baselines*2)] # standard + 4pi
                    else:
                        # Secondary ground phase solution.
                        hv_options[0:3] = 0
                        hv_options[3] = hv[bl+(scene.num_baselines*3)] # alt ground
                        hv_options[4] = hv[bl+(scene.num_baselines*4)] # alt + 2pi
                        hv_options[5] = hv[bl+(scene.num_baselines*5)] # alt + 4pi

                    hv_options[0] = hv[bl] # standard solution
                    hv_options[1] = hv[bl+(scene.num_baselines)] # standard + 2pi
                    hv_options[2] = hv[bl+(scene.num_baselines*2)] # standard + 4pi
                    hv_options[3] = hv[bl+(scene.num_baselines*3)] # alt ground
                    hv_options[4] = hv[bl+(scene.num_baselines*4)] # alt + 2pi
                    hv_options[5] = hv[bl+(scene.num_baselines*5)] # alt + 4pi
                else:
                    # Could be either solution.  Don't have a good idea of the high coherence polarization weight vector yet.
                    hv_options[0] = hv[bl] # standard solution
                    hv_options[1] = hv[bl+(scene.num_baselines)] # standard + 2pi
                    hv_options[2] = hv[bl+(scene.num_baselines*2)] # standard + 4pi
                    hv_options[3] = hv[bl+(scene.num_baselines*3)] # alt ground
                    hv_options[4] = hv[bl+(scene.num_baselines*4)] # alt + 2pi
                    hv_options[5] = hv[bl+(scene.num_baselines*5)] # alt + 4pi

                dist = np.abs(hv_options - np.nanmean(hv_seq[ind]))
                dist[hv_options > hv_max] = 1e10
                dist[hv_options < hv_min] = 1e10

                bestdist = np.min(dist)
                closest = np.argmin(dist)

                if bestdist < 1e9:
                    hv_seq[n] = hv_options[closest]
                    if closest >= 3:
                        coh_seq[n] = low[bl]
                        whigh_seq[n] = wlow[bl]
                    else:
                        coh_seq[n] = high[bl]
                        whigh_seq[n] = whigh[bl]
            elif (hv[bl] < hv_max) and (hv[bl] > hv_min) and ((pch < (hpi_tol*np.pi/np.abs(kz[bl]))) or (n > 1)):
                hv_seq[n] = hv[bl]
                coh_seq[n] = high[bl]
                whigh_seq[n] = whigh[bl]
            else:
                pass

    # print(kz_seq)
    # print(hv_seq)


    if True:
        for n, kzv in enumerate(kz):
            ax[0,3].plot(np.abs(kz_seq[n]),np.abs(kz_seq[n])*hv_seq[n]/2,marker=next(marker),markersize=next(markersize))
        ax[0,3].plot(np.abs(kz),np.abs(kz)*refheight/2,'g-')


    ind = (hv_seq > 0)
    ax[1,3].plot(np.abs(kz_seq[ind]),hv_seq[ind],'+k',label=r'Reasonable Solutions')
    ymax = np.max(hv[0:scene.num_baselines][np.abs(kz) > 0.03]) + 15
    if refheight is not None:
        ax[1,3].plot([np.min(np.abs(kz)), np.max(np.abs(kz))], [refheight, refheight], '--', color='Green', label='Reference')
        if ymax < refheight+15:
            ymax = refheight+15
        if ymax < np.max(hv_seq)+15:
            ymax = np.max(hv_seq)+15

    overallhv = np.nanmean(hv_seq[ind])
    seqhv = np.cumsum(hv_seq[ind])/np.linspace(1,hv_seq[ind].shape[0],num=hv_seq[ind].shape[0])
    ax[1,3].plot([np.min(np.abs(kz)), np.max(np.abs(kz))], [overallhv, overallhv], '--', color='Blue', label='Mean of Reasonable Solutions')
    ax[1,3].plot(np.abs(kz_seq)[ind], seqhv, '--', color='Red', label='Running Mean of Reasonable Solutions')

    ax[1,3].set_xlabel(r'$|k_{z}|$ (rad/m)')
    ax[1,3].set_ylabel(r'$h_{v}$ (m)')
    ax[1,3].set_ylim([0, ymax])

    print('Reference Height: '+str(np.round(refheight,decimals=2))+' m.')
    print('MBS RVoG Solution: '+str(np.round(overallhv,decimals=2))+' m.')
    print('MBS RVoG Error: '+str(np.round(overallhv-refheight,decimals=2))+' m.')
    print('MBS RVoG Percent Error: '+str(np.round((overallhv-refheight)/refheight*100,decimals=2))+'%')




    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def binplot(x, y, mask=None, xlim=None, ylim=None, xbins=50, xname=None,
            yname=None, mincount=10, figsize=None, savefile=None, dpi=125,
            simline=False):
    """Plot the average and standard deviation of one parameter as a function
    of another.

    The mean value of the y data in each x bin is plotted with a solid
    line.  The mean plus and minus the standard deviation is shown as a
    shaded region.

    Arguments:
        x (array): Data array for the x axis.
        y (array): Data array for the y axis.
        mask (array): Boolean array.  Where mask == True, the data will be
            included in the plot.  Where mask == False, the data will be
            excluded.
        xlim (tuple): X-axis limits.  Tuple of (min, max).  Default: Min and
            max of data in x.
        ylim (tuple): Y-axis limits.  Tuple of (min, max).  Default: Min and
            max of data in y.
        xbins (int): Number of bins for the x axis.  Default: 50.
        xname (str): The x-axis label.
        yname (str): The y-axis label.
        mincount (int): Minimum number of samples in a bin for the bin to
            be plotted.  Default: 10.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        savefile (str): Path and filename to save the figure.  Default: None.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            200.

    """
    if mask is None:
        mask = np.ones(x.shape, dtype='bool')

    ind = mask & np.isfinite(x) & np.isfinite(y)

    if np.any(ind):
        x = x[ind]
        y = y[ind]
    else:
        print('kapok.plot.binplot | No valid data in masked region.  Aborting.')
        return

    if xlim is None:
        xlim = (np.nanmin(x), np.nanmax(x))

    binwidth = (xlim[1] - xlim[0]) / xbins
    bincenters = np.linspace(xlim[0]+(binwidth/2),xlim[1]-(binwidth/2),num=xbins)

    y_avg = np.zeros(xbins, dtype=y.dtype)
    y_std = np.zeros(xbins, dtype=y.dtype)

    for num in range(xbins):
        lowerbound = num*binwidth + xlim[0]
        upperbound = (num+1)*binwidth + xlim[0]

        ind = (x >= lowerbound) & (x < upperbound)

        if np.sum(ind) >= mincount:
            y_avg[num] = np.nanmean(y[ind])
            y_std[num] = np.nanstd(y[ind])
        else:
            y_avg[num] = np.nan
            y_std[num] = np.nan

    if figsize is None:
        fig, ax = plt.subplots(1)
    else:
        fig, ax = plt.subplots(1, figsize=figsize)

    if simline:
        plt.plot(xlim,xlim,'k--')

    ax.fill_between(bincenters, y_avg+y_std, y_avg-y_std, facecolor='blue', alpha=0.1)
    plt.plot(bincenters, y_avg, linewidth=3)
    plt.xlim(xlim)

    if xname is not None:
        plt.xlabel(xname)

    if ylim is not None:
        plt.ylim(ylim)

    if yname is not None:
        plt.ylabel(yname)

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def doublebinplot(x, y1, y2, mask=None, xlim=None, ylim=None, xbins=50, xname=None,
            yname=None, y1label=None, y2label=None, mincount=10, figsize=None, savefile=None,
            dpi=125, simline=False):
    """Plot the average and standard deviation of one parameter as a function
    of another.

    The mean value of the y data in each x bin is plotted with a solid
    line.  The mean plus and minus the standard deviation is shown as a
    shaded region.

    Arguments:
        x (array): Data array for the x axis.
        y1 (array): Data array for the y axis.  First series.
        y2 (array): Data array for the y axis.  Second series.
        mask (array): Boolean array.  Where mask == True, the data will be
            included in the plot.  Where mask == False, the data will be
            excluded.
        xlim (tuple): X-axis limits.  Tuple of (min, max).  Default: Min and
            max of data in x.
        ylim (tuple): Y-axis limits.  Tuple of (min, max).  Default: Min and
            max of data in y.
        xbins (int): Number of bins for the x axis.  Default: 50.
        xname (str): The x-axis label.
        yname (str): The y-axis label.
        mincount (int): Minimum number of samples in a bin for the bin to
            be plotted.  Default: 10.
        figsize (tuple): Tuple containing the figure size in the (x,y)
            dimensions (in inches).  Passed to the plt.figure() call.
        savefile (str): Path and filename to save the figure.  Default: None.
        dpi: DPI value for the saved plot, if savefile is specified.  Default:
            200.

    """
    if mask is None:
        mask = np.ones(x.shape, dtype='bool')

    ind = mask & np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)

    if np.any(ind):
        x = x[ind]
        y1 = y1[ind]
        y2 = y2[ind]
    else:
        print('kapok.plot.binplot | No valid data in masked region.  Aborting.')
        return

    if xlim is None:
        xlim = (np.nanmin(x), np.nanmax(x))

    binwidth = (xlim[1] - xlim[0]) / xbins
    bincenters = np.linspace(xlim[0]+(binwidth/2),xlim[1]-(binwidth/2),num=xbins)

    y_avg = np.zeros((2, xbins), dtype=y1.dtype)
    y_std = np.zeros((2, xbins), dtype=y1.dtype)

    for num in range(xbins):
        lowerbound = num*binwidth + xlim[0]
        upperbound = (num+1)*binwidth + xlim[0]

        ind = (x >= lowerbound) & (x < upperbound)

        if np.sum(ind) >= mincount:
            y_avg[0, num] = np.nanmean(y1[ind])
            y_std[0, num] = np.nanstd(y1[ind])
            y_avg[1, num] = np.nanmean(y2[ind])
            y_std[1, num] = np.nanstd(y2[ind])
        else:
            y_avg[:, num] = np.nan
            y_std[:, num] = np.nan

    if figsize is None:
        fig, ax = plt.subplots(1)
    else:
        fig, ax = plt.subplots(1, figsize=figsize)

    if simline:
        plt.plot(xlim,xlim,'k--')

    ax.fill_between(bincenters, y_avg[0]+y_std[0], y_avg[0]-y_std[0], facecolor='C0', alpha=0.1)
    ax.fill_between(bincenters, y_avg[1]+y_std[1], y_avg[1]-y_std[1], facecolor='C1', alpha=0.1)
    plt.plot(bincenters, y_avg[0], linewidth=3, color='C0', label=y1label)
    plt.plot(bincenters, y_avg[1], linewidth=3, color='C1', label=y2label)

    plt.legend(loc='upper left', frameon=True, fontsize=10)

    plt.xlim(xlim)

    if xname is not None:
        plt.xlabel(xname)

    if ylim is not None:
        plt.ylim(ylim)

    if yname is not None:
        plt.ylabel(yname)

    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
