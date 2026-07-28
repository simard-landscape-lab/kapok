# -*- coding: utf-8 -*-
"""Visualization and display functions.
   
    Author: Michael Denbina
    
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
import numpy as np
import matplotlib.pyplot as plt



def show_linear(data, bounds=None, vmin=None, vmax=None, cmap='viridis',
    cbar=True, cbar_label=None, xlabel='Range Index', ylabel='Azimuth Index', 
    figsize=None, dpi=125, rotate=False, hideaxis=False, savefile=None, **kwargs):
    """Display data in linear units (e.g., tree heights, kz).
    
        Arguments:
            data (array): 2D array containing the values to display.
            bounds (tuple): Tuple containing (azimuth start, azimuth end, range
                start, range end) bounds in that order.  Only the subset of the
                data within bounds will be displayed.  For a full swath subset,
                two element bounds can be given: (azimuth start, azimuth end).
            vmin (float): Minimum value for colormap. Default: None.
            vmax (float): Maximum value for colormap. Default: None.
            cmap: Colormap.  Default: 'viridis'.
            cbar (bool): Set to False to not show colorbar.
            cbar_label (str): Text label on the colorbar.
            xlabel (str): Text label on the x axis.
            ylabel (str): Text label on the y axis.
            figsize (tuple): figsize used to create the matplotlib figure.
            dpi (int): DPI (dots per inch) used in the matplotlib figure.
            rotate (bool): Set to True to display azimuth as x-axis and range
                as y-axis.  Default: Azimuth as y-axis, range as x-axis.
            hideaxis (bool): Set to True to hide the axis ticks and labels.
            savefile (str): If specified, the plotted figure is saved under this
                filename.
            **kwargs: Additional keyword arguments which will be passed
                directly to the matplotlib imshow() function call.
    
    """       
    if data is not None:
        if figsize is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        else:
            plt.figure()
            
        if bounds is not None:
            if len(bounds) == 2:
                bounds = (bounds[0], bounds[1], 0, data.shape[1])
                
            data = data[bounds[0]:bounds[1],bounds[2]:bounds[3]]

        if rotate:
            data = np.flipud(np.transpose(data))
            temp = xlabel
            xlabel = ylabel
            ylabel = temp
            if bounds is not None:
                bounds = (bounds[2], bounds[3], bounds[0], bounds[1])
        
        if bounds is None:           
            plt.imshow(np.real(data), vmin=vmin, vmax=vmax, cmap=cmap, aspect=1, interpolation='nearest', **kwargs)
        else:                
            plt.imshow(np.real(data), extent=(bounds[2],bounds[3],bounds[1],bounds[0]), vmin=vmin, vmax=vmax, cmap=cmap, aspect=1, interpolation='nearest', **kwargs)
        
        if cbar and (cbar_label is not None):
            plt.colorbar(label=cbar_label)
        elif cbar:
            plt.colorbar()
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        
        if hideaxis:
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
        
        if savefile is not None:
            plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)



def show_power(data, bounds=None, vmin=None, vmax=None, cmap='gray',
    cbar=True, cbar_label='Backscatter (dB)', xlabel='Range Index',
    ylabel='Azimuth Index', figsize=None, dpi=125, rotate=False,
    hideaxis=False, savefile=None, **kwargs):
    """Display a power image (e.g., backscatter) in dB units.
    
        Arguments:
            data (array): 2D array containing the power for each pixel,
                in linear units.  Converted to dB for display.
            bounds (tuple): Tuple containing (azimuth start, azimuth end, range
                start, range end) bounds in that order.  Only the subset of the
                data within bounds will be displayed.  For a full swath subset,
                two element bounds can be given: (azimuth start, azimuth end).
            vmin (float): Minimum dB value for colormap. Default: -25.
            vmax (float): Maximum dB value for colormap. Default: 3.
            cmap: Colormap.  Default: 'afmhot'.
            cbar (bool): Set to False to not show colorbar.
            cbar_label (str): Text label on the colorbar.
            xlabel (str): Text label on the x axis.
            ylabel (str): Text label on the y axis.
            figsize (tuple): figsize used to create the matplotlib figure.
            dpi (int): DPI (dots per inch) used in the matplotlib figure.
            rotate (bool): Set to True to display azimuth as x-axis and range
                as y-axis.  Default: Azimuth as y-axis, range as x-axis.
            hideaxis (bool): Set to True to hide the axis ticks and labels.
            savefile (str): If specified, the plotted figure is saved under this
                filename.
            **kwargs: Additional keyword arguments which will be passed
                directly to the matplotlib imshow() function call.
    
    """
    if vmin is None:
        vmin = -25
    
    if vmax is None:
        vmax = 3
    
    if data is not None:
        if figsize is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        else:
            plt.figure()
        
        if bounds is not None:
            if len(bounds) == 2:
                bounds = (bounds[0], bounds[1], 0, data.shape[1])
                
            data = data[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        
        data = np.real(data)
        data[data <= 1e-10] = 1e-10
        data = 10*np.log10(data)
        
        if rotate:
            data = np.flipud(np.transpose(data))
            temp = xlabel
            xlabel = ylabel
            ylabel = temp
            if bounds is not None:
                bounds = (bounds[2], bounds[3], bounds[0], bounds[1])
        
        if bounds is None:
            plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect=1, interpolation='nearest', **kwargs)
        else:
            plt.imshow(data, extent=(bounds[2],bounds[3],bounds[1],bounds[0]), vmin=vmin, vmax=vmax, cmap=cmap, aspect=1, interpolation='nearest', **kwargs)
                   
        if cbar and (cbar_label is not None):
            plt.colorbar(label=cbar_label)
        elif cbar:
            plt.colorbar()
            
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        
        if hideaxis:
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
        
        if savefile is not None:
            plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)   



def show_complex(data, bounds=None, cbar=False, xlabel='Range Index',
    ylabel='Azimuth Index', figsize=None, dpi=125, rotate=False,
    hideaxis=False, savefile=None, **kwargs):
    """Display a complex-valued image (e.g., coherence) using the HSV color
    system, with the phase as the hue, and the magnitude as saturation and
    value.
    
        Arguments:
            data (array): 2D complex array containing coherence or other
                complex values to display.
            bounds (tuple): Tuple containing (azimuth start, azimuth end, range
                start, range end) bounds in that order.  Only the subset of the
                data within bounds will be displayed.    For a full swath
                subset, two element bounds can be given: (azimuth start,
                azimuth end).
            cbar (bool): Set to True to display a colorbar for the phases
                only (the hues, with full saturation and value).
            xlabel (str): Text label on the x axis.
            ylabel (str): Text label on the y axis.       
            figsize (tuple): figsize used to create the matplotlib figure.
            dpi (int): DPI (dots per inch) used in the matplotlib figure.
            rotate (bool): Set to True to display azimuth as x-axis and range
                as y-axis.  Default: Azimuth as y-axis, range as x-axis.
            hideaxis (bool): Set to True to hide the axis ticks and labels.
            savefile (str): If specified, the plotted figure is saved under this
                filename.
            **kwargs: Additional keyword arguments which will be passed
                directly to the matplotlib imshow() function call.
        
    """
    if data is not None:
        # Subsetting
        if bounds is not None:
            if len(bounds) == 2:
                bounds = (bounds[0], bounds[1], 0, data.shape[1])
                
            data = data[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    
        # HSV based on magnitude and phase of data.        
        h = np.clip(np.angle(data)/(2*np.pi) + 0.5,0,1)
        s = np.clip(np.abs(data),0,1)
        v = np.clip(np.abs(data),0,1)
        
        # HSV to RGB Conversion
        red = np.zeros((data.shape[0],data.shape[1]),dtype='float32')
        green = np.zeros((data.shape[0],data.shape[1]),dtype='float32')
        blue = np.zeros((data.shape[0],data.shape[1]),dtype='float32')
        
        ind = (s == 0)
        if np.any(ind):
            red[ind] = v[ind]
            green[ind] = v[ind]
            blue[ind] = v[ind]
        
        a = (h*6.0).astype('int')
        f = (h*6.0) - a
        p = v*(1.0 - s)
        q = v*(1.0 - s*f)
        t = v*(1.0 - s*(1.0-f))
        a = a % 6
        
        ind = (a == 0)
        if np.any(ind):
            red[ind] = v[ind]
            green[ind] = t[ind]
            blue[ind] = p[ind]
            
        ind = (a == 1)
        if np.any(ind):
            red[ind] = q[ind]
            green[ind] = v[ind]
            blue[ind] = p[ind]
            
        ind = (a == 2)
        if np.any(ind):
            red[ind] = p[ind]
            green[ind] = v[ind]
            blue[ind] = t[ind]
            
        ind = (a == 3)
        if np.any(ind):
            red[ind] = p[ind]
            green[ind] = q[ind]
            blue[ind] = v[ind]
            
        ind = (a == 4)
        if np.any(ind):
            red[ind] = t[ind]
            green[ind] = p[ind]
            blue[ind] = v[ind]
            
        ind = (a == 5)
        if np.any(ind):
            red[ind] = v[ind]
            green[ind] = p[ind]
            blue[ind] = q[ind]
        
        
        if figsize is not None:
            plt.figure(figsize=figsize, dpi=dpi)
        else:
            plt.figure()
            
        if rotate:
            red = np.flipud(np.transpose(data))
            green = np.flipud(np.transpose(green))
            blue = np.flipud(np.transpose(blue))
            temp = xlabel
            xlabel = ylabel
            ylabel = temp
            if bounds is not None:
                bounds = (bounds[2], bounds[3], bounds[0], bounds[1])
    
        if bounds is None:
            plt.imshow(np.dstack((red,green,blue)), aspect=1, interpolation='nearest', **kwargs)
        else:            
            plt.imshow(np.dstack((red,green,blue)), extent=(bounds[2],bounds[3],bounds[1],bounds[0]), aspect=1, interpolation='nearest', **kwargs)
    
        if cbar is True:
            if bounds is None:
                plt.imshow(red, aspect=1, interpolation='nearest', cmap='hsv', vmin=-np.pi, vmax=np.pi, alpha=0.0, **kwargs)
            else:
                plt.imshow(red, extent=(bounds[2],bounds[3],bounds[1],bounds[0]), aspect=1, interpolation='nearest', cmap='hsv', vmin=-np.pi, vmax=np.pi, alpha=0.0, **kwargs)
            cbar = plt.colorbar(label='Phase (radians)')
            cbar.set_alpha(1)
            cbar.draw_all()
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        
        if hideaxis:
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
        
        if savefile is not None:
            plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)



def show_paulirgb(cov, bounds=None, vmin=None, vmax=None, xlabel='Range Index', 
             ylabel='Azimuth Index', figsize=None, dpi=125, rotate=False,
             hideaxis=False, savefile=None, **kwargs):
    """Display a Pauli RGB color composite image from a covariance matrix.
        
        Color mapping is as follows.  Red: 0.5*(HH-VV).  Green: 2*HV.
        Blue: 0.5*(HH+VV).
    
        Arguments:
            cov (array): Array containing a single track's 
                polarimetric covariance matrix with dimensions (az, rng, 3, 3).
            bounds (tuple): Tuple containing (azimuth start, azimuth end, range
                start, range end) bounds in that order.  Only the subset of the
                data within bounds will be displayed.  For a full swath subset,
                two element bounds can be given: (azimuth start, azimuth end).
            vmin (int): Minimum value in dB of color range. Default: -25.
            vmax (int): Maximum value in dB of color range. Default: 3.
            xlabel (str): Text label on the x axis.
            ylabel (str): Text label on the y axis.       
            figsize (tuple): figsize used to create the matplotlib figure.
            dpi (int): DPI (dots per inch) used in the matplotlib figure.
            rotate (bool): Set to True to display azimuth as x-axis and range
                as y-axis.  Default: Azimuth as y-axis, range as x-axis.
            hideaxis (bool): Set to True to hide the axis ticks and labels.
            savefile (str): If specified, the plotted figure is saved under
                this filename.
            **kwargs: Additional keyword arguments which will be passed
                directly to the matplotlib imshow() function call.
        
    """
    if vmin is None:
        vmin = -25
    
    if vmax is None:
        vmax = 3
    
    if bounds is not None:
        if len(bounds) == 2:
            bounds = (bounds[0], bounds[1], 0, cov.shape[1])
            
        cov = cov[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        
    rgb = np.zeros((cov.shape[0],cov.shape[1],3))

    # Red: (HH-VV)/2
    w = np.array([1,0,-1]/np.sqrt(2), dtype='complex64')
    wimage = np.array([[w[0]*w[0],w[0]*w[1],w[0]*w[2]],
                       [w[1]*w[0],w[1]*w[1],w[1]*w[2]],
                       [w[2]*w[0],w[2]*w[1],w[2]*w[2]]], dtype='complex64')
    rgb[:,:,0] = np.real(np.sum(cov*wimage, axis=(2,3)))
    
    # Green: (2*HV)
    w = np.array([0,np.sqrt(2),0], dtype='complex64')
    wimage = np.array([[w[0]*w[0],w[0]*w[1],w[0]*w[2]],
                       [w[1]*w[0],w[1]*w[1],w[1]*w[2]],
                       [w[2]*w[0],w[2]*w[1],w[2]*w[2]]], dtype='complex64')     
    rgb[:,:,1] = np.real(np.sum(cov*wimage, axis=(2,3)))
    
    # Blue: (HH+VV)/2
    w = np.array([1,0,1]/np.sqrt(2), dtype='complex64')
    wimage = np.array([[w[0]*w[0],w[0]*w[1],w[0]*w[2]],
                       [w[1]*w[0],w[1]*w[1],w[1]*w[2]],
                       [w[2]*w[0],w[2]*w[1],w[2]*w[2]]], dtype='complex64')
    rgb[:,:,2] = np.real(np.sum(cov*wimage, axis=(2,3)))
    
    rgb[rgb <= 1e-10] = 1e-10
    rgb = 10*np.log10(rgb)
    
    rgb = (rgb-vmin)/(vmax-vmin)
    rgb[rgb < 0] = 0
    rgb[rgb > 1] = 1
    
    if rotate:
        rgb = np.flipud(np.transpose(rgb, (1,0,2)))
        temp = xlabel
        xlabel = ylabel
        ylabel = temp
        if bounds is not None:
            bounds = (bounds[2], bounds[3], bounds[0], bounds[1])
    
    if figsize is not None:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure()
    
    if bounds is None:
        plt.imshow(rgb, aspect=1, interpolation='nearest', **kwargs)
    else:
        plt.imshow(rgb, extent=(bounds[2],bounds[3],bounds[1],bounds[0]), aspect=1, interpolation='nearest', **kwargs)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    if hideaxis:
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
    
    if savefile is not None:
        plt.savefig(savefile, dpi=dpi, bbox_inches='tight', pad_inches=0.1)