# -*- coding: utf-8 -*-
"""Coherence Region Plotting

    Module containing functions for coherence region plotting, including
    interactive coherence regions that show the modelled coherences for
    user-specified parameter values.
    
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
import matplotlib.widgets as widgets

import kapok.cohopt
import kapok.topo
from kapok.lib import makehermitian, mb_cov_index


sliders = [] # array to store slider objects for interactive coherence region (if we don't keep a reference to them, they can become unresponsive)


def cohregion(scene, az, rng, bl=0, mlwin=None, title=None, savefile=None,
              reg=0.0):
    """Plot a coherence region for a given covariance matrix.
    
        The following coherences are plotted: HH, HV, VV, HH+VV, HH-VV
        (e.g., all the standard Lexicographic and Pauli coherences), as
        different colored dots.  The coherence region boundary itself is
        plotted as a solid blue line.  The high and low optimized coherences
        are plotted as brown and dark green dots.  The line fit through the
        optimized coherences is plotted as a dashed green line, with the
        estimated ground coherence shown as a black dot.  The alternate
        ground solution which wasn't chosen is shown as an orange dot.
    
        Arguments:
            scene (object): A kapok.Scene object containing covariance matrix
                and other data.
            az (int): Azimuth index of the plotted coherence region.
            rng (int): Range index of the plotted coherence region.
            bl (int): Baseline index of the plotted coherence region.
            mlwin (tuple): Multilooking window size of the data.  If
                specified, the original SLC coordinates of the region will be
                annotated on the plot.
            title (str): Title string to put at the top of the plot.
            savefile (str): Path and filename to save the figure, if desired.
            reg (float): Optimization regularization argument passed to
                kapok.cohopt.pdopt_pixel.  Default: 0.0 (no regularization).
        
    """
    row, col = mb_cov_index(bl)
    
    tm = makehermitian(0.5*(scene.cov[az,rng,row:row+scene.num_pol,row:row+scene.num_pol]
                         + scene.cov[az,rng,col:col+scene.num_pol,col:col+scene.num_pol]))
    om = scene.cov[az,rng,row:row+scene.num_pol,col:col+scene.num_pol]
    
    gammahigh, gammalow, gammaall = kapok.cohopt.pdopt_pixel(tm, om, reg=reg)
    gammahh = scene.coh(pol=0, bl=bl, pix=(az,rng))
    gammahv = scene.coh(pol=1, bl=bl, pix=(az,rng))
    gammavv = scene.coh(pol=2, bl=bl, pix=(az,rng))
    gammahhpvv = scene.coh(pol=[1,0,1], bl=bl, pix=(az,rng))
    gammahhmvv = scene.coh(pol=[1,0,-1], bl=bl, pix=(az,rng))
    
    kz = scene.kz(bl)[az,rng]   

    # Create the figure.
    plt.figure()
    fig = plt.gcf()
    
    
    # Circles with 0.25, 0.5, 0.75, and 1.0 radius.
    unitcirc = plt.Circle((0,0),1,color='k',fill=False,linestyle='dashed')
    threequartercirc = plt.Circle((0,0),0.75,color='k',fill=False,linestyle='dashed')
    halfcirc = plt.Circle((0,0),0.5,color='k',fill=False,linestyle='dashed')
    quartercirc = plt.Circle((0,0),0.25,color='k',fill=False,linestyle='dashed')
    fig.gca().add_artist(unitcirc)
    fig.gca().add_artist(threequartercirc)
    fig.gca().add_artist(halfcirc)
    fig.gca().add_artist(quartercirc)

    
    # Do a line fit to get the ground coherences.
    gammatemp = np.zeros((2,2,2),dtype='complex')
    gammatemp[0,:,:] = gammahigh
    gammatemp[1,:,:] = gammalow
    ground, groundalt, volindex = kapok.topo.groundsolver(gammatemp, kz=kz, returnall=True, silent=True)
    if volindex[0,0]: # swap high and low coherences
        temp = gammahigh
        gammahigh = gammalow
        gammalow = temp
        
    ground = ground[0,0]
    groundalt = groundalt[0,0]
     
    # Plot region.
    plt.plot(np.real(gammaall),np.imag(gammaall),'-',linewidth=3,markersize=8,label='Region')
    
    # Plot the fitted line.
    gammaline = np.array((ground,gammalow,gammahigh),dtype='complex')
    plt.plot(np.real(gammaline),np.imag(gammaline),'--g',linewidth=3,label='Line Fit')


    # Plot Lexicographic and Pauli coherences.
    plt.plot(np.real(gammahh),np.imag(gammahh),'.r',markersize=20,label='HH')
    plt.plot(np.real(gammavv),np.imag(gammavv),'.b',markersize=20,label='VV')
    plt.plot(np.real(gammahv),np.imag(gammahv),'.',color='LightGreen',markersize=20,label='HV')
    plt.plot(np.real(gammahhpvv),np.imag(gammahhpvv),'.c',markersize=20,label='HH+VV')
    plt.plot(np.real(gammahhmvv),np.imag(gammahhmvv),'.m',markersize=20,label='HH-VV')
    
    # Plot optimized high and low coherences.
    plt.plot(np.real(gammahigh),np.imag(gammahigh),'.',color='DarkGreen',markersize=20,label='Opt. High')
    plt.plot(np.real(gammalow),np.imag(gammalow),'.',color='Maroon',markersize=20,label='Opt. Low')
    
    # Plot ground coherences.
    plt.plot(np.real(ground),np.imag(ground),'.k',markersize=20,label='Ground')
    plt.plot(np.real(groundalt),np.imag(groundalt),'.',color='Orange',markersize=20,label='Alt. Ground')
    
    # Text label SLC coordinates, if appropriate.
    if mlwin is not None:
        azslc = (mlwin[0]*az, mlwin[0]*az + (mlwin[0]-1))
        rngslc = (mlwin[1]*rng, mlwin[1]*rng + (mlwin[1]-1))
        plt.text(0.975,0.94,'SLC Coordinates', fontsize=10, horizontalalignment='right')
        plt.text(0.975,0.89,'Azimuth: '+str(azslc[0])+'–'+str(azslc[1]), fontsize=10, horizontalalignment='right')
        plt.text(0.975,0.84,'Range: '+str(rngslc[0])+'–'+str(rngslc[1]), fontsize=10,horizontalalignment='right')


    # Plot Title
    if title is None:    
        plt.title('Coherence Region for Pixel ('+str(az)+', '+str(rng)+')')
    else:
        plt.title(title)

    plt.xlabel('Real')
    plt.ylabel('Imaginary')



    fig.gca().set_aspect('equal')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, numpoints=1)
    if savefile is not None:
        plt.savefig(savefile, dpi=200, bbox_inches='tight', pad_inches=0.1)
        
    return


def rvogregion(scene=None, az=None, rng=None, bl=0, reg=0.0):
    """Interactive coherence region plot with sliders for the RVoG model
        parameters.
        
        Create an interactive coherence region plot to observe the affect
        of the RVoG model parameters on the modelled coherences.  If a
        Scene object and azimuth and range indices are provided, the
        coherence region and associated coherences from the data will be
        plotted, and the model parameters can be adjusted to see how the
        model fits the data.  If a Scene object is not provided, the
        RVoG modelled coherences are still plotted, but without any
        observed data.  In this case, the function will use a fixed kz
        value of 0.10 rad/m, a fixed incidence angle of 45 degrees, and
        a fixed ground phase of 0.
        
        The sliders allow the user to change the following parameter values:
        hv (forest height, in meters), extinction (in dB/meter), the mu
        (ground-to-volume scattering ratio) for the low coherence, and
        alpha (the volumetric temporal decorrelation magnitude).
        
        The modelled coherences are plotted in black.  A dashed line
        shows the coherence values for a range of forest height values,
        starting at 0.01 m and increasing up towards the forest height
        specified by the user.  The volume coherence (mu = 0, no ground
        contribution) is shown as a black X.  This black X is connected
        to the modelled low coherence (another black X) with a solid black
        line.
    
        Arguments:
            scene (object): A kapok.Scene object containing covariance matrix
                and other data.  If not specified, no actual observed data
                or coherence region will be plotted, but the modelled
                coherences and UI will still be functional.
            az (int): Azimuth index of the plotted coherence region.
            rng (int): Range index of the plotted coherence region.
            bl (int): Baseline index of the plotted coherence region.
            reg (float): Optimization regularization argument passed to
                kapok.cohopt.pdopt_pixel.  Default: 0.0 (no regularization).
        
    """
    # Create the figure.
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    
    
    # Circles with 0.25, 0.5, 0.75, and 1.0 radius.
    unitcirc = plt.Circle((0,0),1,color='k',fill=False,linestyle='dashed')
    threequartercirc = plt.Circle((0,0),0.75,color='k',fill=False,linestyle='dashed')
    halfcirc = plt.Circle((0,0),0.5,color='k',fill=False,linestyle='dashed')
    quartercirc = plt.Circle((0,0),0.25,color='k',fill=False,linestyle='dashed')
    fig.gca().add_artist(unitcirc)
    fig.gca().add_artist(threequartercirc)
    fig.gca().add_artist(halfcirc)
    fig.gca().add_artist(quartercirc)
    
    
    if (scene is not None) and (az is not None) and (rng is not None):
        row, col = mb_cov_index(bl)
        
        tm = makehermitian(0.5*(scene.cov[az,rng,row:row+scene.num_pol,row:row+scene.num_pol]
                           + scene.cov[az,rng,col:col+scene.num_pol,col:col+scene.num_pol]))
        om = scene.cov[az,rng,row:row+scene.num_pol,col:col+scene.num_pol]
        
        gammahigh, gammalow, gammaall = kapok.cohopt.pdopt_pixel(tm, om, reg=reg)
        gammahh = scene.coh(pol=0, bl=bl, pix=(az,rng))
        gammahv = scene.coh(pol=1, bl=bl, pix=(az,rng))
        gammavv = scene.coh(pol=2, bl=bl, pix=(az,rng))
        gammahhpvv = scene.coh(pol=[1,0,1], bl=bl, pix=(az,rng))
        gammahhmvv = scene.coh(pol=[1,0,-1], bl=bl, pix=(az,rng))
        
        kz = scene.kz(bl)[az,rng]            
        inc = scene.inc[az,rng]
        
        # Do a line fit to get the ground coherences.
        gammatemp = np.zeros((2,2,2),dtype='complex')
        gammatemp[0,:,:] = gammahigh
        gammatemp[1,:,:] = gammalow
        ground, groundalt, volindex = kapok.topo.groundsolver(gammatemp, kz=kz, returnall=True, silent=True)
        if volindex[0,0]: # swap high and low coherences
            temp = gammahigh
            gammahigh = gammalow
            gammalow = temp
            
        ground = ground[0,0]
        groundalt = groundalt[0,0]
         
        # Plot region.
        plt.plot(np.real(gammaall),np.imag(gammaall),'-',markersize=8,linewidth=3,label='Region')
        
        # Plot the fitted line.
        gammaline = np.array((ground,gammalow,gammahigh),dtype='complex')
        plt.plot(np.real(gammaline),np.imag(gammaline),'--g',linewidth=3,label='Line Fit')
    
        # Plot Lexicographic and Pauli coherences.
        plt.plot(np.real(gammahh),np.imag(gammahh),'.r',markersize=20,label='HH')
        plt.plot(np.real(gammavv),np.imag(gammavv),'.b',markersize=20,label='VV')
        plt.plot(np.real(gammahv),np.imag(gammahv),'.',color='LightGreen',markersize=20,label='HV')
        plt.plot(np.real(gammahhpvv),np.imag(gammahhpvv),'.c',markersize=20,label='HH+VV')
        plt.plot(np.real(gammahhmvv),np.imag(gammahhmvv),'.m',markersize=20,label='HH-VV')
        
        # Plot optimized high and low coherences.
        plt.plot(np.real(gammahigh),np.imag(gammahigh),'.',color='DarkGreen',markersize=20,label='Opt. High')
        plt.plot(np.real(gammalow),np.imag(gammalow),'.',color='Maroon',markersize=20,label='Opt. Low')
        
        # Plot ground coherences.
        plt.plot(np.real(ground),np.imag(ground),'.k',markersize=20,label='Ground')
        plt.plot(np.real(groundalt),np.imag(groundalt),'.',color='Orange',markersize=20,label='Alt. Ground')
        
        plt.title('Coherence Region for Pixel ('+str(az)+', '+str(rng)+')')
    else:
        ground = 1
        kz = 0.10
        inc = np.radians(45)


    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    

    nptodb = 20/np.log(10) # Nepers to dB Conversion Factor
    
    
    # Set up parameters and starting RVoG modelled coherences.   
    hv = 20.0
    hv_vector = np.linspace(1, hv, num=hv)
    ext = 0.35
    mu_high = 0
    mu_low = 0.5
    alpha = 1.0
    
    # Calculate the RVoG coherences.
    p1 = 2*ext/nptodb/np.cos(inc)
    p2 = p1 + 1j*kz
    gammav = (p1 / p2) * (np.exp(p2*hv_vector)-1) / (np.exp(p1*hv_vector)-1)
    gammahigh = ground * (mu_high + alpha*gammav) / (mu_high + 1)
    gammalow = ground * (mu_low + alpha*gammav[-1]) / (mu_low + 1)
    rvoglocus = np.array([gammahigh[-1],gammalow],dtype='complex')
    
    # Plot RVoG coherences.
    l1 = plt.plot(np.real(gammahigh),np.imag(gammahigh), '--', markersize=20, linewidth=3, color='YellowGreen')[0]
    l2 = plt.plot(np.real(rvoglocus),np.imag(rvoglocus),'.-', markersize=20, linewidth=3, color='YellowGreen', label='RVoG Model')[0]
    
    plt.axis([-1, 1, -1, 1])
    ax.set_aspect('equal')
    
    # UI Sliders
    axcolor = 'lightgoldenrodyellow'
    axhv = plt.axes([0.25, 0.20, 0.5, 0.02], axisbg=axcolor)
    axext = plt.axes([0.25, 0.15, 0.5, 0.02], axisbg=axcolor)
    axmulow = plt.axes([0.25, 0.1, 0.5, 0.02], axisbg=axcolor)
    axalpha  = plt.axes([0.25, 0.05, 0.5, 0.02], axisbg=axcolor)
    
    slidehv = widgets.Slider(axhv, 'hv (m)', 1, 50.0, valinit=hv)
    sliders.append(slidehv)
    slideext = widgets.Slider(axext, 'Ext. (dB/m)', 1e-2, 1.0, valinit=ext)
    sliders.append(slideext)
    slidemulow = widgets.Slider(axmulow, 'Mu', 0.0, 1.0, valinit=mu_low)
    sliders.append(slidemulow)
    slidealpha = widgets.Slider(axalpha, 'Alpha', 0.0, 1.0, valinit=alpha)
    sliders.append(slidealpha)
    
    # Update Function Called When Sliders Are Changed
    def update(val):
        hv = slidehv.val
        hv_vector = np.linspace(0.01, hv, num=hv)
        ext = slideext.val
        mu_low = slidemulow.val
        alpha = slidealpha.val

        # Calculate the RVoG coherences.
        p1 = 2*ext/nptodb/np.cos(inc)
        p2 = p1 + 1j*kz
        gammav = (p1 / p2) * (np.exp(p2*hv_vector)-1) / (np.exp(p1*hv_vector)-1)
        gammahigh = ground * (mu_high + alpha*gammav) / (mu_high + 1)
        gammalow = ground * (mu_low + alpha*gammav[-1]) / (mu_low + 1)
        rvoglocus = np.array([gammahigh[-1],gammalow],dtype='complex')   
        
        # Update plot.
        l1.set_xdata(np.real(gammahigh))
        l1.set_ydata(np.imag(gammahigh))
        
        l2.set_xdata(np.real(rvoglocus))
        l2.set_ydata(np.imag(rvoglocus))
        
        fig.canvas.draw_idle()
        
        
    slidehv.on_changed(update)
    slideext.on_changed(update)
    slidemulow.on_changed(update)
    slidealpha.on_changed(update)
    
    ax.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, numpoints=1)
    
    return