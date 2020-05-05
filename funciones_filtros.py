# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:11:35 2019

@author: Usuario
"""
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


#FILTRO RESONANTE
def digital_resonator(F0,Fs=12000,r=0.95,zeros=[1,-1]):
    #F0: Frecuencia resonancia deseada [Hz]
    #Fs: Frecuencia de muestreo [Hz]
    #r: Factor de selectividad [0 - 1]
    #zeros: array con los zeros a considerar (posicion angular digital)
    from numpy import pi
    from numpy import exp

    if zeros != None:
        num = np.poly1d(zeros,True).c #True indica que la entrada son las raices del poly
        num = num[::-1]
    else:
        num = np.array([1])

    w = F0/Fs*(2*pi)        
    polo = r*exp(w*1j)
    
    polos = [polo,polo.conjugate()]
    den = np.poly1d(polos,True).c #el comando [::-1] es resultado de z^(-1)
    den = den[::-1]
    
    num_w = np.polyval(num,1/exp(w*1j))
    den_w = np.polyval(den,1/exp(w*1j))
    b0 = np.abs(den_w)/np.abs(num_w)
    
    num=b0*num
    return num, den


#VENTANEO
def windowing(num,den,M=100):
    b = num
    a = den
    res = b #inicializaciòn
    h = np.array([])
    for ii in range(0,M):
        u,res = np.polydiv(res,a)
        res = np.append(res,0)
        h = np.append(h,u)
    #print(h)
    u,v = signal.freqz(b = h,worN=512*16)
    plt.figure(figsize = (10,3))
    ax1 = plt.subplot(1,2,1,
                      ylabel='Magnitude', 
                      xlabel='Frequency [rad/sample]',
                      xscale = 'log')
    ax1.grid(True)
    ax1.plot(u, np.abs(v), 'b')
    ax2 = plt.subplot(1,2,2,
                      ylabel='Angle (radians)',
                      xlabel='Frequency [rad/sample]',
                      xscale = 'log')
    ax2.plot(u,np.unwrap(np.angle(v)))
    ax2.grid(True)
    return h

#Obtener envoltura
def GetEnvelope(xfil, Fs = 1):
    #xfil: signal filtered (or without filter)
    xa = signal.hilbert(xfil)
    env_xa = np.abs(xa)**2
    ENV_xa = np.fft.fft(env_xa)
    MAG_ENV_xa = np.abs(ENV_xa)
    ANG_ENV_xa = np.unwrap(np.angle(ENV_xa))

    return xa, env_xa, ENV_xa, MAG_ENV_xa, ANG_ENV_xa

def PlotEnvelope(xfil, Fs=1):
    xa, env_xa, ENV_xa, MAG_ENV_xa, ANG_ENV_xa = GetEnvelope(xfil,Fs)
    time = np.linspace(0,len(xa)/Fs,len(xa))
    frec = np.fft.fftfreq(len(ENV_xa),1/Fs)
    
    plt.figure(figsize=(15,4))

    ax1 = plt.subplot(1,3,1,xlim=(0,0.01));
    ax1.plot(time,env_xa,'r')
    ax1.plot(time,xfil)
    ax1.grid(True)

    ax2 = plt.subplot(1,3,2,xlim=(0,600),ylim=(0,200*np.mean(np.abs(ENV_xa))))
    ax2.plot(frec,np.abs(ENV_xa))
    ax2.grid(True,axis='y')

    ax3 = plt.subplot(1,3,3,xlim=(0.1,100))
    ax3.plot(frec,np.unwrap(np.angle(ENV_xa)))
    ax3.grid(True)
    
    return ax1, ax2, ax3