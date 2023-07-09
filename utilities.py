import sys
sys.path.append("../XRaySimulation/")
from XRaySimulation import util, Pulse, misc, Crystal, GetCrystalParam, MultiDevice
import numpy as np
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt

def rollEfield(eFieldAfterCrystals,ny,nz):
    eFieldAfterCrystalsStatistics = misc.get_statistics(np.square(np.abs(eFieldAfterCrystals)))
    y0 = np.argmax(eFieldAfterCrystalsStatistics['1d projection']['y'])
    z0 = np.argmax(eFieldAfterCrystalsStatistics['1d projection']['z'])
    ind = (ny//2-y0, nz//2-z0)
    eFieldAfterCrystals = np.roll(eFieldAfterCrystals,shift = ind, axis = (1,2))
    return eFieldAfterCrystals

def cropEField(eFieldInitial, nx, ny,nx1, ny1):
    #crop the real field E-Field
    return eFieldInitial[nx//2-nx1//2:nx//2+nx1//2+1, nx//2-ny1//2:nx//2+ny1//2+1], nx1, ny1

def crop_spectrum(eSpectrumAfterCrystals, nz,dz,k0, nz_crop = 3**6):
    dfz = np.diff(np.fft.fftfreq(nz, d=dz))[0] # k space z resolution
    dz_crop = np.diff(np.fft.fftfreq(nz_crop, d=dfz))[0] # new real space z resolution
    eSpectrumAfterCrystals_crop = eSpectrumAfterCrystals[:,:, nz//2-nz_crop//2:nz//2+nz_crop//2+1] # cropped new e space spectrum
    zCoor_crop = np.arange(-nz_crop // 2 + 1, nz_crop - nz_crop // 2) * dz_crop
    tCoor_crop = zCoor_crop / util.c
    kzCoor_crop = np.fft.fftshift(np.fft.fftfreq(nz_crop, d=dz_crop) * 2 * np.pi)
    kzCoor_crop += k0
    EzCoor_crop = util.wavevec_to_kev(kzCoor_crop - k0) 
    eFieldAfterCrystals_crop = np.fft.ifftn(np.fft.ifftshift(eSpectrumAfterCrystals_crop))
    return eFieldAfterCrystals_crop, eSpectrumAfterCrystals_crop, nz_crop, dz_crop, EzCoor_crop, tCoor_crop

def oneD_Gaussian(x, amplitude, xo, sigma_x, offset):
    g = offset + amplitude*np.exp( - (x-xo)**2/2/sigma_x**2 )
    return g

def fit_bandwidth(eSpectrumStatistics, plotting = False):
    z = eSpectrumStatistics['1d projection']['z']
    n = len(z)
    x = np.arange(n)-n//2
    initial_guess = (z.max(),0, 5, 0)
    param_bounds=([0,-np.inf,0,-np.inf],[np.inf,np.inf,np.inf,np.inf])
    p,pcov = curve_fit(oneD_Gaussian, x, z, absolute_sigma = True,p0=initial_guess,bounds = param_bounds)
    if plotting == True:
        plt.figure(figsize = (3,3))
        plt.plot(x, z,'r.', markersize = 1)
        plt.plot(x, oneD_Gaussian(x, *p),'gray')
    return p[2]*2.35 #convert to FWHM

def crop_spectrum_xyz(eSpectrumAfterCrystals, nx, dx, ny, dy, nz,dz,k0, nx_crop = 3**3,ny_crop = 3**3,nz_crop = 3**6):
    dfz = np.diff(np.fft.fftfreq(nz, d=dz))[0] # k space z resolution
    dz_crop = np.diff(np.fft.fftfreq(nz_crop, d=dfz))[0] # new real space z resolution
    
    dfx = np.diff(np.fft.fftfreq(nx, d=dx))[0]
    dx_crop = np.diff(np.fft.fftfreq(nx_crop, d=dfx))[0] 
    
    dfy = np.diff(np.fft.fftfreq(ny, d=dy))[0]
    dy_crop = np.diff(np.fft.fftfreq(ny_crop, d=dfy))[0]     
    
    eSpectrumAfterCrystals_crop = eSpectrumAfterCrystals[nx//2-nx_crop//2:nx//2+nx_crop//2+1,ny//2-ny_crop//2:ny//2+ny_crop//2+1, nz//2-nz_crop//2:nz//2+nz_crop//2+1] # cropped new e space spectrum
    
    xCoor_crop = np.arange(-nx_crop // 2 + 1, nx_crop - nx_crop // 2) * dx_crop
    yCoor_crop = np.arange(-ny_crop // 2 + 1, ny_crop - ny_crop // 2) * dy_crop
    zCoor_crop = np.arange(-nz_crop // 2 + 1, nz_crop - nz_crop // 2) * dz_crop
    
    tCoor_crop = zCoor_crop / util.c
    # Get k mesh
    kxCoor_crop = np.fft.fftshift(np.fft.fftfreq(nx_crop, d=dx_crop) * 2 * np.pi)
    kyCoor_crop = np.fft.fftshift(np.fft.fftfreq(ny_crop, d=dy_crop) * 2 * np.pi)
    kzCoor_crop = np.fft.fftshift(np.fft.fftfreq(nz_crop, d=dz_crop) * 2 * np.pi)
    kzCoor_crop += k0

    ExCoor_crop = util.wavevec_to_kev(kxCoor_crop)
    EyCoor_crop = util.wavevec_to_kev(kyCoor_crop)
    EzCoor_crop = util.wavevec_to_kev(kzCoor_crop - k0)
    eFieldAfterCrystals_crop = np.fft.ifftn(np.fft.ifftshift(eSpectrumAfterCrystals_crop))
    return eFieldAfterCrystals_crop, eSpectrumAfterCrystals_crop, nx_crop, dx_crop, ny_crop, dy_crop, nz_crop, dz_crop, xCoor_crop, yCoor_crop, zCoor_crop, ExCoor_crop, EyCoor_crop, EzCoor_crop, tCoor_crop


#plotting
def plotting_e_beam(eFieldStatistics, eSpectrumStatistics, xCoor, yCoor, zCoor, tCoor, ExCoor, EyCoor, EzCoor):
    print ("Plotting Electric Field")
    fig, axes = plt.subplots(ncols=3, nrows=2)
    fig.set_figheight(6)
    fig.set_figwidth(12)

    axes[0,0].plot(xCoor, eFieldStatistics['1d projection']['x'])
    axes[0,0].set_title("x Proj")
    axes[0,0].set_xlabel("x (um)")

    axes[0,1].plot(yCoor, eFieldStatistics['1d projection']['y'])
    axes[0,1].set_title("y Proj")
    axes[0,1].set_xlabel("y (um)")

    axes[0,2].plot(tCoor, eFieldStatistics['1d projection']['z'])
    axes[0,2].set_title("z Proj")
    axes[0,2].set_xlabel("t (fs)")

    img1 = axes[1,0].imshow(eFieldStatistics['2d projection']['xy'], aspect="auto", cmap='jet',
                     extent=[yCoor.min(), yCoor.max(), xCoor.min(), xCoor.max(), ])
    fig.colorbar(img1, ax=axes[1,0])
    axes[1,0].set_title("xy Proj")
    axes[1,0].set_ylabel("x (um)")
    axes[1,0].set_xlabel("y (um)")

    img2 = axes[1,1].imshow(eFieldStatistics['2d projection']['yz'], aspect="auto", cmap='jet',
                     extent=[zCoor.min(), zCoor.max(), yCoor.min(), yCoor.max(), ])
    fig.colorbar(img2, ax=axes[1,1])
    axes[1,1].set_title("yz Proj")
    axes[1,1].set_ylabel("y (um)")
    axes[1,1].set_xlabel("z (um)")

    img3 = axes[1,2].imshow(eFieldStatistics['2d projection']['xz'], aspect="auto", cmap='jet',
                     extent=[zCoor.min(), zCoor.max(), xCoor.min(), xCoor.max()])
    fig.colorbar(img3, ax=axes[1,2])
    axes[1,2].set_title("xz Proj")
    axes[1,2].set_ylabel("x (um)")
    axes[1,2].set_xlabel("z (um)")

    plt.tight_layout()
    plt.show()

    print ("Plotting Electric Field Spectrum")
    fig, axes = plt.subplots(ncols=3, nrows=2)
    fig.set_figheight(6)
    fig.set_figwidth(12)

    axes[0,0].plot(ExCoor * 1000, eSpectrumStatistics['1d projection']['x'])
    axes[0,0].set_title("x Proj")
    axes[0,0].set_xlabel("Ex (eV)")

    axes[0,1].plot(EyCoor * 1000, eSpectrumStatistics['1d projection']['y'])
    axes[0,1].set_title("y Proj")
    axes[0,1].set_xlabel("Ey (eV)")

    axes[0,2].plot(EzCoor * 1000, eSpectrumStatistics['1d projection']['z'])
    axes[0,2].set_title("z Proj")
    axes[0,2].set_xlabel("Ez (eV)")

    img1 = axes[1,0].imshow(eSpectrumStatistics['2d projection']['xy'], aspect="auto", cmap='jet',
                     extent=[EyCoor.min() * 1000, EyCoor.max() * 1000, ExCoor.min() * 1000, ExCoor.max() * 1000, ])
    fig.colorbar(img1, ax=axes[1,0])
    axes[1,0].set_title("xy Proj")
    axes[1,0].set_ylabel("Ex (eV)")
    axes[1,0].set_xlabel("Ey (eV)")

    img2 = axes[1,1].imshow(eSpectrumStatistics['2d projection']['yz'], aspect="auto", cmap='jet',
                     extent=[EzCoor.min()* 1000, EzCoor.max()* 1000, EyCoor.min()* 1000, EyCoor.max()* 1000, ])
    fig.colorbar(img2, ax=axes[1,1])
    axes[1,1].set_title("yz Proj")
    axes[1,1].set_ylabel("Ey (eV)")
    axes[1,1].set_xlabel("Ez (eV)")

    img3 = axes[1,2].imshow(eSpectrumStatistics['2d projection']['xz'], aspect="auto", cmap='jet',
                     extent=[EzCoor.min()* 1000, EzCoor.max()* 1000, ExCoor.min()* 1000, ExCoor.max()* 1000])
    fig.colorbar(img3, ax=axes[1,2])
    axes[1,2].set_title("xz Proj")
    axes[1,2].set_ylabel("Ex (eV)")
    axes[1,2].set_xlabel("Ez (eV)")

    plt.tight_layout()
    plt.show()

def prop_beam(eSpectrum, kzCoor, omega, propDistance):
    t = propDistance / util.c
    propagationPhaseComplex = np.exp(1.j * (kzCoor[np.newaxis, np.newaxis, :] * propDistance - omega * t))
    eSpectrumPropagation = np.multiply(eSpectrum, propagationPhaseComplex)
    eFieldPropagation = np.fft.ifftn(np.fft.ifftshift(eSpectrumPropagation))
    return eFieldPropagation, eSpectrumPropagation

def focus_beam(eField, xCoor, yCoor, kzCoor, focalLengthX, focalLengthY):
    phaseX = np.exp(-1.j * np.outer(np.square(xCoor), kzCoor / 2. / focalLengthX))
    phaseY = np.exp(-1.j * np.outer(np.square(yCoor), kzCoor / 2. / focalLengthY))

    # Get different frequency slice for the collimation
    xywFieldBeforeLens = np.fft.fftshift(np.fft.fft(eField, axis=-1), axes=-1)
    # Add the transmission function to the electric field along each direction
    xywFieldAfterLens = np.multiply(xywFieldBeforeLens, phaseY[np.newaxis,:,:])
    xywFieldAfterLens = np.multiply(xywFieldAfterLens, phaseX[:,np.newaxis,:])
    eFieldAfterLens = np.fft.ifft(np.fft.ifftshift(xywFieldAfterLens, axes=-1), axis=-1) 
    eSpectrumAfterLens = np.fft.fftshift(np.fft.fftn(eFieldAfterLens))
    return eFieldAfterLens, eSpectrumAfterLens
