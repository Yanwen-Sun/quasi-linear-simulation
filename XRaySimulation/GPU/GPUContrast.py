import math

import numpy as np
from numba import cuda


import math

import numpy as np
from numba import cuda


#########################################################################################
#                         Method 2
#
#            Get the contrast of each pulse individually.
#########################################################################################
def getGammaT(eFieldComplex, qVec, k0, nx, ny, nz, dx, dy, dz, dsz, nSampleZ, alpha = 0.0):
    """
    Very challenging calculation.
    Need to check with Yanwen about the definition of the calculation.

    :param eFieldComplexFiles:
    :param qVec:
    :param k0:
    :param nx:
    :param ny:
    :param nz:
    :param dx:
    :param dy:
    :param dz:
    :param nSampleZ:
    :param alpha:
    :return:
    """
    # Step1, prepare the variables
    numXY = nx * ny

    # The phase change according to Q/k0 * r

    deltaZx, deltaZy = np.indices((nx,ny))
    
    deltaZx = deltaZx*dx*qVec[0]/k0/dz
    deltaZx = np.ascontiguousarray(np.reshape(deltaZx, numXY))
    
    deltaZy = deltaZy*dy*qVec[1]/k0/dz
    deltaZy = np.ascontiguousarray(np.reshape(deltaZy, numXY))

    
    deltaZz = np.ascontiguousarray(np.arange(-(nSampleZ - 1), nSampleZ, 1) * dsz * qVec[2] / k0 / dz)

    # Get the weight of the summation over z2-z1
    zz = np.abs(np.arange(-nSampleZ+1, nSampleZ))
    if alpha == 0.0:
        weight = np.ascontiguousarray((nSampleZ - zz).astype(np.float64))
    else:
        weight = np.ascontiguousarray(np.exp(-2*alpha*zz)*(np.exp(-4*alpha*(nSampleZ-zz))-1)/(np.exp(-4*alpha)-1))       
    # Move the gpu to reduce traffic
    cuDeltaZx = cuda.to_device(deltaZx)
    cuDeltaZy = cuda.to_device(deltaZy)
    cuDeltaZz = cuda.to_device(deltaZz)
    cuWeight = cuda.to_device(weight)

    eFieldRealFlat = np.ascontiguousarray(np.reshape(eFieldComplex.real, (numXY, nz)))
    eFieldImagFlat = np.ascontiguousarray(np.reshape(eFieldComplex.imag, (numXY, nz)))
    del eFieldComplex
    # Define gpu calculation batch
    threadsperblock = (int(16), int(16))
    blockspergrid_x = int(math.ceil(numXY / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(numXY / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    contrastLocal = np.zeros(1, dtype=np.float64)
    cuContrast = cuda.to_device(contrastLocal)
    countLocal = np.zeros(1, dtype=np.float64)
    # Update the coherence function
    getCoherenceFunction[[blockspergrid, threadsperblock]](
        numXY,
        nz,
        nSampleZ,
        cuDeltaZx,
        cuDeltaZy,
        cuDeltaZz,
        cuWeight,
        eFieldRealFlat,
        eFieldImagFlat,
        cuContrast,
    )

    contrast = cuContrast.copy_to_host()[0]
    return contrast


@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:])')
def getCoherenceFunction(nSpatial,
                         nz,
                         nSample,
                         deltaZx,
                         deltaZy,
                         deltaZz,
                         weight,
                         eFieldReal,
                         eFieldImag,
                         contrastHolder):
    """
    We divide the reshaped time-averaged coherence function along the first dimension.

    :param nSpatial:  The length of the spatial index, which = nx * ny
    :param nz:   The ends of the summation of the mutual coherence function
    :param nSample:
    :param deltaZx:
    :param deltaZy:
    :param deltaZz:
    :param weight:
    :param eFieldReal:
    :param eFieldImag:
    :return:
    """
    idx1, idx2 = cuda.grid(2)

    if (idx1 < nSpatial) & (idx2 < nSpatial):
        deltaZxy = (deltaZx[idx2] + deltaZy[idx2]) - (deltaZx[idx1] + deltaZy[idx1])  # get the contribution from the xy direction in Q*(r2-r1)/k0

        for sIdx in range(2*nSample-1):
            deltaZ = int(round(deltaZxy + deltaZz[sIdx]))  # Get the contribution from the z dimension

            if abs(deltaZ) >= nz:
                continue

            # The delta Z determines the range over which time we calculate the average
            zStart = max(0, -deltaZ)
            zStop = min(nz, nz - deltaZ)
            
            # Add the temporary holder
            holderRealTmp = 0.0
            holderImagTmp = 0.0

            for tIdx in range(zStart, zStop):
                holderRealTmp += eFieldReal[idx1, tIdx] * eFieldReal[idx2, tIdx + deltaZ]
                holderRealTmp += eFieldImag[idx1, tIdx] * eFieldImag[idx2, tIdx + deltaZ]

                holderImagTmp += eFieldReal[idx1, tIdx] * eFieldImag[idx2, tIdx + deltaZ]
                holderImagTmp -= eFieldImag[idx1, tIdx] * eFieldReal[idx2, tIdx + deltaZ]
            nts = float(nz-abs(deltaZ))
            holderRealTmp /=  nts
            holderImagTmp /= nts
            newValue = holderRealTmp ** 2 + holderImagTmp ** 2
            newValueWeight = newValue * weight[sIdx]
            cuda.atomic.add(contrastHolder, 0, newValueWeight)


