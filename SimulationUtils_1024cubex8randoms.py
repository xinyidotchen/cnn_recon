"""
Collect simulation utilities here.

We use torch Tensors as storage everywhere, we assume these are
stored on the CPU.
"""

import numpy as np
import random
import torch
import os

_system=os.environ.get("NP_SYSTEM","")

class SimulationDataset:
    """
    A comtainer for holding the simulation data, and extracting subgrids from it.
    """

    def __init__(self, nsims=1, ngrid=512, subgridSize=39, batchSide=1):
        # Check to see that the subgrid size is odd, otherwise our
        # padWidth calculations will be wrong.
        if (subgridSize%2)!=1:
            raise ValueError("subgrid size assumed to be odd")

        self.nsims=nsims
        self.ngrid=ngrid
        self.subgridSize=subgridSize
        self.padWidth=self.subgridSize//2  # subgridSize is odd
        self.batchSide=batchSide
        self.Ng = ngrid + 2*self.padWidth+self.batchSide-1
        self.grids = torch.empty((self.nsims, self.Ng, self.Ng, self.Ng),dtype=torch.float32)
        self.simvariance=np.empty(self.nsims, dtype=np.float64)


    def load(self, sim, slot, normalize=False):
        """
        Load a simulation into a particular slot, taking care of the padding etc.

        If normalize is true, then normalize each grid with its standard deviation.
        """
        self.simvariance[slot] = np.var(sim)
        pad=(self.padWidth, self.padWidth+self.batchSide-1)
        if normalize:
            sim /= np.sqrt(self.simvariance[slot])
            self.simvariance[slot]=1.0
        tmp = np.pad(sim, (pad,pad,pad),'wrap')
        tmp = np.array(tmp, dtype=np.float32)
        self.grids[slot,:,:,:] = torch.from_numpy(tmp)

    def getNormalization(self):
        """
        Return the normalization of the set of simulations. Will only return something
        meaningful after all simulations have been loaded.
        """
        return np.sqrt(np.mean(self.simvariance))

    def normalize(self, norm=None):
        """
        Normalize all the simulations. If norm is None, then will use the result of getNormalization.
        """
        norm1=norm
        if norm is None:
            norm1=self.getNormalization()
        self.grids /= norm1


    def getCenters(self, centerList, outArr):
        """
        Return the centers in outArr.

        No dimensions checking is done for speed.
        """
        for ii, icen in enumerate(centerList):
            isim, x, y, z = icen
            x += self.padWidth
            y += self.padWidth
            z += self.padWidth
            outArr[ii] = self.grids[isim, x, y, z]

    def getSubgrids(self, centerList, outArr):
        """
        Return the centers in outArr.

        outArr has dimensions [igrid,0,x,y,z] for simplicity.

        No dimensions checking is done for speed.
        """
        for ii, icen in enumerate(centerList):
            # Since the centers are in the unpadded frame, when padded, they exactly become
            # the low parts of the array
            isim, xlo, ylo, zlo = icen
            xhi = xlo + self.subgridSize
            yhi = ylo + self.subgridSize
            zhi = zlo + self.subgridSize
            outArr[ii,0,:,:,:] = self.grids[isim, xlo:xhi, ylo:yhi, zlo:zhi]

    def addNoise(self, stddev):
        """
        Add in noise -- this is a debug step. Assumes that the tensors all have 
        variance 1.
        """
        noise = torch.randn_like(self.grids)
        self.grids += stddev*noise
        newstd = np.sqrt(1.0+stddev**2)
        self.grids /= newstd

    def getBlock(self, outArr, isim, i0, j0, k0):
        ilo = i0+self.padWidth
        ihi = i0+self.batchSide+self.padWidth
        jlo = j0+self.padWidth
        jhi = j0+self.batchSide+self.padWidth
        klo = k0+self.padWidth
        khi = k0+self.batchSide+self.padWidth
        outArr[:] = self.grids[isim, ilo:ihi,jlo:jhi,klo:khi].flatten()

    def getPaddedBlock(self, outArr, isim, i0, j0, k0):
        ilo = i0
        ihi = i0+self.batchSide+2*self.padWidth
        jlo = j0
        jhi = j0+self.batchSide+2*self.padWidth
        klo = k0
        khi = k0+self.batchSide+2*self.padWidth
        outArr[0,0,:,:,:] = self.grids[isim, ilo:ihi,jlo:jhi,klo:khi]




class CenterLoader:
    """
    A container for handling the centers during training/testing.

    This is useful when we are looking at a relatively small subset of the grid. 
    If you were doing this on the full grid, you would do this differently.
    """

    def __init__(self, centers, nsims=1, blocksize=1):
        """
        centers : ([xcen], [ycen], [zcen]) -- tuple of centers
        nsims : number of simulations [1]
        blocksize : size of blocks to iterate; must divide the number of centers.

        The number of centers is nx*ny*nz*nsims
        """
        xcen, ycen, zcen = centers
        nx, ny, nz = len(xcen), len(ycen), len(zcen)
        self.ncenters = nx*ny*nz*nsims
        self.blocksize = blocksize
        if (self.ncenters%blocksize) != 0 : 
            raise ValueError(f"blocksize={blocksize} must divide ncenters={self.ncenters}")
        self.nblocks= self.ncenters//self.blocksize
        
        self.centerlist = []
        for isim in range(nsims):
            for ix in xcen:
                for iy in ycen:
                    for iz in zcen:
                        self.centerlist.append((isim,ix,iy,iz))
        
        if (len(self.centerlist) != self.ncenters):
            raise RuntimeError(f"Oops, length of centerlist isn't correct!")


    def shuffle(self):
        """
        Shuffle the order of the centers.  

        We make this an explicit call to allow for repeatability.
        """
        random.shuffle(self.centerlist)


    def iterate(self):
        for iblock in range(self.nblocks):
            lo = iblock*self.blocksize
            hi = lo + self.blocksize
            yield self.centerlist[lo:hi]


def genRandomCenters(Ngrid, nx, ny, nz):
    iGrid=np.arange(Ngrid)
    np.random.shuffle(iGrid)
    xcen=iGrid[0:nx]
    np.random.shuffle(iGrid)
    ycen=iGrid[0:ny]
    np.random.shuffle(iGrid)
    zcen=iGrid[0:nz]
    return (xcen, ycen, zcen)


#########################
# Some useful helper functions here, to read Xinyi's files.

def loadRecon(simnum):
#    if _system=="google":
#        template="/home/npadmana/data/recon/sm10/delta_r_{}.npy"
#    else:
#    template="/home/xc298/palmer_scratch/scratch60/reconstruction/output/Quijote/Snapshots/fiducial_HR/{}/grid512/z0.0/standard/sm10/large/delta_r_standard_spacereal_ani1.0.dat.npy"
    template="/home/xc298/palmer_scratch/scratch60/reconstruction/output/Quijote/Snapshots/fiducial_HR/{}/grid512/z0.0/standard/sm10/large/1024cubex8randoms/delta_r_standard_spacereal_ani1.0.dat.npy"
    inputfn=template.format(simnum)
#    return np.arcsinh(np.load(inputfn))
    return np.arcsinh(np.real(np.load(inputfn)))

def loadSmoothedIni(simnum, ftype="sim3"):
#    if _system=="google":
#        template="/home/npadmana/data/ini/{}/delta_r_ini_{}.npy"
#    else:
#        raise NotImplementedError("Haven't implemented ini reading on other machines")
#    template="/home/xc298/palmer_scratch/scratch60/reconstruction/cnn_recon_subgrid/Quijote/Snapshots/fiducial_HR/{}/grid512/ini/set4/delta_r_ini_smoothed_sm3.dat.npy"
    template="/home/xc298/palmer_scratch/scratch60/reconstruction/output/Quijote/Snapshots/fiducial_HR/{}/grid512/ini/large/delta_r_ini.dat.npy"
#    template="/home/xc298/palmer_scratch/scratch60/reconstruction/cnn_recon_subgrid/Quijote/Snapshots/fiducial_HR/{}/grid512/ini/set4/delta_r_ini_smoothed_sm1.dat.npy"
    inputfn=template.format(simnum)
#    return np.load(inputfn)
    return np.real(np.load(inputfn))
