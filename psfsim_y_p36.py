"""
Created on Fri Apr 20 13:15:07 2012

@author: kner
"""
import numpy as N
import Utility_p36 as U
import zernike_p36 as Z
fft2 = N.fft.fft2
ifft2 = N.fft.ifft2
fftshift = N.fft.fftshift
pi = N.pi
from scipy.special import j1

class psf(object):

    def __init__(self):
        self.wl = 0.515
        self.na = 0.8
        self.n2 = 1.333
        self.dx = 0.1
        self.ny = 128
        self.nx = 128
        self.dp = 1 / (self.nx * self.dx)
        self.radius = self.na / self.wl / self.dp
        self.zarr = N.zeros(15)

    def __del__(self):
        pass

    def setParams(self, wl=None, na=None, dx=None, ny=None, nx=None):
        if wl != None:
            self.wl = wl
        if na != None:
            self.na = na
        if dx != None:
            self.dx = dx
        if ny != None:
            self.nx = ny
        if nx != None:
            self.nx = nx
        else:
            self.ny = self.nx * 2
        self.dp = 1 / (self.ny * self.dx)
        self.radius = self.na / self.wl / self.dp

    def getFlatWF(self):
        self.bpp = U.discArray((self.nx, self.nx), self.radius)

    def getZArrWF(self, zarr):
        self.zarr = zarr
        msk = U.discArray((self.nx, self.nx), self.radius)
        ph = N.zeros((self.nx, self.nx), dtype=N.float32)
        for j, m in enumerate(zarr):
            ph += m * Z.Zm(j, rad=self.radius, orig=None, Nx=self.nx)
        self.bpp = msk * N.exp(1j * ph)

    def focusmode(self, d):
        x = N.arange(-self.nx / 2, self.nx / 2, 1)
        X, Y = N.meshgrid(x, x)
        rho = N.sqrt(X ** 2 + Y ** 2) / self.radius
        msk = (rho <= 1.0).astype(N.float64)
        wf = msk * (self.n2 * d / self.wl) * N.sqrt(1 - (self.na * msk * rho / self.n2) ** 2)
        return wf

    def get3Dpsf(self, start, stop, step):
        nsteps = int((stop - start) / step + 1)
        zarr = N.linspace(start, stop, nsteps)
        self.stack = N.zeros((nsteps, self.nx, self.nx))
        for m, z in enumerate(zarr):
            ph = self.focusmode(z)
            wf = self.bpp * N.exp(2j * pi * ph)
            self.stack[m] = N.abs(fftshift(fft2(wf))) ** 2
        return True

    def getOTF3D(self):
        self.otf3D = N.fft.fftn(self.stack)
        self.otf3D = self.otf3D / self.otf3D[0, 0, 0]
        return True

    def otf2d(self):
        nx = self.nx
        nx2 = nx / 2
        ds = self.wl / self.na / self.dx / self.nx
        g = lambda r: N.select([ds * r < 2], [(2 * N.arccos(ds * r / 2) - N.sin(2 * N.arccos(ds * r / 2))) / N.pi], 0.0)
        otf = U.radialArray((nx, nx), g, origin=None)
        self.otf = U.radialArray((nx, nx), g, origin=(0, 0))
        return otf

    def otf2dstok(self, z):
        """ w is defocus in microns
            approximation from Stokseth paper """
        sina = self.na / self.n2
        w = z * (1 - N.sqrt(1 - sina ** 2))
        if w == 0:
            otf = self.otf2d()
        else:
            nx = self.nx
            ds = self.wl / self.na / self.dx / self.nx
            da = 4 * pi * w * ds / self.wl
            g = lambda r: N.select([r <= 0.8, ds * r < 2], [1.0, 2 * (1 - 0.69 * ds * r + 0.0076 * (ds * r) ** 2 + 0.043 * (ds * r) ** 3) * (j1(da * r - 0.5 * (da * r) * (ds * r)) / (da * r - 0.5 * (da * r) * (ds * r)))], 0.0)
            otf = U.radialArray((nx, nx), g, origin=None)
            self.otf = U.radialArray((nx, nx), g, origin=(0, 0))
        return otf

    def otf2daberr(self, zarr, w):
        ph = self.focusmode(w)
        wf = self.bpp * N.exp(2j * pi * ph)
        t = N.abs(fftshift(fft2(wf))) ** 2
        otf = ifft2(t)
        otf = otf / otf[0, 0]
        self.otf = otf
        return otf

    def get3Dpsf2Obj(self, start, stop, step):
        nsteps = (stop - start) / step + 1
        zarr = N.linspace(start, stop, nsteps)
        self.stack = N.zeros((nsteps, self.nx, self.nx))
        for m, z in enumerate(zarr):
            ph1 = self.focusmode(z)
            wf1 = self.bpp * N.exp(2j * pi * ph1)
            ph2 = self.focusmode(-z)
            wf2 = self.bpp * N.exp(2j * pi * ph2)
            self.stack[m] = N.abs(fftshift(fft2(wf1)) + fftshift(fft2(wf2))) ** 2
        return True
