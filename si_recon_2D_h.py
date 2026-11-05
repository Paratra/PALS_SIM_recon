"""2D structured illumination reconstruction.

Original algorithm: Peter Kner, University of Georgia, 2019.
Image directions: top=0, left=pi/2, bottom=pi, right=-pi/2.
The Wiener parameter mu is supplied by the caller.
"""

import matplotlib.pyplot as P
import psfsim_y_p36 as psfsim
import numpy as N
pi = N.pi
fft2 = N.fft.fft2
ifft2 = N.fft.ifft2

class si2D(object):

    def __init__(self, img_stack=None, show_plots=False):
        if img_stack is None:
            raise ValueError('Provide a stack of 3 or 9 phase images.')
        img_stack = N.asarray(img_stack)
        if img_stack.ndim != 3 or img_stack.shape[0] not in (3, 9):
            raise ValueError('Expected a stack with 3 or 9 phase images.')
        nz, nx, ny = img_stack.shape
        if nx != ny or nx < 2 or nx % 2:
            raise ValueError('Images must be square with an even side length.')
        if not N.issubdtype(img_stack.dtype, N.number) or N.iscomplexobj(img_stack) or (not N.isfinite(img_stack).all()):
            raise ValueError('Image intensities must be real, finite numbers.')
        self.show_plots = show_plots
        self.dx = 0.089
        self.na = 1.2
        self.wavelength = 0.515
        self.img = img_stack
        self.nx = nx
        self.mu = 0.01
        self.cutoff = 0.001
        self.psf = self.getpsf()
        self.Ns = 10
        self.r_ang = 4 * 0.02
        self.r_sp = 4 * 0.005

    def getpsf(self):
        psf = psfsim.psf()
        psf.setParams(wl=self.wavelength, na=self.na, dx=self.dx, nx=self.nx)
        psf.getFlatWF()
        wf = psf.bpp
        psf1 = N.abs(fft2(wf)) ** 2
        psf1 = psf1 / psf1.sum()
        return psf1

    def MakeSepMatrix(self):
        nphases = 3
        sepmat = N.zeros((3, nphases), dtype=N.float32)
        norders = int((nphases + 1) / 2)
        phi = 2 * N.pi / nphases
        for j in range(nphases):
            sepmat[0, j] = 1.0 / nphases
            for order in range(1, norders):
                sepmat[2 * order - 1, j] = 2.0 * N.cos(j * order * phi) / nphases
                sepmat[2 * order, j] = 2.0 * N.sin(j * order * phi) / nphases
        return sepmat

    def separate(self, imgs):
        """ imgs.shape = (3,nx,ny) """
        Nw = self.nx
        mat = self.MakeSepMatrix()
        outr = N.dot(mat, imgs.reshape(3, Nw ** 2))
        self.separr = N.zeros((3, self.nx, self.nx), dtype=N.complex64)
        self.separr[0] = outr[0].reshape(Nw, Nw)
        self.separr[1] = (outr[1] + 1j * outr[2]).reshape(Nw, Nw)
        self.separr[2] = (outr[1] - 1j * outr[2]).reshape(Nw, Nw)
        return True

    def shift(self, angle, spacing):
        """ shift data in freq space by multiplication in real space """
        Nw = self.nx
        kx = N.cos(angle) / spacing
        ky = N.sin(angle) / spacing
        ysh = N.zeros((3, 2 * Nw, 2 * Nw), dtype=N.complex64)
        imgsf = N.zeros((3, 2 * Nw, 2 * Nw), dtype=N.complex64)
        otfs = N.zeros((3, 2 * Nw, 2 * Nw), dtype=N.complex64)
        ysh[0] = 1.0
        g = lambda ii, jj: N.exp(1j * N.pi * (kx * ii + ky * jj) * self.dx).astype(N.complex64)
        ysh[1] = N.fromfunction(g, (2 * Nw, 2 * Nw)) * N.exp(-1j * N.pi * Nw * self.dx / spacing)
        g = lambda ii, jj: N.exp(-1j * N.pi * (kx * ii + ky * jj) * self.dx).astype(N.complex64)
        ysh[2] = N.fromfunction(g, (2 * Nw, 2 * Nw)) * N.exp(1j * N.pi * Nw * self.dx / spacing)
        psfdbl = self.interp(self.psf)
        for m in range(3):
            imgsf[m] = fft2(self.interp(self.separr[m]) * ysh[m])
            ysh[m] = N.roll(N.roll(ysh[m], Nw, 0), Nw, 1)
            otfs[m] = fft2(psfdbl * ysh[m])
            otfs[m] = self.phasenuller(otfs[m])
        return (imgsf, otfs)

    def recon(self, angles, linespacing, mags, phases):
        """ 2D si reconstruction """
        Nw = self.nx
        if N.isscalar(linespacing):
            linespacing = N.ones(3) * linespacing
        imgsf = N.zeros((9, 2 * Nw, 2 * Nw), dtype=N.complex64)
        otfa = N.zeros((9, 2 * Nw, 2 * Nw), dtype=N.complex64)
        ph = N.zeros(9, dtype=N.complex64) + 1.0
        if len(self.img.shape) == 3:
            for m, ang in enumerate(angles):
                self.separate(self.img[3 * m:3 * (m + 1)])
                a, b = self.shift(ang, linespacing[m])
                phase = phases[m]
                mag = mags[m]
                ph[3 * m + 1] = N.exp(-1j * phase).astype(N.complex64) * mag
                ph[3 * m + 2] = N.exp(1j * phase).astype(N.complex64) * mag
                imgsf[3 * m:3 * (m + 1), :, :] = a[:]
                otfa[3 * m:3 * (m + 1), :, :] = b[:]
            Snum = N.zeros((2 * Nw, 2 * Nw), dtype=N.complex64)
            Sden = N.zeros((2 * Nw, 2 * Nw), dtype=N.complex64)
            Sden += self.mu ** 2
            for m in range(9):
                Snum += ph[m] * otfa[m].conj() * imgsf[m]
                Sden += abs(otfa[m]) ** 2
            self.S = Snum / Sden
            self.finalimage = ifft2(self.S)
            if self.show_plots:
                P.figure()
                P.imshow(abs(self.S), interpolation='nearest', vmax=0.1 * abs(self.S).max())
                P.figure()
                P.imshow(abs(self.finalimage), interpolation='nearest')
        return True

    def recon2(self, nbr, angle, linespacing, phase, mag, verbose=True):
        """ 2D si reconstruction for one angle
            can omit 0 order for optical sectioning """
        mu = self.mu
        Nw = self.nx
        imgsf = N.zeros((3, 2 * Nw, 2 * Nw), dtype=N.complex64)
        otfa = N.zeros((3, 2 * Nw, 2 * Nw), dtype=N.complex64)
        ang = angle
        self.separate(self.img[3 * nbr:3 * (nbr + 1)])
        a, b = self.shift(ang, linespacing)
        ks, m = self.getoverlap(a, b, ang, linespacing, verbose)
        imgsf = a
        otfa = b
        Snum = N.zeros((2 * Nw, 2 * Nw), dtype=N.complex64)
        Sden = N.zeros((2 * Nw, 2 * Nw), dtype=N.complex64)
        Sden += mu ** 2
        ph = N.exp(-1j * N.array([0.0, phase, -phase])).astype(N.complex64)
        magarr = N.array([1.0, mag, mag])
        for m in range(0, 3):
            Snum += magarr[m] * ph[m] * otfa[m].conj() * imgsf[m]
            Sden += abs(otfa[m]) ** 2
        S = Snum / Sden
        finalimage = ifft2(S)
        if verbose and self.show_plots:
            P.imshow(abs(finalimage))
        self.finalimage = finalimage
        self.S = S
        return finalimage

    def decon(self, nbr, angle, linespacing, phase, mag, verbose=True):
        """ 2D si reconstruction for one angle
            can omit 0 order for optical sectioning """
        mu = self.mu
        Nw = self.nx
        imgsf = N.zeros((3, 2 * Nw, 2 * Nw), dtype=N.complex64)
        otfa = N.zeros((3, 2 * Nw, 2 * Nw), dtype=N.complex64)
        ang = angle
        self.separate(self.img[3 * nbr:3 * (nbr + 1)])
        a, b = self.shift(ang, linespacing)
        ks, m = self.getoverlap(a, b, ang, linespacing, verbose)
        imgsf = a
        otfa = b
        Snum = N.zeros((2 * Nw, 2 * Nw), dtype=N.complex64)
        Sden = N.zeros((2 * Nw, 2 * Nw), dtype=N.complex64)
        Sden += mu ** 2
        ph = N.exp(-1j * N.array([0.0, phase, -phase])).astype(N.complex64)
        magarr = N.array([1.0, mag, mag])
        Snum += magarr[0] * ph[0] * otfa[0].conj() * imgsf[0]
        Sden += abs(otfa[0]) ** 2
        S = Snum / Sden
        finalimage = ifft2(S)
        if verbose and self.show_plots:
            P.imshow(abs(finalimage))
        self.finalimage = finalimage
        self.S = S
        return finalimage

    def pad(self, arr):
        nx, ny = arr.shape
        out = N.zeros((2 * nx, 2 * nx), arr.dtype)
        nxh = int(nx / 2)
        out[:nxh, :nxh] = arr[:nxh, :nxh]
        out[:nxh, 3 * nxh:4 * nxh] = arr[:nxh, nxh:nx]
        out[3 * nxh:4 * nxh, :nxh] = arr[nxh:nx, :nxh]
        out[3 * nxh:4 * nxh, 3 * nxh:4 * nxh] = arr[nxh:nx, nxh:nx]
        return out

    def interp(self, arr):
        if len(arr.shape) == 3:
            ns, nx, ny = arr.shape
            outarr = N.zeros((ns, 2 * nx, 2 * ny), dtype=arr.dtype)
            for q in range(ns):
                arrf = fft2(arr[q])
                arro = self.pad(arrf)
                outarr[q] = ifft2(arro)
        if len(arr.shape) == 2:
            nx, ny = arr.shape
            outarr = N.zeros((2 * nx, 2 * ny), dtype=arr.dtype)
            arrf = fft2(arr)
            outarr = ifft2(self.pad(arrf))
        return outarr

    def getoverlap(self, imgsf, otfa, ang0, spacing, verbose=False):
        mu = self.mu
        cutoff = self.cutoff
        imgf0 = imgsf[0]
        otf0 = otfa[0]
        imgf1 = imgsf[1]
        otf1 = otfa[1]
        wimgf0 = otf1 * imgf0
        wimgf1 = otf0 * imgf1
        msk = (abs(otf0 * otf1) > cutoff).astype(N.float32)
        a = N.sum(msk * wimgf1 * wimgf0.conj()) / N.sum(msk * wimgf0 * wimgf0.conj())
        phase = N.angle(a)
        mag = N.abs(a)
        if verbose and self.show_plots:
            t = msk * wimgf1 * wimgf0.conj() / (msk * wimgf0 * wimgf0.conj())
            t[N.isnan(t)] = 0.0
            P.figure()
            P.imshow(abs(t), interpolation='nearest', vmin=0.0, vmax=2.0)
            P.figure()
            P.imshow(N.angle(t), interpolation='nearest')
        return (phase, mag)

    def test(self, ind=0):
        """ can the angle be found without a guess """
        Nw = self.nx
        imgsf = N.zeros((3, Nw, Nw), dtype=N.complex64)
        self.separate(self.img[3 * ind:3 * (ind + 1)])
        otf = fft2(self.psf)
        mu = 0.0001
        for m in range(3):
            imgsf[m] = otf.conj() * fft2(self.separr[m]) / (abs(otf) ** 2 + mu ** 2)
        q = conv(imgsf[0], imgsf[1])
        if self.show_plots:
            P.imshow(abs(q))
        return q

    def mapoverlap2(self, angle, spacing, ind=0, marr=True):
        self.separate(self.img[3 * ind:3 * (ind + 1)])
        d_ang = 2 * self.r_ang / self.Ns
        d_sp = 2 * self.r_sp / self.Ns
        ang_iter = N.arange(-self.r_ang, self.r_ang + d_ang / 2, d_ang) + angle
        sp_iter = N.arange(-self.r_sp, self.r_sp + d_sp / 2, d_sp) + spacing
        magarr = N.zeros((self.Ns + 1, self.Ns + 1))
        for m, ang in enumerate(ang_iter):
            for n, sp in enumerate(sp_iter):
                a, b = self.shift(ang, sp)
                phase, mag = self.getoverlap(a, b, ang, sp, False)
                if N.isnan(mag):
                    magarr[m, n] = 0.0
                elif marr:
                    magarr[m, n] = mag
                else:
                    magarr[m, n] = phase
        if self.show_plots:
            P.figure()
            P.imshow(magarr, interpolation='nearest')
        self.magarr = magarr
        magarr = abs(magarr)
        mind = magarr.argmax()
        angmax = int(mind / (self.Ns + 1)) * d_ang - self.r_ang + angle
        spmax = mind % (self.Ns + 1) * d_sp - self.r_sp + spacing
        return (angmax, spmax, magarr.max())

    def viewoverlap(self, angle, spacing, ind=0):
        self.separate(self.img[3 * ind:3 * (ind + 1)])
        a, b = self.shift(angle, spacing)
        phase, mag = self.getoverlap(a, b, angle, spacing, True)
        return (phase, mag)

    def findphase(self, ind, angle, linespacing, mag, loc):
        phasarr = N.arange(0, 2 * pi, 0.25)
        outvar = []
        for phase in phasarr:
            fi = self.recon2(ind, angle, linespacing, phase, mag, False)
            metric = fi[loc[0] - 4:loc[0] + 4, loc[1] - 4:loc[1] + 4].max()
            outvar.append(metric)
        outvar = N.array(outvar)
        return (phasarr, outvar)

    def phasenuller(self, imgf):
        nxm, nym = N.unravel_index(N.abs(imgf).argmax(), imgf.shape)
        phi0 = N.angle(imgf[nxm, nym])
        return N.exp(-1j * phi0) * imgf

def phaseest(img):
    ni, nx, ny = img.shape
    phi = 2 * pi / 3
    if ni == 9:
        q = N.zeros((3, nx, ny))
        for m in range(3):
            temp = img[3 * m] + N.exp(1j * phi) * img[3 * m + 1] + N.exp(2j * phi) * img[3 * m + 2]
            q[m] = N.abs(N.fft.fftshift(fft2(temp)))
    if ni == 3:
        q = N.zeros((nx, ny))
        temp = img[0] + N.exp(1j * phi) * img[1] + N.exp(2j * phi) * img[2]
        q = N.abs(N.fft.fftshift(fft2(temp)))
    return q

def conv(a, b):
    af = fft2(a.astype(N.complex64))
    bf = fft2(b.astype(N.complex64))
    cf = af * bf.conj()
    c = ifft2(cf) / N.prod(a.shape)
    return c
