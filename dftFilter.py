import numpy as np


def nearest(array, val):
    index = (np.abs(array - val)).argmin()
    return index


def ft_preproc_dftfilter(data=None, Fs=None, Fl=None, dftreplace=None,
                         dftbandwidth=None, dftneighbourwidth=None):
    """
    FT_PREPROC_DFTFILTER reduces power line noise (50 or 60Hz) via two
    alternative methods:
    A) DFT filter (Flreplace = 'zero') or
    B) Spectrum Interpolation (Flreplace = 'neighbour').

    A) The DFT filter applies a notch filter to the data to remove the 50Hz
    or 60Hz line noise components ('zeroing'). This is done by fitting a sine
    and cosine at the specified frequency to the data and subsequently
    subtracting the estimated components. The longer the data is, the sharper
    the spectral notch will be that is removed from the data.
    Preferably the data should have a length that is a multiple of the
    oscillation period of the line noise (i.e. 20ms for 50Hz noise). If the
    data is of different lenght, then only the first N complete periods are
    used to estimate the line noise. The estimate is subtracted from the
    complete data.

    B) Alternatively line noise is reduced via spectrum interpolation
    (Leske & Dalal, 2019, NeuroImage 189,
     doi: 10.1016/j.neuroimage.2019.01.026)
    The signal is:
    I)   transformed into the frequency domain via a discrete Fourier
          transform (DFT),
    II)  the line noise component (e.g. 50Hz, Flwidth = 1 (±1Hz): 49-51Hz) is
          interpolated in the amplitude spectrum by replacing the amplitude
          of this frequency bin by the mean of the adjacent frequency bins
          ('neighbours', e.g. 49Hz and 51Hz).
          Neighwidth defines frequencies considered for the mean (e.g.
          Neighwidth = 2 (±2Hz) implies 47-49 Hz and 51-53 Hz).
          The original phase information of the noise frequency bin is
          retained.
    III) the signal is transformed back into the time domain via inverse DFT
          (iDFT).
    If Fline is a vector (e.g. [50 100 150]), harmonics are also considered
    Preferably the data should be continuous or consist of long data segments
    (several seconds) to avoid edge effects. If the sample rate and the data
    length are such, that a full cycle of the line noise and the harmonics
    fit in the data and if the line noise is stationary (e.g. no variations
    in amplitude or frequency), then spectrum interpolation can also be
    applied to short trials. But it should be used with caution and checked
    for edge effects.

    Use as
      filt = ft_preproc_dftfilter(data, Fsample, Fline, varargin)
    where
      data             data matrix (Nchans X Ntime)
      Fsample         sampling frequency in Hz
      Fline           line noise frequency (and harmonics)

    Additional input arguments come as key-value pairs:

      Flreplace       'zero' or 'neighbour', method used to reduce line
                      noise, 'zero' implies DFT filter, 'neighbour'
                      implies spectrum interpolation
      Flwidth         bandwidth of line noise frequencies, applies to
                      spectrum interpolation, in Hz
      Neighwidth      width of frequencies neighbouring line noise
                      frequencies, applies to spectrum interpolation
                      (Flreplace = 'neighbour'), in Hz

    The line frequency should be specified as a single number
    for the DFT filter. If omitted, a European default of
    50Hz will be assumed

    Undocumented option:
      Fline can be a vector, in which case the regression is done for all
      frequencies in a single shot. Prerequisite is that the requested
      frequencies all fit with an integer number of cycles in the data.

    Copyright (C) 2003, Pascal Fries
    Copyright (C) 2003-2015, Robert Oostenveld
    Copyright (C) 2016, Sabine Leske

    This function was adapted for python by James Lubell in 2019 at
    Aarhus University.  Contact at:
    www.github.com/lubell
    or
    jlubell 'at' cfin 'dot' au 'dot' com
    """

    # determine the size of the data
    nchans, nsamples = data.shape
    # set the default filter frequency
    if Fl is None:
        Fl = 50

    # FL should be a column  vector
    if dftreplace is None:
        Flreplace = 'zero'
    else:
        Flreplace = dftreplace
    if dftbandwidth is None:
        Flwidth = np.array([1, 2, 3])
    else:
        Flwidth = dftbandwidth
    if dftneighbourwidth is None:
        Neighwidth = np.array([2, 2, 2])
    else:
        Neighwidth = dftneighbourwidth

    # preprocessing fails on channels that contain NaN
    if np.any(np.isnan(data)):
        raise ValueError('Data contains NaN values')

    # Method A): DFT filter
    if Flreplace == 'zero':
        # determine the largest integer number of line-noise
        # cycles that fits in the data
        n = np.round((nsamples * Fl // Fs) * Fs // Fl)
        if np.all(n == n[0]):
            # make a selection of samples
            # such that the line-noise fits the data
            sel = range(int(n[0]))
        else:
            # the different frequencies require different
            # numbers of samples, apply the filters sequentially
            filt = data
            for i in range(Fl.size):
                filt = ft_preproc_dftfilter(data, Fs, Fl[i], dftreplace,
                                            dftbandwidth, dftneighbourwidth)
        # temporarily remove mean to avoid leakage
        meandat = np.mean(data[:, sel], axis=1)
        for i in range(nsamples):
            # demean the data
            data[:, i] = data[:, i] - meandat

        # fit a sine and cosine to each channel in the data and subtract them
        time = np.arange(0, nsamples).reshape((1, nsamples)) / Fs
        # complex sin and cos
        tmp = np.exp(1j * 2 * np.pi * Fl.reshape((3, 1)).dot(time))
        # estimated amplitude of complex sin and cos
        # on integer number of cycles
        ampl = 2 * np.dot(data[:, sel], np.linalg.pinv(tmp[:, sel]))
        # estimated signal at this frequency
        est = np.matmul(ampl, tmp)
        filt = data - est  # subtract estimated signal
        filt = np.real(filt)

        for i in range(nsamples):
            # add the mean back to the filtered data
            filt[:, i] = filt[:, i] + meandat
    elif Flreplace == 'neighbour':
        # error message if periodicity of the interference
        # frequency doesn't match the DFT length
        n = np.round(np.floor(nsamples * Fl / Fs) * Fs / Fl)

        if np.all(n != nsamples):
            errorMsg = ('Spectrum interpolation requires that the data length'
                        ' fits complete cycles of the powerline frequency,'
                        ' e.g., exact multiples of 20 ms for a 50 Hz line'
                        'frequency (sampling rate of 1000 Hz).')
            raise ValueError(errorMsg)
        if (Fl.size != len(Flwidth)) or (Fl.size != len(Neighwidth)):
            errorMsg2 = ('The number of frequencies to interpolate'
                         ' (cfg.dftfreq) should be the same as the number of'
                         ' bandwidths (cfg.dftbandwidth)'
                         ' and bandwidths of neighbours (cfg.neighbourwidth)')
            raise ValueError(errorMsg2)

        # frequencies to interpolate
        f2int = np.zeros([Fl.size, 2])
        f4int = np.zeros([len(Neighwidth), 4])
        for i in range(Fl.size):
            f2int[i] = [Fl[i] - Flwidth[i], Fl[i] + Flwidth[i]]
        # frequencies used for interpolation
        # f4int = np.array()
        for i in range(len(Neighwidth)):
            f4int[i] = [f2int[i, 0] - Neighwidth[i], f2int[i, 0], f2int[i, 1],
                        f2int[i, 1] + Neighwidth[i]]

        # calculate fft to obtain spectrum that will be interpolated
        data_fft = np.fft.fft(data, nsamples, 1)
        frq = Fs * np.linspace(0, 1, nsamples + 1)

        # interpolate 50Hz (and harmonics) amplitude in spectrum
        for i in range(Fl.size):
            # samples of frequencies that will be interpolated
            smpl2int = np.arange(nearest(frq, f2int[i, 0]),
                                 nearest(frq, f2int[i, 1]))
            # samples of neighbouring frequencies used to calculate the mean
            smplIndicesOne = np.arange(nearest(frq, f4int[i, 0]),
                                       nearest(frq, f4int[i, 1]) - 1)
            smplIndicesTwo = np.arange(nearest(frq, f4int[i, 2]) + 1,
                                       nearest(frq, f4int[i, 3]))
            smpl4int = np.concatenate((smplIndicesOne, smplIndicesTwo))

            # new amplitude is calculated as the mean of
            # the neighbouring frequencies
            l, m = np.shape(data_fft[:, smpl2int])
            A = np.ones((l, m)).transpose()
            B = np.mean(np.abs(data_fft[:, smpl4int]), axis=1)
            mns4int = A * B
            # Eulers formula: replace noise components with
            # new mean amplitude combined with phase, that
            # is retained from the original data
            AA = np.exp(np.multiply(np.angle(data_fft[:, smpl2int]), 1j))
            data_fft[:, smpl2int] = np.multiply(AA, mns4int.transpose())

        # complex fourier coefficients are transformed back into time domin,
        # fourier coefficients are treated as conjugate 'symmetric'
        # to ensure a real valued signal after iFFT
        print(data_fft.shape)
        # filt = np.real(np.fft.ifft(data_fft, n=None, axis=1))
        idx = int(nsamples / 2 + 1)
        filt = np.fft.irfft(data_fft[:, 0:idx], n=None, axis=1)

    return filt
