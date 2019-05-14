# python_dftfilter
This is an adaptation of the fieldtrip preprocessing function ft_preproc_dftfilter into python

Credit to:
Copyright (C) 2003, Pascal Fries
Copyright (C) 2003-2015, Robert Oostenveld
Copyright (C) 2016, Sabine Leske

Currently both the dft filter and the spectrum interpolation are working.
The dft filter is quite slow and should be optimized.

Plot showing the difference betweem outputs when using np.fft.irfft and np.fft.ifft when they are compared
to the matlab output.

![plot compare](https://github.com/Lubell/python_dftfilter/blob/master/comparisonofIFFT.png)

The dft filter results are almost exactly the same as the matlab code's, the only difference is precision, when rounded to 4 places (only precision I looked at) the results were identical.
