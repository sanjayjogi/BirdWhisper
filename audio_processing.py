import noisereduce as nr
from scipy import signal

def reduce_noise(y, sr):
    reduced_noise = nr.reduce_noise(y=y, sr=sr, thresh_n_mult_nonstationary=2, stationary=False)
    return f_high(reduced_noise, sr)

def f_high(y, sr):
    b, a = signal.butter(10, 2000 / (sr / 2), btype='highpass')
    yf = signal.lfilter(b, a, y)
    return yf
