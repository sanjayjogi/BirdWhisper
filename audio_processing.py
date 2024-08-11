import noisereduce as nr
from scipy import signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reduce_noise(y, sr):
    try:
        logger.info("Starting noise reduction")
        reduced_noise = nr.reduce_noise(y=y, sr=sr, thresh_n_mult_nonstationary=2, stationary=False)
        logger.info("Noise reduction completed")
        return f_high(reduced_noise, sr)
    except Exception as e:
        logger.error(f"Error reducing noise: {e}")
        raise

def f_high(y, sr):
    try:
        logger.info("Applying high-pass filter")
        b, a = signal.butter(10, 2000 / (sr / 2), btype='highpass')
        yf = signal.lfilter(b, a, y)
        logger.info("High-pass filter applied")
        return yf
    except Exception as e:
        logger.error(f"Error applying high-pass filter: {e}")
        raise
