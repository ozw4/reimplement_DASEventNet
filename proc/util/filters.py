import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
	data: np.ndarray,
	fs: float = 1000,
	lowcut: float = 25,
	highcut: float = 150,
	order: int = 4,
) -> np.ndarray:
	"""Apply a Butterworth band-pass filter.

	Parameters
	----------
	data : np.ndarray
	Input array ``(channels, samples)``.
	fs : float, default 1000
	Sampling rate in hertz.
	lowcut : float, default 25
	Low cutoff frequency in hertz.
	highcut : float, default 150
	High cutoff frequency in hertz.
	order : int, default 4
	Butterworth filter order.

	Returns
	-------
	np.ndarray
	Filtered array with the same shape as ``data``.

	"""
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return filtfilt(b, a, data, axis=1)
