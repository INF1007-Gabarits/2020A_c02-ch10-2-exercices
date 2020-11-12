#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math

import numpy as np
import scipy.fft, scipy.signal
import scipy as sp


def apply_fft(sig, sampling_rate):
	freq_axis = np.linspace(0, sampling_rate // 2, sig.size // 2)
	val_axis = np.abs(sp.fft.fft(sig)[:freq_axis.size]) / (sig.size / 2)
	return val_axis, freq_axis

def spectrogram(sig, fft_size, sampling_rate, window=None):
	for i in range(0, sig.size - sig.size % fft_size, fft_size):
		subsignal = sig[i:i+fft_size]
		if window is not None:
			subsignal *= sp.signal.get_window(window, subsignal.size)
		yield apply_fft(subsignal, sampling_rate)

