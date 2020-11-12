#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import wave
import math
import random
from collections import deque

import numpy as np
import scipy.fft, scipy.signal
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as anim


SAMPLING_FREQ = 44100 # Hertz, taux d'échantillonnage standard des CD
SAMPLE_WIDTH = 16 # Échantillons de 16 bit
MAX_INT_SAMPLE_VALUE = 2**(SAMPLE_WIDTH-1) - 1


def merge_channels(channels):
	# Équivalent de :  [sample for samples in zip(*channels) for sample in samples]
	return np.fromiter((sample for samples in zip(*channels) for sample in samples), float)

def separate_channels(samples, num_channels):
	return [samples[i::num_channels] for i in range(num_channels)]

def generate_sample_time_points(duration):
	# Générer un tableau de points temporels également espacés en seconde. On a SAMPLING_FREQ points par seconde.
	return np.linspace(0, duration, int(duration * SAMPLING_FREQ))

def sine(freq, amplitude, duration):
	# Générer une onde sinusoïdale à partir de la fréquence et de l'amplitude donnée, sur le temps demandé et considérant le taux d'échantillonnage.
	# Formule de la valeur y d'une onde sinusoïdale à l'angle x en fonction de sa fréquence F et de son amplitude A :
	# y = A * sin(F * x), où x est en radian.
	# Si on veut le x qui correspond au moment t, on peut dire que 2π représente une seconde, donc x = t * 2π,
	# Or t est en secondes, donc t = i / nb_échantillons_par_secondes, où i est le numéro d'échantillon.

	# y = A * sin(F * 2π*t)
	time_points = generate_sample_time_points(duration)
	return amplitude * np.sin(freq * 2 * np.pi * time_points)

def square(freq, amplitude, duration):
	# Générer une onde carrée d'une fréquence et amplitude donnée.
	# y = A * sgn(sin(F * 2π*t))
	return amplitude * np.sign(sine(freq, 1, duration))

def sine_with_overtones(root_freq, amplitude, overtones, duration):
	# Générer une onde sinusoïdale avec ses harmoniques. Le paramètre overtones est une liste de tuple où le premier élément est le multiple de la fondamentale et le deuxième élément est l'amplitude relative de l'harmonique.
	# On bâtit un signal avec la fondamentale
	signal = sine(root_freq, amplitude, duration)
	# Pour chaque harmonique (overtone en Anglais), on a un facteur de fréquence et un facteur d'amplitude :
	for freq_factor, amp_factor in overtones:
		# Construire le signal de l'harmonique en appliquant les deux facteurs.
		overtone = sine(root_freq * freq_factor, amplitude * amp_factor, duration)
		# Ajouter l'harmonique au signal complet.
		np.add(signal, overtone, out=signal)
	return signal

def normalize(samples, norm_target):
	# Normalisez un signal à l'amplitude donnée
	# 1. il faut trouver l'échantillon le plus haut en valeur absolue
	abs_samples = np.abs(samples)
	max_sample = max(abs_samples)
	# 2. Calcule coefficient entre échantillon max et la cible
	coeff = norm_target / max_sample
	# 3. Applique mon coefficient
	normalized_samples = coeff * samples
	return normalized_samples

def convert_to_bytes(samples):
	# Convertir les échantillons en tableau de bytes en les convertissant en entiers 16 bits.
	# Les échantillons en entrée sont entre -1 et 1, nous voulons les mettre entre -MAX_INT_SAMPLE_VALUE et MAX_INT_SAMPLE_VALUE
	# Juste pour être certain de ne pas avoir de problème, on doit clamper les valeurs d'entrée entre -1 et 1.
	# 1. Limiter (ou clamp/clip) les échantillons entre -1 et 1
	clipped = np.clip(samples, -1, 1)
	# 2. convertir en entier 16-bit signés
	int_samples = (clipped * MAX_INT_SAMPLE_VALUE).astype("<i2")
	# 3. convertir en bytes
	sample_bytes = int_samples.tobytes()
	# Retourne le tout.
	return sample_bytes

def convert_to_samples(bytes):
	# Faire l'opération inverse de convert_to_bytes, en convertissant des échantillons entier 16 bits en échantillons réels
	# 1. Convertir en numpy array du bon type (entier 16 bit signés)
	int_samples = np.frombuffer(bytes, dtype="<i2")
	# 2. Convertir en réel dans [-1, 1]
	samples = int_samples.astype(float) / MAX_INT_SAMPLE_VALUE
	return samples

def apply_fft(sig):
	freq_axis = np.linspace(0, SAMPLING_FREQ//2, sig.size//2)
	val_axis = np.abs(sp.fft.fft(sig)[:freq_axis.size]) / (sig.size / 2)
	return val_axis, freq_axis

def spectrogram(sig, fft_size, window=None):
	for i in range(0, sig.size - sig.size % fft_size, fft_size):
		subsignal = sig[i:i+fft_size]
		if window is not None:
			subsignal *= sp.signal.get_window(window, subsignal.size)
		yield apply_fft(subsignal)

def build_spectrogram_animation(sig, fft_size, x_range=(0, SAMPLING_FREQ/2), y_range=(0, 1)):
	def animate_spectrogram(frame, fig, graph, line, spec):
		try:
			y, x = next(spec)
		except StopIteration:
			plt.close(fig)
			y, x = [], []

		line.set_xdata(x)
		line.set_ydata(y)
		fig.canvas.draw()
		fig.canvas.flush_events()

	fig = plt.figure("spectro")
	graph = fig.add_subplot(1, 1, 1)
	graph.set_xscale("log")
	graph.set_xlim(*x_range)
	graph.set_ylim(*y_range)
	line = graph.plot([], [])[0]
	spec = spectrogram(sig, fft_size, "hann")
	return anim.FuncAnimation(fig, animate_spectrogram, fargs=(fig, graph, line, spec), interval=1000/(SAMPLING_FREQ/fft_size))


def main():
	try:
		os.mkdir("output")
	except:
		pass

	with wave.open("data/stravinsky.wav", "rb") as reader:
		data = reader.readframes(reader.getnframes())
		samples = separate_channels(convert_to_samples(data), reader.getnchannels())[0]
		ani = build_spectrogram_animation(samples, 2048, (20, 10_000), (0, 0.3))
		plt.show()

if __name__ == "__main__":
	main()
