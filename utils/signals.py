#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math

import numpy as np


_signals_sampling_rate = 44100


def set_signal_gen_sampling_rate(sampling_rate):
	_signals_sampling_rate = sampling_rate

def generate_sample_time_points(duration):
	# Générer un tableau de points temporels également espacés en seconde. On a SAMPLING_FREQ points par seconde.
	return np.linspace(0, duration, int(duration * _signals_sampling_rate))

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

def mix_signals(signals, norm_target=None):
	result = signals[0]
	for sig in signals[1:]:
		result += sig
	return result if norm_target is None else normalize(result, norm_target)

