#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math

import numpy as np
import scipy.fft, scipy.signal
import scipy as sp


def apply_fft(sig, sampling_rate):
	"""
	Applique une tranformée de Fourier discrète rapide (FFT) sur un signal 1D.

	:param sig: Le signal, en numpy.ndarray réel.

	:param sampling_rate: Le taux d'échantillonnage, en Hz, du signal.

	:returns: L'axe de magnitude normalisée de la FFT (partie réelle seulement) et axe fréquentiel associé.
	"""

	# TODO: Créer l'axe fréquentiel approprié.
	#       On veut un tableau ainsi :
	#         - Valeurs réelles espacées uniformément
	#         - Taille = moitié de la longueur du signal
	#         - Première valeur = 0, dernière valeur = taux d'échantillonnage / 2 (fréquence de Nyquist du signal)

	# TODO: Créer l'axe de magnitude en appliquant une FFT.
	#       On veut un axe ainsi :
	#         - Même taille que l'axe de fréquence, donc on prend juste la première moitié des valeurs retournées par `scipy.fft.fft`, c'est-à-dire la partie réelle de la FFT.
	#         - En valeurs absolues (les valeurs négatives sont des résultats déphasés)
	#         - On normalise en divisant par la moitié du nombre d'échantillons (taille du signal)

	# On retourne les deux axes, avec l'axe de magnitude en premier
	pass

def spectrogram(sig, fft_size, sampling_rate, window=None):
	"""
	Génère un spectrogramme en appliquant une FFT par tranche sur un signal. Chaque itération 

	:param sig: Le signal, en :code:`numpy.ndarray` réel.

	:param fft_size: Taille, en nombre d'échantillons, des tranches à analyser
	
	:param sampling_rate: Le taux d'échantillonnage, en Hz, du signal.
	
	:param window: La fenêtre à appliquer sur les tranches. Soit :code:`None` ou une valeur dans https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html

	:yields: L'axe de valeur et l'axe fréquentiel d'une tranche
	"""

	# Exemple de tranche :
	#    Soit le signal [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (11 échantillons),
	#    Si on sépare en tranches de 3, on aurait les tranches [0, 1, 2], [3, 4, 5] et [6, 7, 8]
	#    Les derniers éléments qui ne sont pas assez nombreux pour faire une tranche peuvent être ignorés.

	# TODO: Pour chaque tranche :
		# TODO : Extraire les échantillons de la tranche.
		
		# TODO : Appliquer la fenêtre (si `window` pas vide).
		
		# TODO : Donner le résultat de la FFT sur la tranche.
		
	pass

