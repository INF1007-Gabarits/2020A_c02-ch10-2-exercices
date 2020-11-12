#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import wave
import math
import threading as th
import time

import numpy as np
import scipy.fft, scipy.signal
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.widgets as wid
import matplotlib.gridspec as grid
from playsound import playsound

from utils.wav import *
from utils.signals import *
from utils.fft import *


SAMPLING_FREQ = 44100


def build_spectrogram_animation(filename, fft_size, x_range=None, y_range=None):
	# Fonction qui se fait appelée périodiquement pour redissiner le graphique
	def draw_frame(frame, fig, graph, line, spec):
		global playing
		if not playing:
			fig.canvas.draw()
			fig.canvas.flush_events()
			return

		# TODO: Mettre dans la variable x l'axe fréquentiel et dans y l'axe de valeurs de la prochaine itération du spectrogramme.

		# TODO: S'il ne reste rien à traiter, on ferme le graphique avec plt.close(fig) et on met des listes vides dans x et y

		# On met à jour seulement les données des lignes (avec nos deux axes) et on redesinne le graphique.
		line.set_xdata(x)
		line.set_ydata(y)
		fig.canvas.draw()
		fig.canvas.flush_events()

	# TODO: Charger le fichier, le mixer (en normalisant) et créer son spectrogramme. On utilse une fenêtre de Hanning (on passe "hann")

	# Création de la figure en laissant de l'espace en bas pour des boutons (ou autres)
	fig = plt.figure("Spectrogram")
	gs = grid.GridSpec(2, 1, height_ratios=(6, 1), figure=fig)

	# Création du graphe dans l'espace du haut.
	graph = fig.add_subplot(gs[0, 0])
	# TODO : Appliquer une échelle logarithmique à l'axe des X.

	# TODO : Contraindre les valeurs des axes si `x_range` ou `y_range` ne sont pas vides.
	if x_range is not None:
		pass #TODO
	if y_range is not None:
		pass #TODO

	# Création de la courbe qui va dessiner la FFT.
	line = graph.plot([], [])[0]

	refresh_period_ms = 1000 / (fps / fft_size)
	return fig, anim.FuncAnimation(fig, draw_frame, fargs=(fig, graph, line, spec), interval=refresh_period_ms)

def wait_and_play(filename):
	global playing
	while not playing:
		# On fait une pause pour relâcher le contrôle sur le processus.
		time.sleep(0.01)
	playsound(filename, block=False)


playing = False # Contrôle le départ du dessin et de la musique.


def main():
	try:
		os.mkdir("output")
	except:
		pass

	set_signal_gen_sampling_rate(SAMPLING_FREQ)

	# Un accord majeur (racine, tierce, quinte, octave) en intonation juste
	root_freq = 220
	root = sine(root_freq, 1, 2.0)
	third = sine(root_freq * 5/4, 1, 2.0)
	fifth = sine(root_freq * 3/2, 1, 2.0)
	octave = sine(root_freq * 2, 1, 2.0)
	notes = (root, third, fifth, octave)
	# On plaque et on arpège.
	block_chord = normalize(root + third + fifth + octave, 0.89)
	arpeggio = normalize(np.concatenate([e[:len(e)//2] for e in notes]), 0.89)

	save_wav(block_chord, "output/major_chord.wav", 1, SAMPLING_FREQ)
	save_wav(arpeggio, "output/major_chord_arpeggio.wav", 1, SAMPLING_FREQ)

	# TODO: Afficher la FFT de `block_chord` dans une fenêtre.
	
	# TODO: Pour chaque note générée précédemment (dans `notes`), afficher sa FFT. On veut ici les afficher indépendamment, mais sur le même graphique.
	

	wav_filename = "data/stravinsky.wav"

	# Création de l'animation. On contraint ici nos axes pour visionner le domaine intéressant des données.
	fig, ani = build_spectrogram_animation(wav_filename, 4096, (20, 10_000), (0, 0.2))
	
	# Création bouton qui part le dessin et la musique.
	btn_pos = fig.add_axes([0.8, 0.05, 0.15, 0.10])
	def start_play(event):
		global playing
		playing = True
	btn = wid.Button(btn_pos, "START")
	btn.on_clicked(start_play)

	# Création du thread qui part la musique quand on appuie sur le bouton.
	p = th.Thread(target=wait_and_play, args=(wav_filename,))
	p.start()

	plt.show()
	p.join()

if __name__ == "__main__":
	main()

