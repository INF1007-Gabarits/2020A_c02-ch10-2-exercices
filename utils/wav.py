#!/usr/bin/env python
# -*- coding: utf-8 -*-


import wave
import numpy as np


SAMPLE_WIDTH = 16 # Échantillons de 16 bit
MAX_INT_SAMPLE_VALUE = 2**(SAMPLE_WIDTH-1) - 1


def merge_channels(channels):
	# Équivalent de :  [sample for samples in zip(*channels) for sample in samples]
	return np.fromiter((sample for samples in zip(*channels) for sample in samples), float)

def separate_channels(samples, num_channels):
	return [samples[i::num_channels] for i in range(num_channels)]

def convert_to_bytes(samples):
	# Convertir les échantillons en tableau de bytes en les convertissant en entiers 16 bits.
	# Les échantillons en entrée sont entre -1 et 1, nous voulons les mettre entre -MAX_INT_SAMPLE_VALUE et MAX_INT_SAMPLE_VALUE
	# Juste pour être certain de ne pas avoir de problème, on doit clamper les valeurs d'entrée entre -1 et 1.
	# 1. Limiter (ou clamp/clip) les échantillons entre -1 et 1
	clipped = np.clip(samples, -1, 1)
	# 2. convertir en entier 16-bit signés
	int_samples = (clipped * MAX_INT_SAMPLE_VALUE).astype("<i" + str(SAMPLE_WIDTH//8))
	# 3. convertir en bytes
	sample_bytes = int_samples.tobytes()
	# Retourne le tout.
	return sample_bytes

def convert_to_samples(bytes):
	# Faire l'opération inverse de convert_to_bytes, en convertissant des échantillons entier 16 bits en échantillons réels
	# 1. Convertir en numpy array du bon type (entier 16 bit signés)
	int_samples = np.frombuffer(bytes, dtype="<i"+str(SAMPLE_WIDTH//8))
	# 2. Convertir en réel dans [-1, 1]
	samples = int_samples.astype(float) / MAX_INT_SAMPLE_VALUE
	return samples

def save_wav(samples, filename, num_channels, sampling_rate):
	with wave.open(filename, "wb") as writer:
		writer.setnchannels(num_channels)
		writer.setsampwidth(SAMPLE_WIDTH // 8)
		writer.setframerate(sampling_rate)
		writer.writeframes(convert_to_bytes(samples))

def load_wav(filename):
	with wave.open(filename, "rb") as reader:
		data = reader.readframes(reader.getnframes())
		sampling_rate = reader.getframerate()
		channels = separate_channels(convert_to_samples(data), reader.getnchannels())
	return channels, sampling_rate

