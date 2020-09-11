# -*- coding: utf-8 -*-
"""
================
Vocal separation
================

This notebook demonstrates a simple technique for separating vocals (and
other sporadic foreground signals) from accompanying instrumentation.

This is based on the "REPET-SIM" method of `Rafii and Pardo, 2012
<http://www.cs.northwestern.edu/~zra446/doc/Rafii-Pardo%20-%20Music-Voice%20Separation%20using%20the%20Similarity%20Matrix%20-%20ISMIR%202012.pdf>`_, but includes a couple of modifications and extensions:

		- FFT windows overlap by 1/4, instead of 1/2
		- Non-local filtering is converted into a soft mask by Wiener filtering.
			This is similar in spirit to the soft-masking method used by `Fitzgerald, 2012
			<http://arrow.dit.ie/cgi/viewcontent.cgi?article=1086&context=argcon>`_,
			but is a bit more numerically stable in practice.
"""

# Code source: Brian McFee
# License: ISC

##################
# Standard imports
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

def separate_vocals(y, sr):
	# And compute the spectrogram magnitude and phase
	S_full, phase = librosa.magphase(librosa.stft(y))

	###########################################################
	# The wiggly lines above are due to the vocal component.
	# Our goal is to separate them from the accompanying
	# instrumentation.
	#

	# We'll compare frames using cosine similarity, and aggregate similar frames
	# by taking their (per-frequency) median value.
	#
	# To avoid being biased by local continuity, we constrain similar frames to be
	# separated by at least 2 seconds.
	#
	# This suppresses sparse/non-repetetitive deviations from the average spectrum,
	# and works well to discard vocal elements.

	S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))

	# The output of the filter shouldn't be greater than the input
	# if we assume signals are additive.  Taking the pointwise minimium
	# with the input spectrum forces this.
	S_filter = np.minimum(S_full, S_filter)


	##############################################
	# The raw filter output can be used as a mask,
	# but it sounds better if we use soft-masking.

	# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
	# Note: the margins need not be equal for foreground and background separation
	margin_v = 10
	power = 2

	mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)

	# Once we have the masks, simply multiply them with the input spectrum
	# to separate the components

	S_foreground = mask_v * S_full

	return librosa.istft(S_foreground)