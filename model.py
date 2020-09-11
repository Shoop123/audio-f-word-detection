import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

import preprocess, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# THIS CODE IS A MODIFICATION UPON SETH ADAMS' FROM HIS AUDIO CLASSIFICATION SERIES FOUND HERE: https://github.com/seth814/Audio-Classification

NUM_CLASSES = 2

N_DFT = 2048
HOP_LENGTH = 64
N_MELS = 512

def conv1d(input_shape, sr):
	i = layers.Input(shape=input_shape, name='input')
	x = Melspectrogram(n_dft=N_DFT, n_hop=HOP_LENGTH,
						padding='same', sr=sr, n_mels=N_MELS,
						fmin=0.0, fmax=sr/2, power_melgram=1.0,
						return_decibel_melgram=True, trainable_fb=False,
						trainable_kernel=False,
						name='melbands')(i)
	x = Normalization2D(str_axis='batch', name='batch_norm')(x)
	x = layers.Permute((2,1,3), name='permute')(x)
	x = TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='tanh'), name='td_conv_1d_tanh')(x)
	x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_1')(x)
	x = TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_1')(x)
	x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_2')(x)
	x = TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_2')(x)
	x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_3')(x)
	x = TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_3')(x)
	x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_4')(x)
	x = TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_4')(x)
	x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_5')(x)
	x = TimeDistributed(layers.Conv1D(256, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_5')(x)
	x = TimeDistributed(layers.Conv1D(512, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_6')(x)
	x = layers.GlobalMaxPooling2D(name='global_max_pooling_2d')(x)
	x = layers.Dropout(rate=0.2, name='dropout')(x)
	x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
	o = layers.Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

	model = Model(inputs=i, outputs=o, name='1d_convolution')
	model.compile(optimizer='adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])

	return model

def conv2d(input_shape, sr):
	i = layers.Input(shape=input_shape, name='input')
	x = Melspectrogram(n_dft=N_DFT, n_hop=HOP_LENGTH,
		padding='same', sr=sr, n_mels=N_MELS,
		fmin=0.0, fmax=sr/2, power_melgram=1.0,
		return_decibel_melgram=True, trainable_fb=False,
		trainable_kernel=False,
		name='melbands')(i)
	x = Normalization2D(str_axis='batch', name='batch_norm')(x)
	x = layers.Conv2D(16, kernel_size=(3,3), activation='tanh', padding='same', name='conv2d_tanh')(x)
	x = layers.MaxPooling2D(pool_size=(4,4), padding='same', name='max_pool_2d_1')(x)
	x = layers.Conv2D(24, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
	x = layers.MaxPooling2D(pool_size=(4,4), padding='same', name='max_pool_2d_2')(x)
	# x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
	# x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
	# x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
	# x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
	# x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
	# x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_5')(x)
	x = layers.Flatten(name='flatten')(x)
	x = layers.Dropout(rate=0.2, name='dropout')(x)
	x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
	o = layers.Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

	model = Model(inputs=i, outputs=o, name='2d_convolution')
	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	return model

def lstm(input_shape, sr):
	i = layers.Input(shape=input_shape, name='input')
	x = Melspectrogram(n_dft=N_DFT, n_hop=HOP_LENGTH,
		padding='same', sr=sr, n_mels=N_MELS,
		fmin=0.0, fmax=sr/2, power_melgram=2.0,
		return_decibel_melgram=True, trainable_fb=False,
		trainable_kernel=False,
		name='melbands')(i)
	x = Normalization2D(str_axis='batch', name='batch_norm')(x)
	x = layers.Permute((2,1,3), name='permute')(x)
	x = TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
	s = TimeDistributed(layers.Dense(64, activation='tanh'),
		name='td_dense_tanh')(x)
	x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
		name='bidirectional_lstm')(s)
	x = layers.concatenate([s, x], axis=2, name='skip_connection')
	x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
	x = layers.MaxPooling1D(name='max_pool_1d')(x)
	# x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
	x = layers.Flatten(name='flatten')(x)
	x = layers.Dropout(rate=0.2, name='dropout')(x)
	x = layers.Dense(32, activation='relu',
		activity_regularizer=l2(0.001),
		name='dense_3_relu')(x)
	o = layers.Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

	model = Model(inputs=i, outputs=o, name='long_short_term_memory')
	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	return model

# data_chunks = preprocess.load_chunks('no_haters.wav', 1)
# x_train = data_chunks[0]

# x_train = x_train.reshape(1, 1, x_train.shape[0])
# y_train = np.array((
# 	(1, 0),
# ))

x_train, y_train = preprocess.generate_training_set()

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.15)

print(x_train.shape, y_train.shape)

model = conv2d(x_train.shape[1:], x_train.shape[2])

model.summary()

model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))