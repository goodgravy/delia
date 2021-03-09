# Experiment training an Audio -> Image -> Audio CycleGAN
import logging
import math
from random import random

import numpy
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import UpSampling1D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Reshape
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO)

def define_image_discriminator(image_shape):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)

	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)

	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	model = Model(in_image, patch_out)
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model

# Naively take the same kind of approach for 1D data as that for 2D image data
def define_audio_discriminator(audio_shape):
	init = RandomNormal(stddev=0.02)
	in_audio = Input(shape=audio_shape)

	# 1-dimension C64
	d = Conv1D(64, 4, strides=2, padding='same', kernel_initializer=init)(in_audio)
	d = LeakyReLU(alpha=0.2)(d)
	# 1-dimension C128
	d = Conv1D(128, 4, strides=2, padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# 1-dimension C256
	d = Conv1D(256, 4, strides=2, padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# 1-dimension C512
	d = Conv1D(512, 4, strides=2, padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# 1-dimension C512
	d = Conv1D(512, 4, padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)

	patch_out = Conv1D(1, 4, padding='same', kernel_initializer=init)(d)
	model = Model(in_audio, patch_out)
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model

# generator a resnet block to work on 1D data
def resnet_1d_block(n_filters, input_layer):
	init = RandomNormal(stddev=0.02)
	g = Conv1D(n_filters, 3, padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	g = Conv1D(n_filters, 3, padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)

	g = Concatenate()([g, input_layer])
	return g

# generator a resnet block to work on 2D data
def resnet_2d_block(n_filters, input_layer):
	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)

	g = Concatenate()([g, input_layer])
	return g

# I take audio as input, and generate an image as output
def define_image_generator(audio_shape, n_resnet=9):
	init = RandomNormal(stddev=0.02)
	in_audio = Input(shape=audio_shape)

	axis_size = int(math.sqrt(audio_shape[0]))
	reshaped = Reshape((axis_size, axis_size, 2))(in_audio)

	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(reshaped)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for i in range(n_resnet):
		g = resnet_2d_block(256, g)

	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)

	model = Model(in_audio, out_image)
	return model

# I take pixel data as input, and generate audio as output.
# This implementation is based on the original CycleGAN generator,
# and we transform to 1D audio data at the end (cf. the image generator which
# transforms FROM 1D audio data at the beginning.
def define_audio_generator(image_shape, n_resnet=9):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)

	# c7s1-64
	g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for i in range(n_resnet):
		g = resnet_2d_block(256, g)

	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)

	# c7s1-2: two filters for two audio channels
	g = Conv2D(2, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_data = Activation('tanh')(g)

	# Naively reshape the image data into 1D audio data
	axis_size = image_shape[0] ** 2
	out_audio = Reshape((axis_size, 2))(g)

	model = Model(in_image, out_audio)
	return model

# define a composite model for updating generators by adversarial and cycle loss
# Here:
# A is the domain we are transforming from
# B is the domain we are transforming to
# (more or less: the cyclic transformations mean that we go back and forth)
def define_composite_model(g_model_AB, d_model_B, g_model_BA, input_shape_A, input_shape_B):
	# ensure the model we're updating is trainable
	g_model_AB.trainable = True
	# mark discriminator as not trainable
	d_model_B.trainable = False
	# mark other generator model as not trainable
	g_model_BA.trainable = False

	input_A = Input(shape=input_shape_A)
	input_B = Input(shape=input_shape_B)

	# discriminator element
	genAB_out = g_model_AB(input_A)
	output_d = d_model_B(genAB_out)

	# forward cycle
	output_ABA = g_model_BA(genAB_out)

	# backward cycle
	genBA_out = g_model_BA(input_B)
	output_BAB = g_model_AB(genBA_out)

	# define model graph
	model = Model([input_A, input_B], [output_d, output_ABA, output_BAB])

	# define optimization algorithm configuration
	opt = Adam(lr=0.0002, beta_1=0.5)

	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae'], loss_weights=[1, 10, 10], optimizer=opt)
	return model

# load and prepare training images
def load_real_samples(filename):
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']

	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5

	# scale from [-32768,-32767] to [-1,1]
	X2 = (X2 / 32768)

	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# save the generator models to file
def save_models(step, g_model_ImageToAudio, g_model_AudioToImage):
	# save the first generator model
	filename1 = 'g_model_ImageToAudio_%06d.h5' % (step+1)
	g_model_ImageToAudio.save(filename1)
	# save the second generator model
	filename2 = 'g_model_AudioToImage_%06d.h5' % (step+1)
	g_model_AudioToImage.save(filename2)
	logging.info('>Saved: %s and %s' % (filename1, filename2))

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# select a sample of input audio
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_out = (X_out + 1) / 2.0

	for i in range(n_samples):
		# save audio inputs
		audio_data = (X_in[i] * 32768).astype(numpy.int16)
		wavfile.write('%s_input_%06d-%d.wav' % (name, (step+1), i), 44100, audio_data)

		# plot translated image
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	# save plot to file
	filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
	pyplot.savefig(filename1)
	pyplot.close()

# update pool for fake examples
def update_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing example and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected)


# train cyclegan models
def train(d_model_Image, d_model_Audio, g_model_ImageToAudio, g_model_AudioToImage, c_model_ImageToAudio, c_model_AudioToImage, dataset):
	# define properties of the training run
	n_epochs, n_batch, = 100, 1
	# determine the output square shape of the discriminator
	n_patch_Image = d_model_Image.output_shape[1]
	n_patch_Audio = d_model_Audio.output_shape[1]
	# unpack dataset
	trainImage, trainAudio = dataset
	# prepare image pool for fakes
	poolImage, poolAudio = list(), list()
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainImage) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		X_realImage, y_realImage = generate_real_samples(trainImage, n_batch, n_patch_Image)
		X_realAudio, y_realAudio = generate_real_samples(trainAudio, n_batch, n_patch_Audio)
		# generate a batch of fake samples
		X_fakeImage, y_fakeImage = generate_fake_samples(g_model_AudioToImage, X_realAudio, n_patch_Image)
		X_fakeAudio, y_fakeAudio = generate_fake_samples(g_model_ImageToAudio, X_realImage, n_patch_Audio)
		# update fakes from pool
		X_fakeImage = update_pool(poolImage, X_fakeImage)
		X_fakeAudio = update_pool(poolAudio, X_fakeAudio)
		# update generator Audio->Image via adversarial and cycle loss
		g_loss2, _, _, _ = c_model_AudioToImage.train_on_batch([X_realAudio, X_realImage], [y_realImage, X_realAudio, X_realImage])
		# update discriminator for Image -> [real/fake]
		dImage_loss1 = d_model_Image.train_on_batch(X_realImage, y_realImage)
		dImage_loss2 = d_model_Image.train_on_batch(X_fakeImage, y_fakeImage)
		# update generator Image->Audio via adversarial and cycle loss
		g_loss1, _, _, _ = c_model_ImageToAudio.train_on_batch([X_realImage, X_realAudio], [y_realAudio, X_realImage, X_realAudio])
		# update discriminator for Audio -> [real/fake]
		dAudio_loss1 = d_model_Audio.train_on_batch(X_realAudio, y_realAudio)
		dAudio_loss2 = d_model_Audio.train_on_batch(X_fakeAudio, y_fakeAudio)
		# summarize performance
		logging.info('>%d, dImage[%.3f,%.3f] dAudio[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dImage_loss1,dImage_loss2, dAudio_loss1,dAudio_loss2, g_loss1,g_loss2))
		# evaluate the model performance every so often
		if (i+1) % (bat_per_epo * 1) == 0:
			# plot Audio->Image translation
			summarize_performance(i, g_model_AudioToImage, trainAudio, 'AudioToImage')
		if (i+1) % (bat_per_epo * 5) == 0:
			# save the models
			save_models(i, g_model_ImageToAudio, g_model_AudioToImage)

def generate_one_fake_audio(file_prefix):
	g = define_audio_generator(image_shape)
	realImage, _ = generate_real_samples(images, 1, 16)
	# scale all pixels from [-1,1] to [0,1]
	realToOutput = (realImage + 1) / 2.0
	pyplot.imshow(realToOutput[0])
	pyplot.savefig('%s_real.png' % prefix)
	pyplot.close()
	logging.info('realImage', realImage.shape, realImage)
	fakeAudio = g.predict(realImage)[0]
	logging.info('fakeAudio', fakeAudio.shape, fakeAudio)
	fakeData = (fakeAudio * 32767).astype(numpy.int16)
	logging.info('fakeData', fakeData)
	wavfile.write('%s_fake.wav' % prefix, 44100, fakeData)

def generate_one_fake_image(file_prefix):
	g = define_image_generator(audio_shape)
	realAudio, _ = generate_real_samples(audio, 1, 16)
	logging.info('realAudio', realAudio.shape, realAudio)
	realData = (realAudio[0] * 128).astype(numpy.int16)
	logging.info('realData', realData)
	wavfile.write('%s_real.wav' % file_prefix, 44100, realData)
	fakeImage = g.predict(realAudio)[0]
	# scale all pixels from [-1,1] to [0,1]
	fakeToOutput = (fakeImage + 1) / 2.0
	pyplot.imshow(fakeToOutput[0])
	pyplot.savefig('%s_fake.png' % file_prefix)
	pyplot.close()

def main():
	# load data
	dataset = load_real_samples('stroke2notes.npz')
	images, audio = dataset
	image_shape = images.shape[1:]
	audio_shape = audio.shape[1:]
	logging.info('image_shape: %s', image_shape)
	logging.info('audio_shape: %s', audio_shape)

	g_model_ImageToAudio = define_audio_generator(image_shape)
	g_model_AudioToImage = define_image_generator(audio_shape)

	d_model_Image = define_image_discriminator(image_shape)
	d_model_Audio = define_audio_discriminator(audio_shape)

	# composite: Image -> Audio -> [real/fake, Image]
	c_model_ImageToAudio = define_composite_model(g_model_ImageToAudio, d_model_Audio, g_model_AudioToImage, image_shape, audio_shape)
	# composite: Audio -> Image -> [real/fake, Audio]
	c_model_AudioToImage = define_composite_model(g_model_AudioToImage, d_model_Image, g_model_ImageToAudio, audio_shape, image_shape)

	train(d_model_Image, d_model_Audio, g_model_ImageToAudio, g_model_AudioToImage, c_model_ImageToAudio, c_model_AudioToImage, dataset)

if __name__ == '__main__':
	main()
