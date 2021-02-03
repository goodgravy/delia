# example of preparing the horses and zebra dataset
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from scipy.io import wavfile

# load all images in a directory into memory
def load_images(path, size=(208,208)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)

# load all sound clips in a directory into memory
def load_audio(path, sample_count=208*208):
	data_list = list()
	# enumerate filenames in directory, assume all are audio
	for filename in listdir(path):
		# load and resize the image
		samples = asarray(wavfile.read(path + filename)[1])
		data_list.append(samples[:sample_count])
	return asarray(data_list)

# load dataset A
dataA = load_images('Strokes/')
print('Loaded dataA: ', dataA.shape)

# load dataset B
dataB = load_audio('Notes/')
print('Loaded dataB: ', dataB.shape)

# save as compressed numpy array
filename = 'stroke2notes.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)
