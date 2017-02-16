################################################################
#	Original code: data/cifar10.py in lemontree by Kyuhong Shim
#	Modified for cifar100 -	Min-J
################################################################
#	Only works for the "fine-labels"
################################################################


import numpy as np
import scipy.io
from lemontree.utils.data_utils import split_data, int_to_onehot
from lemontree.utils.type_utils import uint_to_float, uint_to_int
from lemontree.data.dataset import BaseDataset


class CIFAR100(BaseDataset):
	"""
	This class load cifar-100 dataset to a tensor form.
	Flat mode returns 2D matrix, and tensor mode returns 4D matrix.
	"""
	def __init__(self, base_datapath, mode='tensor', seed=999, one_hot=False):
		"""
		This function initializes the class.
		CIFAR-100 is not a big dataset, so we can hold all floating values in memory.

		Parameters
		----------
		base_datapath: string
			a string path where mnist is saved.
		mode: string, default: 'tensor'
			a string either {'flat', 'tensor'}.
		seed: integer, default: 999
			a random seed to shuffle the data.
		one_hot: bool, default: False
			a bool value that choose wheter the label will be one-hot encoded or just integer.

		Returns
		-------
		None.
		"""
		super(CIFAR100, self).__init__(base_datapath, seed)
		# check asserts
		assert mode in ['flat', 'tensor'], '"mode" should be a string either "flat" or "tensor".'

		# valid not yet divided
		self.valid_exist = False  # flag off

		# load
		cifar_train = scipy.io.loadmat(self.base_datapath + 'cifar100/train.mat')
		cifar_test = scipy.io.loadmat(self.base_datapath + 'cifar100/test.mat')
		train_data = uint_to_float(cifar_train['data']) / 255.0
		test_data = uint_to_float(cifar_test['data']) / 255.0
		train_label = uint_to_int(cifar_train['fine_labels'])
		test_label = uint_to_int(cifar_test['fine_labels'])

		train_label = np.squeeze(train_label)
		test_label = np.squeeze(test_label)

		train_order = self.rng.permutation(train_label.shape[0])
		test_order = self.rng.permutation(test_label.shape[0])

		self.train_data = train_data[train_order]
		self.test_data = test_data[test_order]
		self.train_label = train_label[train_order]
		self.test_label = test_label[test_order]

		if mode == 'tensor':
			self.train_data = np.reshape(self.train_data, (self.train_data.shape[0], 3, 32, 32))
			self.test_data = np.reshape(self.test_data, (self.test_data.shape[0], 3, 32, 32))
		elif mode == 'flat':
			pass
		else:
			raise NotImplementedError('No such mode exist')

		if one_hot:
			self.train_label = int_to_onehot(self.train_label, 100)
			self.test_label = int_to_onehot(self.test_label, 100)


def label_to_real(label):
	if label == 0:
		return 'apple'
	elif label == 1:
		return 'aquarium_fish'
	elif label == 2:
		return 'baby'
	elif label == 3:
		return 'bear'
	elif label == 4:
		return 'beaver'
	elif label == 5:
		return 'bed'
	elif label == 6:
		return 'bee'
	elif label == 7:
		return 'beetle'
	elif label == 8:
		return 'bicycle'
	elif label == 9:
		return 'bottle'
	elif label == 10:
		return 'bowl'
	elif label == 11:
		return 'boy'
	elif label == 12:
		return 'bridge'
	elif label == 13:
		return 'bus'
	elif label == 14:
		return 'butterfly'
	elif label == 15:
		return 'camel'
	elif label == 16:
		return 'can'
	elif label == 17:
		return 'castle'
	elif label == 18:
		return 'caterpillar'
	elif label == 19:
		return 'cattle'
	elif label == 20:
		return 'chair'
	elif label == 21:
		return 'chimpanzee'
	elif label == 22:
		return 'clock'
	elif label == 23:
		return 'cloud'
	elif label == 24:
		return 'cockroach'
	elif label == 25:
		return 'couch'
	elif label == 26:
		return 'crab'
	elif label == 27:
		return 'crocodile'
	elif label == 28:
		return 'cup'
	elif label == 29:
		return 'dinosaur'
	elif label == 30:
		return 'dolphin'
	elif label == 31:
		return 'elephant'
	elif label == 32:
		return 'flatfish'
	elif label == 33:
		return 'forest'
	elif label == 34:
		return 'fox'
	elif label == 35:
		return 'girl'
	elif label == 36:
		return 'hamster'
	elif label == 37:
		return 'house'
	elif label == 38:
		return 'kangaroo'
	elif label == 39:
		return 'keyboard'
	elif label == 40:
		return 'lamp'
	elif label == 41:
		return 'lawn_mower'
	elif label == 42:
		return 'leopard'
	elif label == 43:
		return 'lion'
	elif label == 44:
		return 'lizard'
	elif label == 45:
		return 'lobster'
	elif label == 46:
		return 'man'
	elif label == 47:
		return 'maple_tree'
	elif label == 48:
		return 'motorcycle'
	elif label == 49:
		return 'mountain'
	elif label == 50:
		return 'mouse'
	elif label == 51:
		return 'mushroom'
	elif label == 52:
		return 'oak_tree'
	elif label == 53:
		return 'orange'
	elif label == 54:
		return 'orchid'
	elif label == 55:
		return 'otter'
	elif label == 56:
		return 'palm_tree'
	elif label == 57:
		return 'pear'
	elif label == 58:
		return 'pickup_truck'
	elif label == 59:
		return 'pine_tree'
	elif label == 60:
		return 'plain'
	elif label == 61:
		return 'plate'
	elif label == 62:
		return 'poppy'
	elif label == 63:
		return 'porcupine'
	elif label == 64:
		return 'possum'
	elif label == 65:
		return 'rabbit'
	elif label == 66:
		return 'raccoon'
	elif label == 67:
		return 'ray'
	elif label == 68:
		return 'road'
	elif label == 69:
		return 'rocket'
	elif label == 70:
		return 'rose'
	elif label == 71:
		return 'sea'
	elif label == 72:
		return 'seal'
	elif label == 73:
		return 'shark'
	elif label == 74:
		return 'shrew'
	elif label == 75:
		return 'skunk'
	elif label == 76:
		return 'skyscraper'
	elif label == 77:
		return 'snail'
	elif label == 78:
		return 'snake'
	elif label == 79:
		return 'spider'
	elif label == 80:
		return 'squirrel'
	elif label == 81:
		return 'streetcar'
	elif label == 82:
		return 'sunflower'
	elif label == 83:
		return 'sweet_pepper'
	elif label == 84:
		return 'table'
	elif label == 85:
		return 'tank'
	elif label == 86:
		return 'telephone'
	elif label == 87:
		return 'television'
	elif label == 88:
		return 'tiger'
	elif label == 89:
		return 'tractor'
	elif label == 90:
		return 'train'
	elif label == 91:
		return 'trout'
	elif label == 92:
		return 'tulip'
	elif label == 93:
		return 'turtle'
	elif label == 94:
		return 'wardrobe'
	elif label == 95:
		return 'whale'
	elif label == 96:
		return 'willow_tree'
	elif label == 97:
		return 'wolf'
	elif label == 98:
		return 'woman'
	elif label == 99:
		return 'worm'
	else:
		raise NotImplementedError('No such label exist')
