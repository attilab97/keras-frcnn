from keras import backend as K
import math

class Config:

	def __init__(self):

		self.verbose = True

		self.network = 'resnet50'

		# setari pentru augmentarea datelor
		self.use_horizontal_flips = False
		self.use_vertical_flips = False
		self.rot_90 = False

		# dimensiunile ancorelor
		self.anchor_box_scales = [128, 256, 512]

		# proportiile ancorelor
		self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

		# cea mai mica latura a imaginii
		self.im_size = 1024

		# image channel-wise mean to subtract
		self.img_channel_mean = [103.939, 116.779, 123.68]
		self.img_scaling_factor = 1.0

		# numarul pe regiuni de interes scanate de-o data
		self.num_rois = 4

		# stride-ul RPN-ului
		self.rpn_stride = 16

		self.balanced_classes = False

		# scalarea stdev-ului
		self.std_scaling = 4.0
		self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

		# limitele suprapunerilor pentru RPN
		self.rpn_min_overlap = 0.3
		self.rpn_max_overlap = 0.7

		# limitele suprapunerilor pentru regiunile de interes ale R-CNN-ului
		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5

		self.class_mapping = None

		#locatia ponderilor preantrenate pentru reteaua de baza pentru tensorflow si theano
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

		self.model_path = 'model_frcnn.vgg.hdf5'
