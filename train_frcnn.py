from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import logging

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import roi_helpers as roi_helpers
from keras_frcnn import resnet as nn
from keras_frcnn.simple_parser import get_data
#setam nivelul de debug

logging.basicConfig(filename = '/content/drive/My Drive/Lara_log/logs.log', level = logging.INFO)
logger = logging.getLogger('keras_frcnn.train_frcnn')
sys.setrecursionlimit(40000)

parser = OptionParser()
parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--epoch_length", type="int", dest="epoch_length", help="Number of steps per epoch.", default=1000)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).",default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_option("--log_path", dest="log_path", help="Path of the logs.", default='./logs')

(options, args) = parser.parse_args()

# daca nu e data calea catre datele de antrenare
if not options.train_path:  
	parser.error('Error: path to training data must be specified. Pass --path to command line')
logging.basicConfig(filename = options.log_path + 'logs.log', level = logging.INFO)
logger = logging.getLogger('keras_frcnn.train_frcnn')

# trimitem argumentele din linia de comanda mai departe pentru procesare
C = config.Config()

#augmentarea datelor
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)

# verificam daca ponderile preantrenate au fost trimise prin parametru
if options.input_weight_path:
	C.base_net_weights = options.input_weight_path
else:
	# daca nu, setam calea catre ponderile retelei de baza
	C.base_net_weights = nn.get_weight_path()

# parser-ul
all_imgs, classes_count, class_mapping = get_data(options.train_path)

# bg
if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.image_dim_ordering(), mode='val')

# verifican campul img_dim_ordering pentru backkend-ul de keras
if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

# placeholder-ul pentru input
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# definim reteaua de baza, resnet50
shared_layers = nn.nn_base(img_input, trainable=True)

# definim RPN-ul contruit pe straturile retelei de baza
# RPN 
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

# reteaua de detectie, R-CNN
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# modelul care contine RPN-ul si R-CNN-ul, folosit pentru a incarca si salva ponderile modelului
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
	print('loading weights from {}'.format(C.base_net_weights))
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

#optimizatorul pentru RPN
optimizer = Adam(lr=1e-5)

#optimizatorul folosit pentru R-CNN (pentru clasificare)
optimizer_classifier = Adam(lr=1e-5)

#compilam RPN-ul si R-CNN-ul
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

#mae = mean absolute error
model_all.compile(optimizer='sgd', loss='mae', metrics=['accuracy'])

epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):
	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
	logging.info('Epochs {}/{}'.format(epoch_num + 1, num_epochs))
	while True:
		try:

			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				logging.info('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					logging.info('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			X, Y, img_data = next(data_gen_train)

			loss_rpn = model_rpn.train_on_batch(X, Y)

			P_rpn = model_rpn.predict_on_batch(X)

			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
			# calc_iou converteste din coordonate (x1,y1,x2,y2) in format (x,y,w,h)
			X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue
	
       			 # esantionam valori pozitive si negative
			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []
			
			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))

			if C.num_rois > 1:
				if len(pos_samples) < C.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
				try:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

				sel_samples = selected_pos_samples + selected_neg_samples
			else:
				# in cazuri extreme cand avem o regiune de interes luam valori aleatoare pozitive sau negative
				selected_pos_samples = pos_samples.tolist()
				selected_neg_samples = neg_samples.tolist()
				if np.random.randint(0, 2):
					sel_samples = random.choice(neg_samples)
				else:
					sel_samples = random.choice(pos_samples)

			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			losses[iter_num, 2] = loss_class[1]
			losses[iter_num, 3] = loss_class[2]
			losses[iter_num, 4] = loss_class[3]

			progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
									  ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

			iter_num += 1
			
			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					logging.info('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					logging.info('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					logging.info('Loss RPN classifier: {}'.format(loss_rpn_cls))
					logging.info('Loss RPN regression: {}'.format(loss_rpn_regr))
					logging.info('Loss Detector classifier: {}'.format(loss_class_cls))
					logging.info('Loss Detector regression: {}'.format(loss_class_regr))
					logging.info('Elapsed time: {}'.format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					if C.verbose:
						logging.info('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(C.model_path)

				break

		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print('Training complete, exiting.')
