import os
import pprint
import tensorflow as tf
import pickle as pkl
import copy

from nltk import word_tokenize
from data import *
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("LSTM_dim", 128, "output dimension of LSTM [128]")
flags.DEFINE_integer("lindim", 300, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 2, "number of hops [7]")
flags.DEFINE_integer("batch_size", 1 , "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 30, "number of epoch to use during training [100]")
flags.DEFINE_integer("pad_idx", 30, "pad_id")
flags.DEFINE_integer("nwords", 30, "nwords")
flags.DEFINE_integer("mem_size", 30, "mem_size")

flags.DEFINE_float("init_lr", 0.05, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.01, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [50]")
flags.DEFINE_string("pretrain_file", "/media/storage/BTP/my_work/emb/glove.42B.300d.txt", "pre-trained glove vectors file path [../data/glove.6B.300d.txt]")
# flags.DEFINE_string("train_data", "data/Laptops_Train.xml.seg", "train gold data set path [./data/Laptops_Train.xml.seg]")
# flags.DEFINE_string("test_data", "data/Laptops_Test_Gold.xml.seg", "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_string("train_data", "data/Restaurants_Train_v2.xml.seg", "train gold data set path [./data/Laptops_Train.xml.seg]")
flags.DEFINE_string("test_data", "data/Restaurants_Test_Gold.xml.seg", "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def main(_):
	source_count, target_count = [], []
	source_word2idx, target_word2idx, word_set = {}, {}, {}
	max_sent_len = -1
	
	max_sent_len = get_dataset_resources(FLAGS.train_data, source_word2idx, target_word2idx, word_set, max_sent_len)
	max_sent_len = get_dataset_resources(FLAGS.test_data, source_word2idx, target_word2idx, word_set, max_sent_len)
	embeddings = load_embedding_file(FLAGS.pretrain_file, word_set)
	print "Embeddings Loaded"
	
	# train_data = get_dataset(FLAGS.train_data, source_word2idx, target_word2idx, embeddings)
	# # pkl.dump(train_data, open('train_data_laptop_init.pkl', 'w'))
	# pkl.dump(train_data, open('train_data_restaurant_init.pkl', 'w'))
	# print "train data done"
	# test_data = get_dataset(FLAGS.test_data, source_word2idx, target_word2idx, embeddings)
	# # pkl.dump(test_data, open('test_data_laptop_init.pkl', 'w'))
	# pkl.dump(test_data, open('test_data_restaurant_init.pkl', 'w'))
	# print "Dump Success!!!"
	# return
	
	
	# train_data = pkl.load(open('train_data_laptop_init.pkl', 'r'))
	# test_data = pkl.load(open('test_data_laptop_init.pkl', 'r'))
	train_data = pkl.load(open('train_data_restaurant_init.pkl', 'r'))
	test_data = pkl.load(open('test_data_restaurant_init.pkl', 'r'))

	print "Getting pretrained weights"
	pre_trained_context_wt, pre_trained_target_wt = get_embedding_matrix(embeddings, source_word2idx,	target_word2idx, FLAGS.edim)
	# pkl.dump((train_data, test_data, pre_trained_context_wt, pre_trained_target_wt, max_sent_len),open('TOTAL_laptop_init.pkl','w'))
	pkl.dump((train_data, test_data, pre_trained_context_wt, pre_trained_target_wt, max_sent_len),open('TOTAL_restaurant_init.pkl','w'))
	print "dump dumped"
	return
	
	# train_data, test_data, pre_trained_context_wt, pre_trained_target_wt, max_sent_len = pkl.load(open('TOTAL_laptop_init.pkl','r'))
	# train_data, test_data, pre_trained_context_wt, pre_trained_target_wt, max_sent_len = pkl.load(open('TOTAL_LAT_const_laptop.pkl','r'))
	train_data, test_data, pre_trained_context_wt, pre_trained_target_wt, max_sent_len = pkl.load(open('TOTAL_data_restaurant_clean.pkl','r'))

	#return source_word2idx, target_word2idx, train_data, test_data

	'''
	#splitting the dataset into with and without rules
	'''
	rul_test_data = pkl.load(open('./metagraph/test_data_restaurant_clean.pkl', 'r'))
	# rul_test_data = pkl.load(open('./metagraph/test_data_laptop.pkl', 'r'))
	wl = rul_test_data[5]
	#Removing the test points with all zero's
	test_data1 = copy.deepcopy(test_data)
	test_data2 = copy.deepcopy(test_data)
	
	print len(wl)
	indices = []
	#with rules
	for i in range(len(wl)):
		if not np.any(wl[i][:-1]):
			indices.append(i)
	for entry in test_data1:
		for index in sorted(indices, reverse=True):
			del entry[index]

	indices = []
	#without rules
	for i in range(len(wl)):
		if np.any(wl[i][:-1]):
			indices.append(i)
	for entry in test_data2:
		for index in sorted(indices, reverse=True):
			del entry[index]

	'''
	'''

	print "train data size - ", len(train_data[0])
	print "test data size - ", len(test_data[0]), len(test_data1[0]), len(test_data2[0])
	# return
	print "max sentence length - ",max_sent_len
	FLAGS.pad_idx = source_word2idx['<pad>']
	FLAGS.nwords = len(source_word2idx)
	FLAGS.mem_size = max_sent_len

	pp.pprint(flags.FLAGS.__flags)

	print('loading pre-trained word vectors...')
	print('loading pre-trained word vectors for train and test data')
	
	# FLAGS.pre_trained_context_wt, FLAGS.pre_trained_target_wt = get_embedding_matrix(embeddings, source_word2idx,	target_word2idx, FLAGS.edim)

	# print FLAGS.pre_trained_context_wt.shape, FLAGS.pre_trained_target_wt.shape
	with tf.Session() as sess:
		model = MemN2N(FLAGS, sess, pre_trained_context_wt, pre_trained_target_wt)
		model.build_model()
		model.run(train_data, test_data, test_data1, test_data2)	

if __name__ == '__main__':
	tf.app.run()	