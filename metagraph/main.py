import os
import pprint
import tensorflow as tf
import pickle as pkl

from nltk import word_tokenize
from data import *
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("LSTM_dim", 128, "output dimension of LSTM [128]")
flags.DEFINE_integer("lindim", 300, "linear part of the state [75]")
flags.DEFINE_integer("rules_dim", 7, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 3, "number of hops [7]")
flags.DEFINE_integer("batch_size", 10, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 20, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.05, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.01, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [50]")
flags.DEFINE_string("pretrain_file", "/media/storage/BTP/my_work/emb/glove.42B.300d.txt", "pre-trained glove vectors file path [../data/glove.6B.300d.txt]")
# flags.DEFINE_string("train_data", "../data/Laptops_Train.xml.seg", "train gold data set path [./data/Laptops_Train.xml.seg]")
flags.DEFINE_string("train_data", "../data/Restaurants_Train_v2.xml.seg", "train gold data set path [./data/Laptops_Train.xml.seg]")
# flags.DEFINE_string("test_data", "../data/Laptops_Test_Gold.xml.seg", "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_string("test_data", "../data/Restaurants_Test_Gold.xml.seg", "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def main(_):
  source_count, target_count = [], []
  source_word2idx, target_word2idx, word_set = {}, {}, {}
  max_sent_len = -1
  
  max_sent_len = get_dataset_resources(FLAGS.train_data, source_word2idx, target_word2idx, word_set, max_sent_len)
  max_sent_len = get_dataset_resources(FLAGS.test_data, source_word2idx, target_word2idx, word_set, max_sent_len)
  # embeddings = load_embedding_file(FLAGS.pretrain_file, word_set)
  # print "Embeddings Loaded"

  # train_data = get_dataset(FLAGS.train_data, source_word2idx, target_word2idx, embeddings)
  # test_data = get_dataset(FLAGS.test_data, source_word2idx, target_word2idx, embeddings)
  
  # pkl.dump(train_data, open('train_data_restaurant.pkl', 'w'))
  # pkl.dump(test_data, open('test_data_restaurant.pkl', 'w'))
  # print "Dump Success!!!"
  # return

  # train_data = pkl.load(open('train_data_laptop.pkl', 'r'))
  # train_data = pkl.load(open('train_data_extra.pkl', 'r'))
  train_data = pkl.load(open('train_data_restaurant_clean.pkl', 'r'))

  # test_data = pkl.load(open('test_data_laptop.pkl', 'r'))
  # test_data = pkl.load(open('test_data_extra.pkl', 'r'))
  test_data = pkl.load(open('test_data_restaurant_clean.pkl', 'r'))

  # GraphMemNetData = pkl.load(open('TOTAL_LAT_const_laptop.pkl','r'))
  GraphMemNetData = pkl.load(open('TOTAL_data_restaurant_clean.pkl','r'))
  print "Dump Loaded!!!"

  Wma_train = GraphMemNetData[0][6]
  Wma_test = GraphMemNetData[1][6]
  
  Wrm = train_data[5]
  for index, wma in enumerate(Wma_train):
    wam = np.reshape(wma,(1,-1))
    Wrm[index] = np.concatenate((Wrm[index], wam),axis=0)
  
  Wrm = test_data[5]
  for index, wma in enumerate(Wma_test):
    wam = np.reshape(wma,(1,-1))
    Wrm[index] = np.concatenate((Wrm[index], wam),axis=0)

  

  print "train data size - ", len(train_data[0])
  print "test data size - ", len(test_data[0])

  print "max sentence length - ",max_sent_len
  FLAGS.pad_idx = source_word2idx['<pad>']
  FLAGS.nwords = len(source_word2idx)
  FLAGS.mem_size = max_sent_len

  pp.pprint(flags.FLAGS.__flags)

  print('loading pre-trained word vectors...')
  print('loading pre-trained word vectors for train and test data')
  
  # FLAGS.pre_trained_context_wt, FLAGS.pre_trained_target_wt = get_embedding_matrix(embeddings, source_word2idx,  target_word2idx, FLAGS.edim)
  FLAGS.pre_trained_context_wt, FLAGS.pre_trained_target_wt = GraphMemNetData[2], GraphMemNetData[3]
  
  with tf.Session() as sess:
    model = MemN2N(FLAGS, sess)
    model.build_model()
    model.run(train_data, test_data)  

if __name__ == '__main__':
  tf.app.run()  