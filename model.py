import os
import sys
import math
import random
import numpy as np
import tensorflow as tf
from past.builtins import xrange
import time as tim

class MemN2N(object):
    def __init__(self, config, sess, pre_trained_context_wt, pre_trained_target_wt):
        ''' Initialisation of Parameters'''
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.LSTM_dim = config.LSTM_dim
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm
        self.pad_idx = config.pad_idx
        self.pre_trained_context_wt = pre_trained_context_wt
        self.pre_trained_target_wt = pre_trained_target_wt
        print self.pre_trained_context_wt.shape
        self.input = tf.placeholder(tf.int32, [self.batch_size, 1], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.int64, [self.batch_size], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.mem_size], name="mask")
        self.A = tf.placeholder(tf.float32, [self.nwords, self.edim], name="A") # Vocab * edim
        self.ASP = tf.placeholder(tf.float32, [self.pre_trained_target_wt.shape[0], self.edim], name="ASP") # V2 * edim
        self.LSTM_inp_dout = tf.placeholder(tf.float32, name="LSTM_inp_dout")
        self.LSTM_out_dout = tf.placeholder(tf.float32, name="LSTM_out_dout")
        self.Final_dout = tf.placeholder(tf.float32, name="Final_dout")

        self.neg_inf = tf.fill([self.batch_size, self.mem_size], -1*np.inf, name="neg_inf")

        self.delta_inv = tf.placeholder(tf.float32, [self.batch_size, self.mem_size, self.mem_size], name="delta_inv")
        self.W_ma      = tf.placeholder(tf.float32, [self.batch_size, self.mem_size], name="W_ma")

        self.show = config.show

        self.hid = []

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
      ''' Building the model '''
      self.global_step = tf.Variable(0, name="global_step")

      self.C = tf.Variable(tf.random_uniform([self.LSTM_dim, self.LSTM_dim], minval=-0.01, maxval=0.01)) #LSTM_dim * LSTM_dim
      self.C_B =tf.Variable(tf.random_uniform([1, self.edim], minval=-0.01, maxval=0.01))
      # self.BL_W = tf.Variable(tf.random_uniform([2 * self.LSTM_dim, 1], minval=-0.01, maxval=0.01))
      # self.BL_B = tf.Variable(tf.random_uniform([1, 1], minval=-0.01, maxval=0.01))
      self.C0 = tf.Variable(tf.random_uniform([self.edim, self.LSTM_dim], minval=-0.01, maxval=0.01)) #edim * LSTM_dim
      self.GW = tf.Variable(tf.random_uniform([self.LSTM_dim, self.LSTM_dim], minval=-0.01, maxval=0.01)) #edim * LSTM_dim


      self.Ain_c = tf.nn.embedding_lookup(self.A, self.context) #batch_size * mem_size * edim
      # # self.Ain = self.Ain_c


      # Embedding Look Up for Aspect Word, self.hid is a list that would contain the output for every layer 
      # memory network and currently it contains the input to the first layer of memory network
      self.ASPin = tf.nn.embedding_lookup(self.ASP, self.input) #batch_size * 1 * edim
      self.ASPout2dim = tf.reshape(self.ASPin, [-1, self.edim]) #batch_size * edim
      self.TransfASPout2dim = tf.matmul(self.ASPout2dim, self.C0) #batch_size * LSTM_dim
      self.hid.append(self.TransfASPout2dim)    #batch_size * LSTM_dim


      # LSTM for converting the words of the sentence from embedding space to input space (300 -> 128)
      # outputs contain the output of LSTM at every timestep
      self.LSTM_input = self.Ain_c #(batch_size , mem_size, e_dim)
      cell = tf.nn.rnn_cell.LSTMCell(self.LSTM_dim, state_is_tuple=True)
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.LSTM_inp_dout,  state_keep_prob = self.LSTM_out_dout)
      outputs, state = tf.nn.dynamic_rnn(cell, \
                                        self.LSTM_input, \
                                        sequence_length=[self.mem_size]*self.batch_size, \
                                        dtype=tf.float32)


      lstm_out = outputs
      self.Ain = outputs #batch_size * mem_size * lstm_dim

      # Calculation of Graph Based Attention
      # Og is the output of Graph Attention

      self.W_ma3dim = tf.reshape(self.W_ma, [self.batch_size, self.mem_size, -1]) #(batch_size * m * 1)
      self.Z  = tf.matmul(self.delta_inv, self.W_ma3dim) #(batch_size * m * 1)
      # uncomment the line 102 if you want to just use the Wma for attention and not the Laplacian
      self.Z  = self.W_ma3dim 
      self.Z2dim = tf.reshape(self.Z, [self.batch_size, -1]) #(batch_size * m )
      self.AddedZ = tf.multiply(self.Z2dim, self.mask)
      self.masked_Z = tf.reshape(self.AddedZ, [self.batch_size, self.mem_size, 1]) #(batch_size * m * 1)

      self.Mm = tf.transpose(lstm_out, perm=[0, 2, 1]) #batch_size * lstm_dim * m

      self.Og = tf.matmul(self.Mm, self.Z)   #(batch_size * lstm_dim * 1)
      self.Og = tf.matmul(self.Mm, self.masked_Z)   #(batch_size * lstm_dim * 1)
      self.Og2dim = tf.reshape(self.Og, [self.batch_size,-1]) #(batch_size * lstm_dim)
      

      for h in xrange(self.nhop):
        '''
        Bi-linear scoring function for a context word and aspect term
        '''
        # Calculation of Semantic Attention for a layer in memory network
        # This is in a for loop which gets repeated as many times as the number of layers in the network

        self.U3dim = tf.reshape(self.hid[-1], [-1, self.LSTM_dim, 1]) #bs * lstm_dim * 1
        self.att3dim = tf.matmul(self.Ain, self.U3dim) #batch_size * mem_size * 1
        self.att2dim = tf.reshape(self.att3dim, [-1, self.mem_size]) #batch_size * mem_size
        self.g_2dim = tf.nn.tanh(self.att2dim) #batch_size * mem_size

        self.masked_g_2dim = tf.multiply(self.g_2dim, self.mask) #batch_size  *  mem_size
        self.P = self.masked_g_2dim #batch_size  *  mem_size
        self.probs3dim = tf.reshape(self.P, [-1, 1, self.mem_size]) #batch_size * 1  *  mem_size

        self.Aout = tf.matmul(self.probs3dim, self.Ain) #batch_size * 1 * lstm_dim
        self.Aout2dim = tf.reshape(self.Aout, [self.batch_size, self.LSTM_dim]) #batch_size * lstm_dim
        
        self.Fout2dim = tf.add(self.Og2dim, tf.matmul(self.Aout2dim, self.GW)) #batch_size * lstm_dim

        # Uncomment the following lines for using only Graph Based, Only Semantic Based or other combinations
        # self.Fout2dim = self.Og2dim #batch_size * lstm_dim
        # self.Fout2dim = self.Aout2dim #batch_size * lstm_dim

        #Comment the line 140 if you are not using semantic attention
        Cout = tf.matmul(self.hid[-1], self.C) #batch_size * lstm_dim
  
        #Comment the line 143 and uncomment 144 if you are not using semantic attention
        self.Dout = tf.add(Cout, self.Fout2dim) #batch_size * lstm_dim
        # self.Dout = self.Fout2dim

        # Ignore this no need to look at it 
        if self.lindim == self.edim:
            self.hid.append(self.Dout)
        elif self.lindim == 0:
            self.hid.append(tf.nn.relu(self.Dout))
        else:
            F = tf.slice(self.Dout, [0, 0], [self.batch_size, self.lindim])
            G = tf.slice(self.Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
            K = tf.nn.relu(G)
            self.hid.append(tf.concat(axis=1, values=[F, K]))

    def build_model(self): 
      ''' Function to predict the polarity of the sentiment from the output of memory network '''
      self.build_memory()

      self.W = tf.Variable(tf.random_uniform([self.LSTM_dim, 3], minval=-0.01, maxval=0.01))

      self.dropped_out = tf.nn.dropout(self.hid[-1], self.Final_dout) 
      
      self.z = tf.matmul(self.dropped_out, self.W)
      
      self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.z, labels=self.target)

      self.lr = tf.Variable(self.current_lr)
      self.opt = tf.train.AdagradOptimizer(self.lr)

      params = None
      self.loss = tf.reduce_mean(self.loss) 

      grads_and_vars = self.opt.compute_gradients(self.loss, params)
      # Uncomment the following lines if you want to apply gradient clipping
      # clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
      #                           for gv in grads_and_vars]
      clipped_grads_and_vars = grads_and_vars

      inc = self.global_step.assign_add(1)
      with tf.control_dependencies([inc]):
          self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

      tf.initialize_all_variables().run()

      self.correct_prediction = tf.argmax(self.z, 1)

    def train(self, data):
      ''' Function to train the model on the provided data '''

      source_data, source_loc_data, target_data, target_label, orig_sent_data, delta_inv_data, W_ma_data= data
      N = int(math.ceil(len(source_data) / self.batch_size))
      cost = 0

      x = np.ndarray([self.batch_size, 1], dtype=np.int32)
      time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      target = np.zeros([self.batch_size], dtype=np.int32) 
      context = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      mask = np.ndarray([self.batch_size, self.mem_size])
      delta_inv = np.ndarray([self.batch_size, self.mem_size, self.mem_size], dtype=np.float32)
      W_ma = np.ndarray([self.batch_size, self.mem_size], dtype=np.float32)

      if self.show:
        from utils import ProgressBar
        bar = ProgressBar('Train', max=N)

      rand_idx, cur = np.random.permutation(len(source_data)), 0
      for idx in xrange(N):
        if self.show: bar.next()
        
        context.fill(self.pad_idx)
        time.fill(self.mem_size)
        target.fill(0)
        mask.fill(0)
        

        for b in xrange(self.batch_size):
            m = rand_idx[cur]
            x[b][0] = target_data[m]
            target[b] = target_label[m]
            time[b,:len(source_loc_data[m])] = source_loc_data[m]
            context[b,:len(source_data[m])] = source_data[m]
            mask[b,:len(source_data[m])].fill(1)

            crt_delta = delta_inv_data[m]
            delta_inv[b] = np.pad(crt_delta, [(0,self.mem_size - len(crt_delta[0]))]*2, 'constant', constant_values = 0)
            crt_wma = W_ma_data[m]
            crt_wma = crt_wma.reshape(crt_wma.shape[0])
            W_ma[b] = np.pad(crt_wma, [(0,self.mem_size - len(crt_wma))], 'constant', constant_values = 0)
            cur = cur + 1

        dinv, dout, do, kout,kinc,kin, aspin, C0, z, a, loss, self.step = self.sess.run([  self.delta_inv, self.Dout, self.dropped_out, self.A, self.Ain_c, self.Ain, self.ASPin, self.C0 , self.z,
                                            self.optim,
                                            self.loss,
                                            self.global_step],
                                            feed_dict={
                                                self.input: x,
                                                self.time: time,
                                                self.target: target,
                                                self.context: context,
                                                self.mask: mask,
                                                self.delta_inv: delta_inv,
                                                self.W_ma: W_ma,
                                                self.A:self.pre_trained_context_wt,
                                                self.ASP:self.pre_trained_target_wt,
                                                self.LSTM_inp_dout:0.5,
                                                self.LSTM_out_dout:0.7,
                                                self.Final_dout:0.7
                                                })
        
       
        if idx == 0:
          pass
            # print idx
            # print "asp - ", asp[0]
            # print "tasp - ", tasp[0]
            # print "A - ", kout
            # print "Ainc - ", kinc[0][:2][:20]
            # print "Ain - ", kin[0][:2][:20]
            # print "maskedZ - ", addedZ[0]
            # print "maskedZ - ", maskedZ[0]
            # print "dinv - ", dinv[:2][:20]
            # print "wma - ", wma[:2][:20]
            # print "Og - ", Ogg[:2][:20]
            # print "Z - ", Z[0]
            # print "ASPin - ", aspin
            # print "C0 - ", C0
            # print "U3dim - ", att[:2][:20]
            # #print "loss - ", loss
            # print "mask - ", mask[:2][:20]
            # print "Semantic Attention - ", P[:2][:20]
            # print "small z - " , z
            # print "dout - ", dout
            # print "dropped_out - ", do
        cost += np.sum(loss)
      
      if self.show: bar.finish()
      _, train_acc = self.test(data)
      return cost/N/self.batch_size, train_acc

    def test(self, data):
      ''' function for testing the trained model on provided data '''

      source_data, source_loc_data, target_data, target_label, orig_sent_data, delta_inv_data, W_ma_data = data
      N = int(math.ceil(len(source_data) / self.batch_size))
      cost = 0

      x = np.ndarray([self.batch_size, 1], dtype=np.int32)
      time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      target = np.zeros([self.batch_size], dtype=np.int32) 
      context = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      mask = np.ndarray([self.batch_size, self.mem_size])
      delta_inv = np.ndarray([self.batch_size, self.mem_size, self.mem_size], dtype=np.float32)
      W_ma = np.ndarray([self.batch_size, self.mem_size], dtype=np.float32)


      context.fill(self.pad_idx)

      m, acc = 0, 0
      for i in xrange(N):
        target.fill(0)
        time.fill(self.mem_size)
        context.fill(self.pad_idx)
        mask.fill(0)
        
        raw_labels = []
        for b in xrange(self.batch_size):
          x[b][0] = target_data[m]
          target[b] = target_label[m]
          time[b,:len(source_loc_data[m])] = source_loc_data[m]
          context[b,:len(source_data[m])] = source_data[m]
          mask[b,:len(source_data[m])].fill(1)
          raw_labels.append(target_label[m])

          crt_delta = delta_inv_data[m]
          delta_inv[b] = np.pad(crt_delta, [(0,self.mem_size - len(crt_delta[0]))]*2, 'constant', constant_values = 0)
          crt_wma = W_ma_data[m]
          crt_wma = crt_wma.reshape(crt_wma.shape[0])
          W_ma[b] = np.pad(crt_wma, [(0,self.mem_size - len(crt_wma))], 'constant', constant_values = 0)

          m += 1

        loss, predictions = self.sess.run([self.loss, self.correct_prediction],
                                        feed_dict={
                                            self.input: x,
                                            self.time: time,
                                            self.target: target,
                                            self.context: context,
                                            self.mask: mask,
                                            self.delta_inv: delta_inv,
                                            self.W_ma: W_ma,
                                            self.A:self.pre_trained_context_wt,
                                            self.ASP:self.pre_trained_target_wt,
                                            self.LSTM_inp_dout:1.0,
                                            self.LSTM_out_dout:1.0,
                                            self.Final_dout:1.0})
        cost += np.sum(loss)

        for b in xrange(self.batch_size):
          if raw_labels[b] == predictions[b]:
            acc = acc + 1

      print 'predictions - ', predictions
      print 'labels - ', raw_labels
      return cost, acc/float(len(source_data))

    # def run(self, train_data, test_data):
    #   ''' function for training and testing the trained model '''
    #   print('training...')

    #   for idx in xrange(self.nepoch):
    #     print('epoch '+str(idx)+'...')
    #     train_loss, train_acc = self.train(train_data)
    #     test_loss, test_acc = self.test(test_data)
    #     print('train-loss=%.4f;train-acc=%.4f;test-acc=%.4f;' % (train_loss, train_acc, test_acc))
    #     self.log_loss.append([train_loss, test_loss])

    def run(self, train_data, test_data, test_data1, test_data2):
      ''' function for training and testing the trained model '''
      print('training...')

      for idx in xrange(self.nepoch):
        print('epoch '+str(idx)+'...')
        train_loss, train_acc = self.train(train_data)
        test_loss, test_acc = self.test(test_data)
        print('train-loss=%.4f;train-acc=%.4f;test-acc=%.4f;' % (train_loss, train_acc, test_acc))
        test_loss, test_acc = self.test(test_data1)
        print('train-loss=%.4f;train-acc=%.4f;test-acc-withRul=%.4f;' % (train_loss, train_acc, test_acc))
        test_loss, test_acc = self.test(test_data2)
        print('train-loss=%.4f;train-acc=%.4f;test-acc-w/o-Rul=%.4f;' % (train_loss, train_acc, test_acc))
        self.log_loss.append([train_loss, test_loss])
        
