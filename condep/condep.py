#!/usr/bin/python
from stanfordcorenlp import StanfordCoreNLP as snlp
import networkx as nx
from nltk.tree import *
import numpy as np
import constree
import deptree

class condepTree:
	def __init__(self):
		self.nlp = snlp(r'./../resources/stanford-corenlp-full-2018-01-31')
		self.deptree = deptree.Deptree()
		self.contree = constree.Constree()
	def getRequiredParameters(self, sentence, aspect_words_indexes):
		_,Wm1_CT = self.deptree.getRequiredParameters(sentence, aspect_words_indexes)
		_, Wm1_DT = self.contree.getRequiredParameters(sentence, aspect_words_indexes)
		return np.array([Wm1_DT.transpose()[0], Wm1_CT.transpose()[0]])


def main():
	c = condepTree()
	# sentence = 'Good and bad sound.'
	sentence = 'Ipod is the best, affordable, cheap player'
	# sentence = "Does the player play dvd with audio and video?"
	print c.gen_matrix(sentence)
	print c.getRequiredParameters(sentence,[8])

if __name__ == '__main__':
	main()