#!/usr/bin/python
from stanfordcorenlp import StanfordCoreNLP as snlp
from nltk.tree import *
import numpy as np
import networkx as nx

class Deptree:
	def __init__(self):
		self.nlp = snlp(r'./../resources/stanford-corenlp-full-2018-01-31')
	def adjMatrix(self,sentence,l=0.1):
		'''Returns the adjacency matrix using path len as the distance between two nodes'''
		pos_tags = self.nlp.pos_tag(sentence)
		deps = self.nlp.dependency_parse(sentence)
		G = nx.DiGraph()
		G.add_node(0,pos="Root")
		#adding nodes
		for node_id, node_tag in enumerate(pos_tags,1):
			G.add_node(node_id, pos=node_tag[1])
		#adding edges
		for dep in deps:
			G.add_edge(dep[2], dep[1], rel=dep[0])
		#setting HIN for dependency parse
		self.G = G
		p=nx.shortest_path(self.G.to_undirected())
		n = nx.number_of_nodes(self.G) - 1
		adj = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				adj[i][j] = len(p[i+1][j+1]) - 1
				adj[j][i] = adj[i][j]
		diam = np.max(adj)
		adj = adj/diam
		adj = -adj*adj/(2*l*l)
		adj = np.exp(adj)
		self.adj = adj
		return adj

	def degMatrix(self, sentence=None, l=0.1, adj = None):
		'''Returns the degree matrix'''
		if adj is None:
			adj = self.adjMatrix(sentence,l)
		n = nx.number_of_nodes(self.G) - 1
		sum_arr = np.apply_along_axis(np.sum,1,adj)
		degMat = np.zeros((n,n),dtype=np.float64)
		for i in range(sum_arr.shape[0]):
			degMat[i][i] = sum_arr[i]
		return degMat

	def getRequiredParameters(self, sentence, aspect_words_indexes, l=0.1):
		'''Returns DeltaInverse_mm and Wights_ma'''
		W = self.adjMatrix(sentence, l)
		deg = self.degMatrix(adj=W)
		D = deg - W
		DI = np.linalg.inv(D)
		DI_mm = np.delete(np.delete(DI,aspect_words_indexes,0), aspect_words_indexes, 1)
		W_am = np.delete(W[aspect_words_indexes], aspect_words_indexes, 1)
		W_m1 = np.transpose([np.mean(W_am,0)])
		return DI_mm, W_m1

def main():
	c = Constree()
	sentence = 'i saw a dog today.'
	print(c.adjMatrix(sentence))

if __name__ == '__main__':
	main()