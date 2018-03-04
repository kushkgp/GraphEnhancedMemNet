#!/usr/bin/python
from stanfordcorenlp import StanfordCoreNLP as snlp
import networkx as nx
from nltk.tree import *
import numpy as np

class HINtree:
	def __init__(self):
		self.nlp = snlp(r'./../resources/stanford-corenlp-full-2018-01-31')

	def gen_networkx(self,sentence):
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
		#setting HIN for dependency parse and POS
		self.G = G

	def gen_matrix(self,ruleG):
		#ruleG is a path(networkx) with each node('pos_tags') and each edge('rels') with '0' as start and '1' as end
		n = len(self.G.nodes())-1 #number of words in the sentence
		adj_mat = np.zeros((n,n))
		for i in range(0,n):
			for j in range(0,n):
				if i==j:
					continue
				val = 1
				paths = nx.all_simple_paths(self.G.to_undirected(),i,j)
				paths = list(paths)
				if len(paths)!=1:
					raise ValueError('Incorrect parsed sentence')
				path = paths[0]

				paths = nx.all_simple_paths(ruleG.to_undirected(),0,1)
				paths = list(paths)
				if len(paths)!=1:
					raise ValueError('Incorrect rule')
				rule_path = paths[0]
				if len(path)!=len(rule_path):
					val = 0 #since if the pathlens are different, the metapath cant hold here
					continue
				for k in range(0,len(path)):#matching nodes
					if self.G.nodes()[path[k]]['pos'] not in ruleG.nodes()[rule_path[k]]['pos_tags'] and len(ruleG.nodes()[rule_path[k]]['pos_tags'])!=0:
						val = 0
				for k in range(0,len(path)-1): #matching edges
					if ruleG.has_edge(rule_path[k],rule_path[k+1]):
						if not ( self.G.has_edge(path[k],path[k+1]) and self.G.edges()[(path[k],path[k+1])]['rel'] in ruleG.edges()[(rule_path[k],rule_path[k+1])]['rels']) and len(ruleG.edges()[(rule_path[k],rule_path[k+1])]['rels'])!=0:
							val=0
					else:
						if not ( self.G.has_edge(path[k+1],path[k]) and self.G.edges()[(path[k+1],path[k])]['rel'] in ruleG.edges()[(rule_path[k+1],rule_path[k])]['rels']) and len(ruleG.edges()[(rule_path[k+1],rule_path[k])]['rels'])!=0:
							val=0
				adj_mat[i][j] = val
		return adj_mat

def main():
	c = HINtree()
	sentence = 'Good and bad sound.'
	c.gen_networkx(sentence)
	ruleG = nx.DiGraph()
	nx.add_path(ruleG,[0,1])
	MR = {'advmod','amod', 'npadvmod', 'quantmod', 'rcmod', 'tmod', 'vmod', 'csubj','xsubj','nsubj','pobj','dobj','iobj'}
	ruleG.edges()[(0,1)]['rels'] = MR
	ruleG.nodes()[0]['pos_tags'] = {}
	ruleG.nodes()[1]['pos_tags'] = {'NN'}
	print c.gen_matrix(ruleG)

if __name__ == '__main__':
	main()