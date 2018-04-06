#!/usr/bin/python
from stanfordcorenlp import StanfordCoreNLP as snlp
import networkx as nx
from nltk.tree import *
import numpy as np

class HINtree:
	def __init__(self):
		self.nlp = snlp(r'./../resources/stanford-corenlp-full-2018-01-31')
		self.rulesG = initialise_all_rules()

	def gen_networkx(self,sentence):
		'''Generates the metagraph'''
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

	def gen_matrix(self,sentence):
		'''Returns the adjacency matrix for rule based metagraph'''
		if self.rulesG is None:
			raise ValueError("No rules defined!")
		self.gen_networkx(sentence)
		#ruleG is a path(networkx) with each node('pos_tags') and each edge('rels') with '0' as start and '1' as end
		n = len(self.G.nodes())-1 #number of words in the sentence
		rule_mats = []
		for ruleG in self.rulesG:	
			adj_mat = np.zeros((n,n))
			for i in range(1,n+1):
				for j in range(1,n+1):
					if i==j:
						continue
					val = 1
					paths = nx.all_simple_paths(self.G.to_undirected(),i,j)
					paths = list(paths)
					if len(paths)!=1: #only a path in dependency TREE
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
					adj_mat[i-1][j-1] = val
			rule_mats.append(adj_mat)
		return rule_mats

	def getRequiredParameters(self, sentence, aspect_words_indexes):
		'''Returns the Wma matrix'''
		mats = self.gen_matrix(sentence)
		Ms = []
		Ws = []
		for mat in mats:
			Ws.append(np.any(mat[aspect_words_indexes,],0))
			Ms.append(np.delete(np.delete(mat, aspect_words_indexes, axis=1), aspect_words_indexes, axis=0))
			Ws.append(np.any(mat[:,aspect_words_indexes],1))
			Ms.append(np.delete(np.delete(np.transpose(mat,[1,0]), aspect_words_indexes, axis=1), aspect_words_indexes, axis=0))
		Ws = np.array(Ws)
		Ws = np.delete(Ws, aspect_words_indexes, axis=1) #dimension is RulesCount * ContextWords
		return Ws, Ms


def initialise_all_rules():
	Rules = []
	MR = {'advmod','amod', 'npadvmod', 'quantmod', 'rcmod', 'tmod', 'vmod', 'csubj','xsubj','nsubj','pobj','dobj','iobj'}
	MR1 = {'neg','advmod','amod', 'npadvmod', 'quantmod', 'rcmod', 'tmod', 'vmod', 'csubj','xsubj','nsubj','pobj','dobj','iobj'}
	
	#rule1 (O -> T)
	ruleG = nx.DiGraph()
	nx.add_path(ruleG,[0,1])
	ruleG.edges()[(0,1)]['rels'] = MR
	ruleG.nodes()[0]['pos_tags'] = {}
	ruleG.nodes()[1]['pos_tags'] = {'NN'}	
	Rules.append(ruleG)

	#rule2 (O -> H <- T)
	ruleG = nx.DiGraph()
	ruleG.add_edges_from([(0,2),(1,2)])
	ruleG.edges()[(0,2)]['rels'] = MR
	ruleG.edges()[(1,2)]['rels'] = MR
	ruleG.nodes()[0]['pos_tags'] = {}
	ruleG.nodes()[1]['pos_tags'] = {'NN'}
	ruleG.nodes()[2]['pos_tags'] = {}
	Rules.append(ruleG)

	#rule3 (T -> T)
	ruleG = nx.DiGraph()
	nx.add_path(ruleG,[0,1])
	ruleG.edges()[(0,1)]['rels'] = {'conj'}
	ruleG.nodes()[0]['pos_tags'] = {}
	ruleG.nodes()[1]['pos_tags'] = {'NN'}
	Rules.append(ruleG)

	# ruleG = nx.DiGraph()
	# nx.add_path(ruleG,[0,1])
	# ruleG.edges()[(0,1)]['rels'] = MR1
	# ruleG.nodes()[0]['pos_tags'] = {}
	# ruleG.nodes()[1]['pos_tags'] = {'NN'}	
	# Rules.append(ruleG)
	
	return Rules


def main():
	c = HINtree()
	# sentence = 'Good and bad sound.'
	sentence = 'Ipod is the best, affordable, cheap player'
	# sentence = "Does the player play dvd with audio and video?"
	print c.gen_matrix(sentence)
	print c.getRequiredParameters(sentence,[8])

if __name__ == '__main__':
	main()