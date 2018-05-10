# GraphEnhancedMemNet
Various deep learning models for Aspect Based sentiment Analysis using attention based memory networks.

root:
		constree.py : for constructing constituency tree
		data.py : prepares the data for training using constree.py
		model.py : tensorflow model for semantic + con approaches
		main.py : generates data and runs model

metagraph:
		metagraph.py : for constructing metagraph out of heterogeneous graph of POS and dependency str
		data.py : prepares the data for training using constree.py
		model.py : tensorflow model for semantic + rul approaches
		main.py : generates data and runs model
