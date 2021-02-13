#!usr/bin/python
# -*- coding: utf-8 -*-

import logging

import networkx as nx
import random
import numpy as np
import time

from TriangulationAlgorithms import TriangulationAlgorithm as ta




def triangulate_LexM(G, randomized=False, repetitions=1, reduce_graph=True, timeout=-1):
	algo = Algorithm_LexM(G, reduce_graph, timeout)
	if not randomized:
		algo.run()
			
		return {
			"H" : algo.get_triangulated(),
			"size" : len(algo.get_triangulation_edges()),
			"alpha" : algo.get_alpha(), 
			"mean" : len(algo.get_triangulation_edges()),
			"variance" : 0,
			"repetitions" : 1
			}
	else:
		H_opt = None
		alpha_opt = None
		size_opt = None
		all_sizes = []
		for i in range(repetitions):
			algo.run_randomized()
			all_sizes.append(len(algo.get_triangulation_edges()))
			if H_opt == None or len(algo.get_triangulation_edges()) < size_opt:
				H_opt = algo.get_triangulated()
				alpha_opt = algo.get_alpha()
				size_opt = len(algo.get_triangulation_edges())
		return {
			"H" : H_opt,
			"size" : size_opt,
			"alpha" : alpha_opt, 
			"mean" : np.mean(all_sizes),
			"variance" : np.var(all_sizes),
			"repetitions" : repetitions
			}

class Algorithm_LexM(ta.TriangulationAlgorithm):
	'''
	Args:
		G : a graph in netwokx format
		randomize : if set to True, the order in which the nodes are processed is randomized
	
	Returns:
		H : a minimal triangulation of G.
		alpha : the corresponding minimal elimination ordering of G 
	'''
	
	def __init__(self, G, reduce_graph=True, timeout=-1):
		logging.info("=== LexM.Algorithm_LexM.init ===")
		super().__init__(G, reduce_graph, timeout)
		self.alpha = {}

	def get_alpha(self):
		return self.alpha

	def triangulate(self, C, randomized=False):
		'''
		Implementation of LEX M Algorithm 
			Rose, Tarjan, Lueker: Algorithmic Aspects of Vertex Elimination on Graphs
			https://epubs.siam.org/doi/abs/10.1137/0205021
		to construct a minimal elemination ordering alpha of a graph G
		and the corresponding minimal triangulation H(G, alpha)
		
		Args:
			C : a graph in networkx format
			randomized : if true, the algorithm get_maxlex_node is randomized.
		
		Returns:
			F : a set of edges s.t. C + F is a minimal triangulation C.
		'''
		logging.info("=== triangulate_LEX_M ===")
		
		F = []
		n = len(C)
		nodelabels = {node : [] for node in C}
		
		all_unnumbered_vertices = [n for n in C if n not in self.alpha]
		if randomized:
			random.shuffle(all_unnumbered_vertices)
		
		for i in range(n,0, -1):
			# check timeout:
			if self.timeout > 0 and time.time() > self.timeout:
				raise ta.TimeLimitExceededException("Time Limit Exceeded!")

			logging.debug("Iteration: "+str(i))
			node_v = self.get_maxlex_node(C, nodelabels, randomized)
			logging.debug("max lex node: "+str(node_v))
			self.alpha[node_v] = i
			all_unnumbered_vertices.remove(node_v)
			S = []
			logging.debug("all unnumbered nodes:")
			logging.debug([str(n)+": "+str(nodelabels[n]) for n in all_unnumbered_vertices])
			for node_u in all_unnumbered_vertices:
				smallerlex_nodes = [n for n in all_unnumbered_vertices if list_lexicographic_is_less_than(nodelabels[n], nodelabels[node_u])]+[node_v, node_u]
				logging.debug("start Node "+str(node_v)+" label: "+str(nodelabels[node_v]))
				logging.debug("target Node "+str(node_u)+" label: "+str(nodelabels[node_u]))
				if nx.has_path(C.subgraph(smallerlex_nodes),node_v, node_u):
					logging.debug("Add target node "+str(node_u)+" to set S")
					S.append(node_u)
			for node_u in S:
				nodelabels[node_u].append(i)
				if (node_v, node_u) not in C.edges():
					F.append((node_v, node_u))
					logging.debug("added edge: "+str((node_v, node_u)))
			logging.debug("End of iteration. all node labels:")
			logging.debug([str(n)+": "+str(nodelabels[n]) for n in C])		
		
		return F
		
	def get_maxlex_node(self, G, nodelabels, randomized=False):
		'''
		Get an unnumbered vertex v of lexicograpohically maximum label from G
	
		Args:
			G : a graph in networkx format
			randomized : if set to True and if there are multiple nodes with the max lex. label, one of these is returned at random
	
		Returns:
			v : an unnumbered vertex v of lexicograpohically maximum label from G
		'''
		logging.info("=== get_maxlex_node ===")
			
		current_max_label = ''
		current_best_node = None
		nodes = [n for n in G]
		if randomized:
			random.shuffle(nodes)
		for node in G: 
			if (node not in self.alpha) and ((current_best_node == None) or (list_lexicographic_is_less_than(current_max_label, nodelabels[node]))):
				current_best_node = node
				current_max_label = nodelabels[node]
		return current_best_node
	
def list_lexicographic_is_less_than(list_1, list_2):
	'''
	computes a lexicographic ordering relation of two lists
	if list_1 < list_2 returns True
	otherwise false
		
	Args:
		list_1 : a list of integers
		list_2 : a list of integers

	Return:
		True, if list_1 < list_2 as defined above, otherwise False
	'''
	#logging.info("=== list_lexicographic_is_less_than ===")
		
	n = min(len(list_1), len(list_2))
	for i in range(n):
		if list_1[i] < list_2[i]:
			return True
		elif list_1[i] > list_2[i]:
			return False
	if len(list_1) < len(list_2):
		return True
	elif len(list_1) > len(list_2):
		return False
	return False
