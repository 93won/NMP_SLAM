#!usr/bin/python
# -*- coding: utf-8 -*-

import logging

import networkx as nx
import random
import numpy as np
import time

from TriangulationAlgorithms import TriangulationAlgorithm as ta

def triangulate_MCSM(G, randomized=False, repetitions=1, reduce_graph=True, timeout=-1):
	algo = Algorithm_MCSM(G, reduce_graph, timeout)
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

class Algorithm_MCSM(ta.TriangulationAlgorithm):
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
		Implementation of MCS-M Algorithm 
			Bery, Blair, Heggernes, Peyton: Maximum Cardinality Search for Computing Minimal Triangulations of Graphs
			https://link.springer.com/article/10.1007/s00453-004-1084-3
		to construct a minimal elemination ordering alpha of a graph G
		and the corresponding minimal triangulation H(G, alpha)
		
		Args:
			C : a graph in networkx format
			randomized : 
		
		Returns:
			F : a set of edges s.t. C + F is a minimal triangulation C.
		'''
		logging.info("=== triangulate_MCS_M ===")
		
		F = []
		unnumbered_nodes = [n for n in C.nodes()]
		if randomized:
			random.shuffle(unnumbered_nodes)
			
		weight = {n : 0 for n in unnumbered_nodes}
		n = len(C)
		for i in range(n,0, -1):
			# check timeout:
			if self.timeout > 0 and time.time() > self.timeout:
				raise ta.TimeLimitExceededException("Time Limit Exceeded!")

			logging.debug("Iteration: "+str(i))
			logging.debug("all unnumbered nodes:")
			logging.debug([str(n)+": "+str(weight[n]) for n in unnumbered_nodes])
			
			# get node with maximum weight:
			node_v = unnumbered_nodes[0]
			maxweight = weight[node_v]
			for n in unnumbered_nodes:
				if weight[n] > maxweight:
					node_v = n
					maxweight = weight[node_v]
			self.alpha[node_v] = i
			unnumbered_nodes.remove(node_v)
			S = []
			for node_u in unnumbered_nodes:
				if not node_u == node_v:
					unnumbered_lowerweight_nodes = [node_x for node_x in unnumbered_nodes if weight[node_x] < weight[node_u]]
					if nx.has_path(C.subgraph(unnumbered_lowerweight_nodes+[node_v, node_u]),node_v, node_u):
						logging.debug("Add target node "+str(node_u)+" to set S")
						S.append(node_u)
			for node_u in S:
				weight[node_u] += 1
				if not (node_v, node_u) in C.edges():
					F.append((node_v, node_u))
					
			logging.debug("End of iteration. all node labels:")
			logging.debug([str(n)+": "+str(weight[n]) for n in C])		
		
		return F
		