#!usr/bin/python
# -*- coding: utf-8 -*-

import logging

import networkx as nx
import random
import numpy as np
import time

from TriangulationAlgorithms import TriangulationAlgorithm as ta

def triangulate_SMS(G, randomized=False, repetitions=1, reduce_graph=True, timeout=-1):
	algo = Algorithm_SMS(G, reduce_graph, timeout)
	if not randomized:
		algo.run()
		return {
			"H" : algo.get_triangulated(),
			"size" : len(algo.get_triangulation_edges()),
			"mean" : len(algo.get_triangulation_edges()),
			"variance" : 0,
			"repetitions" : 1
			}
	else:
		H_opt = None
		size_opt = None
		all_sizes = []
		for i in range(repetitions):
			algo.run_randomized()
			all_sizes.append(len(algo.get_triangulation_edges()))
			if H_opt == None or len(algo.get_triangulation_edges()) < size_opt:
				H_opt = algo.get_triangulated()
				size_opt = len(algo.get_triangulation_edges())
		return {
			"H" : H_opt,
			"size" : size_opt,
			"mean" : np.mean(all_sizes),
			"variance" : np.var(all_sizes),
			"repetitions" : repetitions
			}
	
class Algorithm_SMS(ta.TriangulationAlgorithm):
	def __init__(self, G, reduce_graph=True, timeout=-1):
		logging.info("=== SMS.Algorithm_SMS.init ===")
		super().__init__(G, reduce_graph, timeout)

	def triangulate(self, C, randomized=False):
		'''
		Implementation of the algorithm SMS (Saturate Minimal Seperators)
		to construct a minimal triangulation
		
		Args:
			C : a graph in networkx format
			randomized : if true, the algorithm to find a minimal separator is randomized.
		
		Returns:
			F : a set of edges s.t. C + F is a minimal triangulation C.
		'''
		logging.info("=== triangulate_SMS ===")
		
		F = []
		G_prime = C.copy()
		
		self.node_processing_order = [i for i in range(len(C.nodes()))]#[n for n in G.nodes()]
		if randomized:
			random.shuffle(self.node_processing_order)
		logging.debug("Node processing order: "+str(self.node_processing_order))
		
		finished = False
		while not finished:
			# check timeout:
			if self.timeout > 0 and time.time() > self.timeout:
				raise ta.TimeLimitExceededException("Time Limit Exceeded!")

			logging.debug("Next iteration")
			logging.debug("edges of G_prime: "+str(G_prime.edges()))
			separator = self.get_minimal_separator(G_prime)
			if separator == None:
				finished = True
			else:
				edges_to_add = []
				logging.debug("saturate separator "+str(separator))
				for i_u in range(len(separator)):
					for i_v in range(i_u+1, len(separator)):
						logging.debug("saturate pair "+str(separator[i_u])+","+str(separator[i_v]))
						if not G_prime.has_edge(separator[i_u], separator[i_v]):
							edges_to_add.append((separator[i_u], separator[i_v]))
							F.append((separator[i_u], separator[i_v]))
				G_prime.add_edges_from(edges_to_add)
		return F
			
	def get_minimal_separator(self, G):
		'''
		Searches for a minimal separator in G that is not a clique
			
		see https://cstheory.stackexchange.com/questions/29464/algorithms-for-computing-the-minimal-vertex-separator-of-a-graph
		
		Args:
			G : a graph in networkx format.
			## randomized : if true, the order in which the nodes are processed gets shuffled
		
		Return:
			S : a set of nodes that form a minimal separator of G, if such a set exists that is not a clique. If every minimal separator of G is a clique, returns None
		'''
		logging.info("=== get_minimal_separator ===")
		
		for k in range(len(self.node_processing_order)):
			i = self.node_processing_order[k]
			u = list(G.nodes())[i]
			neighbors_of_u = list(G.neighbors(u))
			for j in self.node_processing_order[k+1:]:
				v = list(G.nodes())[j]
				logging.debug("Consider pair: "+str(u)+", "+str(v))
				# construct S as set of all nodes that are neighbors of u and reachable from v:
				S = []
				marked = {n : False for n in G.nodes()}
				if v in neighbors_of_u:
					logging.debug(str(u)+" and "+str(v)+" are adjacent, no separator exists")
				else:
					bfs_stack = [v]
					marked[v] = True
					while not len(bfs_stack) == 0:
						w = bfs_stack.pop(0)
						for n in G.neighbors(w):
							if not marked[n]:
								marked[n] = True
								if n in neighbors_of_u:
									logging.debug("Add "+str(n)+" to separator set")
									S.append(n)
								else:
									bfs_stack.append(n)
					logging.debug("Seperator: "+str(S))
				if len(S) > 1:
					S_graph = G.subgraph(S)
					logging.debug(S_graph.nodes())
					logging.debug(S_graph.edges())
					if len(S_graph.edges()) < (len(S)*(len(S)-1))/2:
						logging.debug(str(S) + " is a minimal seperator of G")
						return S
					else:
						logging.debug(str(S) + " is a clique")
		logging.debug("No non-clique minimal seperator of G found")
		return None
		