#!usr/bin/python
# -*- coding: utf-8 -*-

import logging

import networkx as nx
import random
import numpy as np
import time
import copy

from TriangulationAlgorithms import TriangulationAlgorithm as ta

def triangulate_CMT(G, randomized=False, repetitions=1, reduce_graph=True, timeout=-1):
	algo = Algorithm_CMT(G, reduce_graph, timeout)
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

class Algorithm_CMT(ta.TriangulationAlgorithm):
	'''
	Implementation of the algorithm "Clique Minimal Triangulation" 
		Mezzini, Moscarini: Simple algorithms for minimal triangulation of a graph and backward selection of a decomposable Markov network
		https://www.sciencedirect.com/science/article/pii/S030439750900735X
	to construct a minimal triangulation H of a graph G
	
	Args:
		G : a graph in netwokx format
	'''
	
	def __init__(self, G, reduce_graph=True, timeout=-1):
		logging.info("=== CMT.Algorithm_CMT.init ===")
		super().__init__(G, reduce_graph, timeout)

	def run(self):
		for C in self.component_subgraphs:
			# get triangulation for each connected component of the reduced graph G_c:
			logging.debug("Next component: "+str(C.nodes()))

			F = self.get_edges_of_inverse_graph(C)
			logging.debug("possible chord edges of this component: "+str(F))

			self.edges_of_triangulation += self.minimize_triangulation(C, F, False)
		
		self.H = self.G.copy()
		self.H.add_edges_from(self.edges_of_triangulation)
		
		if not nx.is_chordal(self.H):
			raise ta.TriangulationNotSuccessfulException("Resulting graph is somehow not chordal!")

	def run_randomized(self):
		self.edges_of_triangulation = []
		for C in self.component_subgraphs:
			# get triangulation for each connected component of the reduced graph G_c:
			logging.debug("Next component: "+str(C.nodes()))
			
			F = self.get_edges_of_inverse_graph(C)
			#logging.debug("possible chord edges of this component: "+str(F))

			self.edges_of_triangulation += self.minimize_triangulation(C, F, True)
		
		self.H = self.G.copy()
		self.H.add_edges_from(self.edges_of_triangulation)
		
		if not nx.is_chordal(self.H):
			raise ta.TriangulationNotSuccessfulException("Resulting graph is somehow not chordal!")

	def minimize_triangulation(self, G, F, randomized, T=None):
		'''
		Minimize a given triangulation.
		
		Following the pseudocode given in:
		Mezzini, Moscarini: Simple algorithms for minimal triangulation of a graph and backward selection of a decomposable Markov network (see above)
		
		Args:
			G : a graph
			F : a set of edges, s.t. G + F is chordal
			randomized : if set to True, get_removeable_edge is randomized
			T : a pre-initialized database of removeable edges. If G + F is not a full graph, it is required to specify T!
			
		Returns:
			F_prime : a subset of F, s.t. G + F_prime is minimal chordal
		'''
		logging.info("=== CMT.minimize_triangulation ===")
		
		F_prime = copy.deepcopy(F)
		H = G.copy()
		H.add_edges_from(F_prime)

		# initialize set V_F of all nodes that are endpoint of some edge in F:
		V_F = list(set([v for e in F_prime for v in e]))
		
		# initialize T: all edges e with len(T[e]) == 0 are removeable
		if T == None:
			T  = {}
			for edge in F_prime:
				T[edge] = set()

		edge_uv = self.get_removeable_edge(F_prime, T, randomized)
		# while there are removeable edges:
		while (not edge_uv == None):
			# check timeout:
			if self.timeout > 0 and time.time() > self.timeout:
				raise ta.TimeLimitExceededException("Time Limit Exceeded!")
				
			logging.debug("Consider removeable edge "+str(edge_uv))
			u = edge_uv[0]
			v = edge_uv[1]
			for edge_rs in T:
				r = edge_rs[0]
				s = edge_rs[1]
				cn_of_rs_cap_VF = [n for n in self.get_common_neighborhood(H,[r,s]) if n in V_F]
				if u in cn_of_rs_cap_VF and v in cn_of_rs_cap_VF:
					T[edge_rs].add(edge_uv)
			cn_of_uv_cap_VF = [n for n in self.get_common_neighborhood(H,[u,v]) if n in V_F]
			for x in cn_of_uv_cap_VF:
				if (u,x) in F_prime:
					for e in [e for e in T[(u,x)] if v in e]:
						T[(u,x)].discard(e)
				if (v,x) in F_prime:
					for e in [e for e in T[(v,x)] if u in e]:
						T[(v,x)].discard(e)
			F_prime.remove(edge_uv)
			H.remove_edges_from([edge_uv])
			edge_uv = self.get_removeable_edge(F_prime, T, randomized)

		return F_prime
		
	def get_edges_of_inverse_graph(self, G):
		'''
		computes all edges that are not in G

		Args:
			G : a graph in networkx format

		Returns:
			F : a set of edges s.t. no edge from F is in G and G + F is a complete graph
		'''

		F = []
		E = [e for e in G.edges()]
		V = list(G.nodes())
		for i in range(len(G.nodes())):
			for j in range(i+1, len(G.nodes())):
				e = (V[i],V[j])
				if e not in E:
					F.append(e)
		return F

	def get_removeable_edge(self, F, T, randomized):
		'''
		Finds and returns an edge e form the set F with T[e] = {}
		
		Args:
			F : A set of edges
			T : The dictionary that tracks removeability for all edges
			randomized : if set to True, in each iteration the next removeable edge is chosen randomly.
							Otherwise, the first edge of F that satisfies the requirements is returned.
		
		Return:
			A removeable edge (i.e. T[e] is empty), if one exists. Otherwise None.
		
		'''
		logging.info("=== CMT.get_removeable_edge ===")
		
		removeable_edges = [e for e in F if len(T[e]) == 0]
		logging.debug("Number of removable edges: "+str(len(removeable_edges)))
		if len(removeable_edges) > 0:
			if not randomized:
				return removeable_edges[0]
			else:
				return random.choice(removeable_edges)
		else:
			return None

	def get_common_neighborhood(self, G, V):
		'''
		Computes the common neighborhood of V in G, that is:
			NC(V') = {w in V(G) | forall v in V': w in N(v)}
		'''
		neighbor_count = {}
		
		for i in range(len(V)):
			for w in G.neighbors(V[i]):
				if not w in neighbor_count:
					neighbor_count[w] = 0
				neighbor_count[w] += 1
		
		common_neighbothood = []
		for w in neighbor_count:
			if neighbor_count[w] == len(V):
				common_neighbothood.append(w)
				
		return common_neighbothood