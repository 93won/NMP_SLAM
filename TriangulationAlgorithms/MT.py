#!usr/bin/python
# -*- coding: utf-8 -*-

import logging

import networkx as nx
import itertools
import time

from TriangulationAlgorithms import TriangulationAlgorithm as ta
from TriangulationAlgorithms import LEX_M

def triangulate_MT(G, randomized=False, repetitions=1, reduce_graph=True, timeout=-1):
	algo = Algorithm_MinimumTriangulation(G, reduce_graph, timeout)
	algo.run()
	return {
		"H" : algo.get_triangulated(),
		"size" : len(algo.get_triangulation_edges()),
		"mean" : len(algo.get_triangulation_edges()),
		"variance" : 0,
		"repetitions" : 1
		}

class Algorithm_MinimumTriangulation(ta.TriangulationAlgorithm):
	def __init__(self, G, reduce_graph=True, timeout=-1):
		logging.info("=== MT.Algorithm_MinimumTriangulation.init ===")
		super().__init__(G, reduce_graph, timeout)
		
	def triangulate(self, C, randomized=False):
		'''
		Find a minimum triangulation of a graph
		
		Args:
			C : a graph in networkx format
			randomized : has no effect
			
		Return:
			F : a set of edges such that C + F is a minimum triangulation
		'''
		logging.info("=== MT.triangulate ===")

		if nx.is_chordal(C):
			logging.debug("Component is already chordal")
			return []
	
		# use LEX-M to determine the size of a minimal triangulation to have an upper bound for the minimum triangulation
		lexm_triang = LEX_M.triangulate_LexM(C)
		size_minimal = lexm_triang["size"]
		logging.debug("size of minimal: "+str(size_minimal))
		
		F = []
		# iterate through all subsets of chord edges by increasing set size.
		# for each subset, check if self.G + additional edges is chordal
		# return first set of edges that makes self.G chordal. this is a minimum triangulation.
		k = 1
		found_minimum = False
		#print (self.chordedge_candidates)
		while not found_minimum and k < size_minimal:
			# check timeout:
			if self.timeout > 0 and time.time() > self.timeout:
				raise ta.TimeLimitExceededException("Time Limit Exceeded!")

			logging.debug("Current iteration: consider edgesets of size "+str(k))
			edgesets_size_k = itertools.combinations(self.chordedge_candidates, k)
			k_edgeset = 0
			for edgeset in edgesets_size_k:
				#print (edgeset)
				k_edgeset += 1
				if k_edgeset%10000 == 0:
					# check timeout every 10k sets:
					if self.timeout > 0 and time.time() > self.timeout:
						raise ta.TimeLimitExceededException("Time Limit Exceeded!")
				H = C.copy()
				H.add_edges_from(edgeset)
				if nx.is_chordal(H):
					F += [e for e in edgeset]
					found_minimum = True
					break
			k += 1
		if not found_minimum:
			F += [e for e in lexm_triang["H"].edges() if e not in C.edges()]
			
		return F
		