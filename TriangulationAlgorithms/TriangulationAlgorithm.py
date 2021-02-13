#!usr/bin/python
# -*- coding: utf-8 -*-

import logging

import networkx as nx
try:
	import tkinter
except ImportError:
	import matplotlib
	matplotlib.use('agg')
	import matplotlib.pyplot as plt
else:
	import matplotlib.pyplot as plt

class TriangulationNotSuccessfulException(Exception):
	'''
	Custom error type that gets thrown
	when an internal check yields that the graph resulting from a trianguliation
	is somehow not chordal.
	'''

class TimeLimitExceededException(Exception):
	'''
	Custom error type that gets thrown
	when a time limit was set and is exceeded.
	'''

class TriangulationAlgorithm:
	'''
	Superclass for all triangulation algorithms

	Args:
		G : the graph to triangulate
		reduce_graph : if set to True, all single-vertex-seperators get removed from the graph,
						and a subgraph for each component gets constructed
		timeout : if timeout is set to a value > 0, it specifies the maximum time in seconds the algorithm
						is allowed to run before it gets terminated

	Attributes:
		G : the original graph
		component_subgraphs : a list of graphs in networkx format
			if G was reduced, this contains each component of the reduced G as a graph.
			otherwise, it contains only G
		H : the triangulated graph
		edges_of_triangulation : the set of edges that are added to G to achieve H
		alpha : a dict {node: int} that contains a perfect elimination ordering, if one gets constructed
	'''
	def __init__(self, G, reduce_graph=True, timeout=-1):
		self.G = G
		self.component_subgraphs = [G]
		if reduce_graph:
			self.get_relevant_components()
			self.get_chordedge_candidates()
			
		self.H = None
		self.edges_of_triangulation = []
		self.alpha = {}
		self.timeout = timeout

	def run(self):
		self.alpha = {}
		for C in self.component_subgraphs:
			# get triangulation for each connected component of the reduced graph G_c:
			self.edges_of_triangulation += self.triangulate(C)
		
		self.H = self.G.copy()
		self.H.add_edges_from(self.edges_of_triangulation)
		
		if not nx.is_chordal(self.H):
			raise ta.TriangulationNotSuccessfulException("Resulting graph is somehow not chordal!")
			
	def run_randomized(self):
		self.edges_of_triangulation = []
		self.alpha = {}
		for C in self.component_subgraphs:
			# get triangulation for each connected component of the reduced graph G_c:
			self.edges_of_triangulation += self.triangulate(C, randomized=True)
		
		self.H = self.G.copy()
		self.H.add_edges_from(self.edges_of_triangulation)
		
		if not nx.is_chordal(self.H):
			raise ta.TriangulationNotSuccessfulException("Resulting graph is somehow not chordal!")
	
	def get_triangulated(self):
		return self.H
		
	def get_triangulation_edges(self):
		return self.edges_of_triangulation

	def get_triangulation_size(self):
		return len(self.edges_of_triangulation)
		
	def get_relevant_components(self):
		logging.info("TA.get_relevant_components")
		# construct set of possible chord edges:
		# only consider subgraphs after all separators of size 1 have been removed from graph:
		cycle_nodes = list(set([n for c in nx.cycle_basis(self.G) for n in c]))
		single_node_separators = [n for n in self.G.nodes() if n not in cycle_nodes]
		self.G_c = self.G.subgraph(cycle_nodes)
		self.component_subgraphs = [self.G_c.subgraph(c) for c in nx.connected_components(self.G_c) if len(c) > 1]
		
		#logging.debug("cycle nodes: "+str(cycle_nodes))
		#logging.debug("nodes to remove: "+str(single_node_separators))
		logging.debug("Number of components of the reduced graph: "+str(len(self.component_subgraphs)))
	
	def get_chordedge_candidates(self):
		logging.info("TA.get_chordedge_candidates")
		if self.G_c == None:
			self.get_relevant_components()

		self.chordedge_candidates = []
		for c in nx.connected_components(self.G_c):
			c = list(c)
			for i in range(len(c)):
				for j in range(i+1, len(c)):
					chord_edge = (c[i], c[j])
					if chord_edge not in self.G.edges():
						self.chordedge_candidates.append(chord_edge)
		
		logging.debug ("chordedge candidates: "+str(self.chordedge_candidates))

	def draw_triangulation(self, node_size=50, font_size=16, fig_size=(14,4), path=None):
		edges_original = self.G.edges()
		plt.rcParams["figure.figsize"] = fig_size

		#pos = nx.shell_layout(self.G)
		pos = nx.kamada_kawai_layout(self.G)
		
		if not self.H == None:
			nx.draw_networkx_nodes(self.H, pos, node_color='r', node_size=node_size)
			nx.draw_networkx_edges(self.H, pos, edgelist=edges_original, width=1, edge_color='black')
			nx.draw_networkx_edges(self.H, pos, edgelist=self.edges_of_triangulation, width=1, edge_color='blue')
		else:
			nx.draw_networkx_nodes(self.G, pos, node_color='r', node_size=node_size)
			nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(), width=1, edge_color='black')

		labels = {}
		for n in self.G.nodes():
			labels[n] = n
		nx.draw_networkx_labels(self.G, pos, labels, font_size=font_size)
		plt.savefig(path, dpi=300)
		plt.axis('off')
		plt.show()
		
