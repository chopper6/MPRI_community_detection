import random as rd
from util import *
import numpy as np


def surf(G,params):
	# where G should be a networkx DiGraph
	# initialized with: node['rank'] = 0 for all nodes
	# and seed should be a set of nodes

	damping = params['damping']
	seeds = G.graph['seeds']
	node = rd.choice(seeds) #start_node
	num_dead_ends = 0

	for i in range(params['iters']):
		
		G.nodes[node]['rank'] += 1

		out_edges = list(G.out_edges(node))

		# note: did not trim dead-ends, since still want to rank them
		# side-effect: this effectively increases the damping param
		if out_edges == [] or rd.random() < damping: #i.e dead-end or randomly chose to restart
			node = rd.choice(seeds)
			if out_edges == []:
				num_dead_ends += 1

		else:
			next_edge = rd.choice(out_edges)
			node = next_edge[1]


	nodes = np.array(G.nodes())

	orig_ranks = np.array([G.nodes[nodes[i]]['rank'] for i in rng(nodes)])
	temp = orig_ranks.argsort()
	top_k_nodes = nodes[temp[:params['community_size']]]
	ranks = np.empty_like(temp)
	ranks[temp] = np.arange(len(orig_ranks))

	minn = []
	for i in rng(nodes):
		if ranks[i]!=0:
			minn+=[ranks[i]]
	minn = min(minn)
	for i in rng(nodes):
		if ranks[i]==0:
			ranks[i]=minn
		G.nodes[nodes[i]]['cardinal_rank'] = 1/ranks[i]


	#max_rank = max([G.nodes[node]['rank'] for node in G.nodes()])
	#for node in G.nodes():
		#G.nodes[node]['rank'] /= params['iters'] 
	#	G.nodes[node]['rank'] /= max_rank

	return top_k_nodes

