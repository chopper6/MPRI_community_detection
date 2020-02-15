import networkx as nx, numpy as np
from util import *

DEBUG = False

def clauset(G, params):
	seed = params['seeds']
	comm_size = params['community_size']
	assert(len(seed) <= comm_size)

	Dendo = [[key for key in params['seeds']]]

	B = seed.copy() #border of community
	C = seed.copy() #community, incld border
	U = [] #not in community, touching B

	recalc_next(G, B, C, U) #i.e. U

	R,iters=0,0
	Rs = []
	layer = []

	while len(C) < comm_size and len(C) < len(G.nodes()) and U != []:

		R_change = np.array([calcRchange(G,U[i],params,R,U,B) for i in rng(U)])

		if params['use_pagerank']:
			R_change = R_change*np.array([G.nodes[U[i]]['cardinal_rank'] for i in rng(U)]) #curr assumes that pagerank vists all

		j = np.argmax(np.array(R_change))
		if params['dR']:
			if R_change[j]<0 and abs(R_change[j])>R*.1: 
				Dendo += [layer.copy()]
				layer = []

				
		layer += [U[j]]
		C += [U[j]]
		B += [U[j]] # recalc_border will remove if it has no relevant links
		del U[j]

		# make more effic? expect B to be small
		# note that calcZ and recalc border are recalc'd since calcRchange
		# but expect the time required to be small

		recalc_border(G, B, C) #update B
		recalc_next(G, B, C, U) #update U
		
		R += R_change[j]
		Rs += [R]
		iters += 1

		if iters % 10 == 0: print('End of iter %s: R = %s, |C| = %s, |B| = %s, |U| = %s' %(iters, R, len(C), len(B), len(U)))
		#print(B,C,U)

	Dendo += [layer.copy()]
	add_statuses(G,C,B,U)
	if params['debug']:
		for d in Dendo:
			for dd in d:
				assert(dd in C)
	return C, U, Rs, Dendo

def add_statuses(G,C,B,U):
	for node in C:
		G.nodes[node]['status'] = 'C'
	for node in B: #these will override C
		G.nodes[node]['status'] = 'B'
	for node in U:
		assert(node not in B and node not in C)
		G.nodes[node]['status'] = 'U'

	return C


def calcRchange(G,u,params, R, U, B):
	# 'u' is the candidate to add to the community

	T= len(G.edges(B))
	x,y=0,0
	z=calcZ(G,B,u,U)

	#undirected
	B_out = [e[1] for e in G.out_edges(B)]
	B_in = [e[0] for e in G.in_edges(B)] 

	for e in G.in_edges(U):
		if e[0] in B_out:
			x+=1
		else: y+=1

	for e in G.out_edges(U):
		if e[1] in B_in:
			x+=1
		else: y+=1

	R_change = (x - R*y - z*(1-R)) / (T-z+y)
	return R_change



def calcZ(G,B,u,U): #where u is the candidate node
	V,z = [],0 # V is nodes that will leave B if u is added to C
	all_edges = list(G.in_edges(u))+list(G.out_edges(u))
	for e in all_edges:
		if e[1]==u: ngh = e[0]
		else: ngh = e[1]
		if ngh in B and ngh not in V:
			stays_in_B = False
			for e2 in G.in_edges(ngh):
				if e2[1]==ngh: ngh2 = e2[0]
				else: ngh2 = e2[1]
				if ngh2 in U:
					stays_in_B = True
					break
			if not stays_in_B:
				V += [ngh]

			for e2 in G.out_edges(ngh): # messy last minute changes, G.edges(node) only uses in_edges...
				if e2[1]==ngh: ngh2 = e2[0]
				else: ngh2 = e2[1]
				if ngh2 in U:
					stays_in_B = True
					break
			if not stays_in_B:
				V += [ngh]

	for ngh in V:
		if e2[1]==ngh: ngh2 = e2[0]
		else: ngh2 = e2[1]
		if ngh2 not in B:
			z+=1

	return z




def recalc_border(G, B, C):
	z=0
	E = []
	dels = []
	for i in rng(B):
		if DEBUG: assert(B[i] in C)
		in_border = False
		for e in G.in_edges(B[i]):
			if e[1] == B[i]:
				ngh = e[0]
				assert(e[0] != B[i]) #self-loop
			else:
				ngh = e[1]
			if ngh not in C:
				in_border = True
				break

		if not in_border:
			dels += [B[i]]

		for e in G.out_edges(B[i]): #messy last-minute code changes...
			if e[1] == B[i]:
				ngh = e[0]
				assert(e[0] != B[i]) #self-loop
			else:
				ngh = e[1]
			if ngh not in C:
				in_border = True
				break

		if not in_border and B[i] not in dels:
			dels += [B[i]]

	for d in dels:
		B.remove(d)




def recalc_next(G, B, C, U):
	for i in rng(B):
		for e in G.out_edges(B[i]):
			if e[1] not in C and e[1] not in U:
				U += [e[1]]
		for e in G.in_edges(B[i]):
			if e[0] not in C and e[0] not in U:
				U += [e[0]]