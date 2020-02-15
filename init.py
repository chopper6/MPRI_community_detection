import networkx as nx, numpy as np, random as rd, time, math, heapq
from util import *
from scipy import sparse



# PARAM INIT, GRAPH BUILDS AND GRAPH IMPORTS


#####################################################################
def params(param_file):
	# EXPECT A TAB DELITED FILE: name\value\dtype
	if not os.path.isfile(param_file):
		assert(False) # unknown file
	with open(param_file) as pfile:
		lines = pfile.readlines()
		pdict = {} # PARAMS
		for line in lines[1:]: #skip header line
			if not line.isspace():
				param = line.strip().split('\t')
				if param[2] in ['str','string']:
					val = param[1]
				elif param[2] == 'int':
					val = int(param[1])
				elif param[2] == 'float':
					val = float(param[1])
				elif param[2] == 'bool':
					val = bool(param[1])
				elif param[2] == 'listInt':
					vals = removes(['[',']'],param[1])
					vals = vals.split(',')
					val = [int(vals[i]) for i in range(len(vals))]

				elif param[2] == 'listStr':
					vals = removes(['[',']'],param[1])
					vals = vals.split(',')
					val = [str(vals[i]) for i in range(len(vals))]
				else: assert(False) #unknown val
				pdict[param[0]] = val

	return pdict


######################################################################

def init_nodes(G,params):
	for node in G.nodes():
		G.nodes[node]['rank'] = 0


def init_seed(G, params):
	for node in G.nodes():
		G.nodes[node]['seed'] = False

	if params['seed_set'] == 'random':
		seeds = []
		for i in range(params['num_seeds']):
			seed = rd.choice(G.nodes())
			j=0
			while seed in seeds: 
				seed = rd.choice(G.nodes())
				j+=1
				if j> 100000: assert(False) #infinite loop
			seeds += [seed]

	else:
		seeds = params['seeds']

	for seed in seeds:
		G.nodes[seed]['seed'] = True
		G.nodes[seed]['rank'] = 1
	G.graph['seeds'] = seeds



#####################################################################


def build_global(params):

	# TOY NETS
	if params['build'] == 'sep_cliques':
		G= sep_cliques(params)
		return prep_global(params,G)
	elif params['build'] in ['two_triangles','twoT']:
		G = two_triangles(params)
		return prep_global(params,G)
	elif params['build'] == 'scale-free':
		G = scale_free(params)
		return prep_global(params,G)

	# REAL DATA	
	elif params['build'] == 'encode':
		return import_and_prep_global(params,'./input/Human-ENCODE-K562.txt')
	elif params['build'] == 'bacteria-ppi':
		return import_and_prep_global(params,'./input/Bacteria-PPI.txt')
	elif params['build'] == 'bacteria-reg':
		return import_and_prep_global(params,'./input/Bacteria-RegulonDB.txt')
	elif params['build'] == 'human-ppi':
		return import_and_prep_global(params,'./input/Human-PPI.txt')
	elif params['build'] == 'human-ppi-iso':
		return import_and_prep_global(params,'./input/Human-PPI-Iso.txt')
	else:
		assert(False) #unknown build parameter


def build(params):
	if params['build'] == 'toy1':
		G = toy1(params)
	elif params['build'] == 'toy3':
		G= toy3(params)
	elif params['build'] == 'ring':
		G= ring(params)
	elif params['build'] == 'ring2way':
		G= ring2way(params)
	elif params['build'] in ['two_triangles','twoT']:
		G= two_triangles(params)
	elif params['build'] == 'sep_cliques':
		G= sep_cliques(params)
	elif params['build'] == 'encode':
		G = import_net(params,'./input/Human-ENCODE-GM.txt')
		print("|V| ",str(len(G.nodes())),", |E|",str(len(G.edges())))
	elif params['build'] == 'human-ppi':
		G = import_net(params,'./input/Human-PPI.txt')
		print("|V| ",str(len(G.nodes())),", |E|",str(len(G.edges())))
	elif params['build'] == 'human-ppi-iso':
		G = import_net(params,'./input/Human-PPI-Iso.txt')
		print("|V| ",str(len(G.nodes())),", |E|",str(len(G.edges())))
	elif params['build'] == 'bacteria-ppi':
		G = import_net(params,'./input/Bacteria-PPI.txt')
	elif params['build'] == 'bacteria-reg':
		G = import_net(params,'./input/Bacteria-RegulonDB.txt')
	elif params['build'] == 'scale-free':
		G = scale_free(params)
	else: assert(False) #unknown

	rm_self_loops(G)
	init_nodes(G,params)
	init_seed(G,params)
	# btw really gotta write a case-switch syntax for py
	return G


def scale_free(params):
	#H = nx.barabasi_albert_graph(params['num_nodes'], 2)
	H = nx.watts_strogatz_graph(params['num_nodes'], 8,.1) #used w 200 nodes, comm size of 20, num comms 8 or something
	G = nx.empty_graph(create_using=nx.DiGraph())
	G.add_edges_from(H.edges())
	largest_cc = G.subgraph(max(nx.weakly_connected_components(G), key=len))
	return largest_cc

def rm_self_loops(G):
	rm = []
	for e in G.edges():
		if e[0] == e[1]:
			rm += [e]

	for e in rm:
		G.remove_edge(e[0],e[1]) 

def import_net(params,filename):

	G = nx.empty_graph(create_using=nx.DiGraph())
	# the ENCODE files have been pre-filtered to include only the largest CC (typically ~95% of the genes)
	# edge sign is there but excluded here

	with open(filename) as f:
		lines = f.readlines()
		for l in lines[1:]: #first line is header
			nodes = l.split(' ')
			G.add_edge(nodes[0],nodes[1])

	return G


def prep_global(params,G):


	t0 = time.time()

	edges = []
	node_map = {}
	num_nodes = len(G.nodes())
	nn = 0
	for e in G.edges():
		nodes = [e[0],e[1]]
		if nodes[0] != nodes[1] or params['self-loops']: #no self-loops
			nodes.sort()
			if nodes[0] not in node_map.keys():
				node_map[nodes[0]] = nn
				nn += 1
			if nodes[1] not in node_map.keys():
				node_map[nodes[1]] = nn
				nn += 1
			edges += [[node_map[nodes[0]],node_map[nodes[1]]]]
			edges += [[node_map[nodes[1]],node_map[nodes[0]]]]

	edges = np.array(edges)
	cols = edges[:,1]
	rows = edges[:,0]
	A = sparse.coo_matrix(([1 for i in range(len(rows))], (rows, cols)), shape=(num_nodes,num_nodes))

	degs = np.squeeze(np.asarray(A.sum(axis=0)))


	E = len(edges)
	a = degs/E

	data = []
	for l in range(len(rows)):
		i,j = rows[l], cols[l]
		data += [1/E - degs[i]*degs[j]/math.pow(E,2)]


	dQ = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes,num_nodes))
	dQ = sparse.dok_matrix(dQ)
	
	H = [dict_max(dQ,k) for k in range(num_nodes)]
	
	if params['debug']:
		for i in range(num_nodes):
			assert(dQ[i,i]==0)

	return dQ, H, a, A, node_map, G



def import_and_prep_global(params,filename):

	t0 = time.time()

	edges, nx_edges = [], []
	node_map = {}

	G = nx.empty_graph(create_using=nx.Graph())

	with open(filename) as f:
		lines = f.readlines()
		num_nodes = 0
		for l in lines[1:]: #first line is header
			nodes = l.split(' ')
			nodes = [nodes[0],nodes[1]]
			if nodes[0] != nodes[1] or params['self-loops']: #no self-loops
				nodes.sort()
				if nodes[0] not in node_map.keys():
					node_map[nodes[0]] = num_nodes
					num_nodes += 1
				if nodes[1] not in node_map.keys():
					node_map[nodes[1]] = num_nodes
					num_nodes += 1
				nx_edges += [[nodes[0],nodes[1]]]
				edges += [[node_map[nodes[0]],node_map[nodes[1]]]]
				edges += [[node_map[nodes[1]],node_map[nodes[0]]]]

	G.add_edges_from(nx_edges)
	if params['timeit']:
		t1= time.time()
		print('Time to import edges from file and build nx = ',t1-t0,'seconds.')

	edges = np.array(edges)
	cols = edges[:,1]
	rows = edges[:,0]
	A = sparse.coo_matrix(([1 for i in range(len(rows))], (rows, cols)), shape=(num_nodes,num_nodes))

	degs = np.squeeze(np.asarray(A.sum(axis=0)))

	if params['debug']: 
		in_degs = np.squeeze(np.asarray(A.sum(axis=1)))
		assert(np.array_equal(degs, in_degs))

	E = len(edges)
	a = degs/E

	data = []
	for l in range(len(rows)):
		i,j = rows[l], cols[l]
		data += [1/E - degs[i]*degs[j]/math.pow(E,2)]


	dQ = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes,num_nodes))

	dQ = sparse.dok_matrix(dQ)	

	H = [dict_max(dQ,k) for k in range(num_nodes)]
	
	if params['debug']:

		for i in range(num_nodes):
			assert(dQ[i,i]==0)

	if params['timeit']:
		t2 = time.time()
		print('Rest of sparse matrix initialization = ',t2-t1,'seconds.')

	return dQ, H, a, A, node_map, G




def two_triangles(params):
	G = nx.empty_graph(create_using=nx.DiGraph())
	G.add_nodes_from([i for i in range(8)])
	edges = [(0,1),(1,0),(1,2),(2,1),(2,0),(0,2),(2,3),(3,4),(4,5),(5,6),(6,5),(6,7),(7,6),(7,5),(5,7)]
	G.add_edges_from(edges)
	return G

def toy1(params):
	G = nx.empty_graph(create_using=nx.DiGraph())
	G.add_node('X')
	G.add_edge('X','X')
	return G

def toy3(params):
	G = nx.empty_graph(create_using=nx.DiGraph())
	G.add_nodes_from(['A','B','C'])
	G.add_edge('A','B')
	G.add_edge('B','C')
	G.add_edge('C','A')
	return G

def ring(params):
	k = params['num_nodes']
	G = nx.empty_graph(create_using=nx.DiGraph())
	for i in range(k):
		G.add_node(i)
	for i in range(k-1):
		G.add_edge(i,i+1)
	G.add_edge(k-1,0)
	return G

def ring2way(params):
	k = params['num_nodes']
	G = nx.empty_graph(create_using=nx.DiGraph())
	for i in range(k):
		G.add_node(i)
	for i in range(k-1):
		G.add_edge(i,i+1)
		G.add_edge(i+1,i)
	G.add_edge(k-1,0)
	G.add_edge(0,k-1)
	return G

def sep_cliques(params):

	G = nx.empty_graph(create_using=nx.DiGraph())
	edges = [(0,1),(0,2),(0,3),(2,4),(2,5),(2,3),(3,4),(3,5),(4,5),(4,6)]
	edges += [(6,7),(6,8),(6,9),(7,8),(7,9),(8,9)]
	G.add_edges_from(edges)
	return G
