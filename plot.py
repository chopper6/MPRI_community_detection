import networkx as nx, os, math, numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, date
from util import rng, timestamp, avg
from matplotlib.lines import Line2D


COLORS = ['blue','red','purple','orange','magenta','cyan','green','yellow']


def get_layout(params,G):

	# positions for all nodes
	if params['draw_layout'] == 'circular':
		pos = nx.circular_layout(G)  	
	elif params['draw_layout'] == 'spring':
		pos = nx.spring_layout(G) 

	elif params['draw_layout'] == 'kamada_kawai':
		pos = nx.kamada_kawai_layout(G)

	elif params['draw_layout'] == 'spectral':
		pos = nx.spectral_layout(G)
	else: assert(False) #unknown draw layout
	return pos



def drawG_by_dendos(G,localD, globalD, params):
	# pass a subgraph induced by union seed community local + global 
	plt.figure(figsize=(10,6))
	node_size, nalpha, ealpha = 200, .5, .2
	
	pos = get_layout(params,G)

	nodes = list(G.nodes())

	#for node in nodes:
	#	assert(node in np.array(localD).flatten() or node in np.array(globalD).flatten())

	labels = {n:'' for n in G.nodes()}
	colors, bs, rs = [],{n:0 for n in G.nodes()},{n:0 for n in G.nodes()}
	for i in rng(localD):
		for n in localD[i]:
			if n in nodes:
				bs[n] = (len(localD)-i)/len(localD)
				labels[n] += 'L'
	for i in rng(globalD):
		for n in globalD[i]:
			if n in nodes:
				rs[n] = (len(globalD)-i)/len(globalD)
				if labels[n] not in ['G','LG']:
					labels[n] += 'G'
	labels[params['seeds'][0]] = 'Seed'

	alphas, sizes = [],[]
	for i in rng(nodes):
		node = nodes[i]
		if node == params['seeds'][0]:
			colors += [(0,0,1)]
			alphas += [1]
			sizes += [1000]
		else:
			colors += [(rs[node],bs[node],0)]
			alphas += [max(math.pow(bs[node],.3),.05)]
			sizes += [max(rs[node],.05)*1000]

	nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=node_size, alpha=nalpha)
	#labels = {n:labels[n] for n in G.nodes()} 
	labels = {n:n for n in G.nodes()} 
	nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

	plt.title("Comparison of Hierarchical Communities", fontsize=26)
	elist = sorted(list(G.edges()))
	nx.draw_networkx_edges(G, pos, arrows=True, edgelist=elist, alpha=ealpha) 

	if params['save_fig']:
		tstamp = timestamp()
		plt.savefig(params['output_path']+'/'+str(tstamp)+'_dendos.png')
	else:
		plt.show()
	plt.clf()
	plt.close()



def drawG_by_comm_global(G,node_C,params):
	# color by seed/not_seed, alpha by rank

	assert(os.path.isdir(params['output_path']))


	plt.figure(figsize=(10,6))
	plt.title('Global Community Detection',fontsize=26)
	node_size, nalpha, ealpha = 40, .3, .1
	
	pos = get_layout(params,G)

	nodes = list(G.nodes())
	#colors = [COLORS[node_C[nodes[n]]%len(COLORS)] for n in rng(nodes)]
	normzd_C = {k:node_C[k] for k in node_C.keys()}
	normzd_C = {}
	i=0
	for k in node_C.keys():
		if node_C[k] not in normzd_C.keys():
			normzd_C[node_C[k]] = i
			i+=1

	# super messy color picking
	indx = [normzd_C[node_C[nodes[n]]]/len(normzd_C) for n in rng(nodes)]
	i1 = [(indx[n] // 0.1 / 10) for n in rng(nodes)]
	sub=[i1[i] for i in rng(i1)]
	for i in rng(i1):
		if i1[i] >= 1:
			sub[i]-=1
	i2 = [10*((indx[n] // 0.01 / 100)-sub[n]) for n in rng(nodes)]
	i3 = [10*(((indx[n] // 0.001 / 1000)-sub[n])*10-i2[n]) for n in rng(nodes)]
	colors = [(i1[n],i2[n],i3[n]) for n in rng(nodes)]

	nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=node_size, alpha=nalpha)
	#labels = {n: G.nodes[n]['gene'] for n in G.nodes()}
	labels = {n:n for n in G.nodes()} # huh?
	#nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

	elist = sorted(list(G.edges()))
	nx.draw_networkx_edges(G, pos, arrows=True, edgelist=elist, alpha=ealpha) 


	if params['save_fig']:
		tstamp = timestamp()
		plt.savefig(params['output_path']+'/'+str(tstamp)+'_global.png')
	else:
		plt.show()
	plt.clf()
	plt.close()



def Rs_over_time(G,params,Rs):
	t = [i for i in rng(Rs)]
	plt.plot(t,Rs)
	plt.title("Local Modularity Score",fontsize=26)
	plt.xlabel("Iteration", fontsize=18)
	plt.ylabel("Local Modularity (R)",fontsize=18)

	if params['save_fig']:
		tstamp = timestamp()
		plt.savefig(params['output_path']+'/'+str(tstamp)+'_local_dR.png')
	else:
		plt.show()
	plt.clf()
	plt.close()


def drawG_by_comm_local(G,params):
	# color by seed/not_seed, alpha by rank

	assert(os.path.isdir(params['output_path']))

	node_size, nalpha, ealpha = 200, .5, .2 	
	

	plt.figure(figsize=(10,6))

	plt.title('Local Community Detection from Seed',fontsize=26)
	custom_lines = [Line2D([0], [0], color='green', lw=8,alpha=nalpha),
                Line2D([0], [0], color='purple', lw=8,alpha=nalpha),
                Line2D([0], [0], color='blue', lw=8,alpha=nalpha),
                Line2D([0], [0], color='grey', lw=8,alpha=nalpha)]

	ax = plt.gca()
	ax.legend(custom_lines, ['Seed', 'Inner Community','Border Community', 'Connected Outsiders'])

	pos = get_layout(params,G)

	nodes = list(G.nodes())
	colors, alphas = [], []
	dels = []
	for i in rng(nodes):
		if G.nodes[nodes[i]]['seed']: 
			colors+=['green']
		elif G.nodes[nodes[i]]['status'] == 'C': 
			colors+=['purple']
		elif G.nodes[nodes[i]]['status'] == 'B': 
			colors+=['blue']
		elif G.nodes[nodes[i]]['status'] == 'U': 
			colors+=['grey']
		else:
			colors+=['white']
			dels += [nodes[i]]
	for d in dels:
		nodes.remove(d)

	if params['draw_pagerank']:
		min_alpha = .1
		nalpha=[]
		for i in rng(nodes):
			if G.nodes[nodes[i]]['seed']: 
				G.nodes[nodes[i]]['alpha'] = 1
				nalpha += [1]
			else: 
				#G.nodes[nodes[i]]['alpha'] = G.nodes[nodes[i]]['rank']/max_rank #for the edges
				G.nodes[nodes[i]]['alpha'] = G.nodes[nodes[i]]['cardinal_rank'] #for the edges
				nalpha += [max(math.pow(G.nodes[nodes[i]]['cardinal_rank'],.3),min_alpha)]#/max_rank] #may not work..also gotta add LOG SCALING


	nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=node_size, alpha=nalpha)
	#labels = {n: G.nodes[n]['gene'] for n in G.nodes()}
	labels = {n:n for n in G.nodes()} # huh?
	nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')


	elist = sorted(list(G.edges()))
	nx.draw_networkx_edges(G, pos, arrows=True, edgelist=elist, alpha=ealpha) 


	if params['save_fig']:
		tstamp = timestamp()
		plt.savefig(params['output_path']+'/'+str(tstamp)+'_local.png')
	else:
		plt.show()
	plt.clf()
	plt.close()

def drawG_by_rank(G,params, topK):
	# color by seed/not_seed, alpha by rank

	assert(os.path.isdir(params['output_path']))


	plt.figure(figsize=(10,6))
	node_size, nalpha, ealpha = 800, .3, .3 	#node_alpha not curr used
	
	# positions for all nodes
	if params['draw_layout'] == 'circular':
		pos = nx.circular_layout(G)  	
	elif params['draw_layout'] == 'spring':
		pos = nx.spring_layout(G) 
	else: assert(False) #unknown draw layout

	nodes = list(G.nodes())
	colors, alphas = [], []

	max_rank = max(G.nodes[nodes[i]]['rank'] for i in rng(nodes))

	for i in rng(nodes):
		if G.nodes[nodes[i]]['seed']: 
			G.nodes[nodes[i]]['alpha'] = 1
			alphas += [1]
			colors+=['green']
		else: 
			colors+=['blue']
			#G.nodes[nodes[i]]['alpha'] = G.nodes[nodes[i]]['rank']/max_rank #for the edges
			G.nodes[nodes[i]]['alpha'] = G.nodes[nodes[i]]['cardinal_rank'] #for the edges
			alphas += [G.nodes[nodes[i]]['cardinal_rank']]#/max_rank] #may not work..also gotta add LOG SCALING

	nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=node_size, alpha=alphas)
	#labels = {n: G.nodes[n]['gene'] for n in G.nodes()}

	topK2 = topK.tolist()
	top = topK2 + params['seeds']
	labels = {n:n for n in top} #huh?
	nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='blue')

	#edge_alphas = []
	ecolors = []
	elist = sorted(list(G.edges()))
	for i in rng(elist):
		n1,n2 = elist[i][0],elist[i][1]
		#edge_alphas += [min(G.nodes[n1]['alpha'],G.nodes[n2]['alpha'])]
		ecolors += [(0,0,0,avg([G.nodes[n1]['alpha'],G.nodes[n2]['alpha']]))]


	nx.draw_networkx_edges(G, pos, arrows=True, edgelist=elist, edge_color=ecolors) #,alpha=edge_alphas)


	if params['save_fig']:
		now = datetime.now()
		curr_date = str(date.today()).strip('2020-')
		curr_time = str(datetime.now().strftime("%H-%M-%S"))
		plt.savefig(params['output_path']+'/'+curr_date+'_'+curr_time+'_rankedG.png')
	else:
		plt.show()	
	plt.clf()
	plt.close()