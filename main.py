import init, page_rank,plot,local_community, global_community
import numpy as np, networkx as nx

# for the lazy: cd Google*/*M2/*S2/Gra*/gra*/


def main():
	param_file = './params.txt'
	params = init.params(param_file)
	if params['run'] == 'GLOBAL':
		Dendo_global, C, node_C = global_C(params,plot_results=True)
		seed = params['seeds'][0] #assume just 1
		C_global = C[node_C[seed]]

		print("\nResulting size of global community = ",len(C_global))

	elif params['run'] == 'LOCAL':
		C, Dendo = local_C(params)
		print('Local Dendo = ', Dendo)
	elif params['run'] == 'LOCAL_PAGERANK':
		local_pagerank(params)
	elif params['run'] == 'BOTH':
		both(params)
	elif params['run'] == 'ALL':
		all(params)


def local_pagerank(params):
	G = init.build(params) 
	print("\n\nRUNNING PAGE RANK\n")
	top_k_nodes = page_rank.surf(G,params)
	#plot.drawG_by_rank(G,params,top_k_nodes)
	print('\nPagerank topk = ', top_k_nodes)

	print("\n\nLOCAL COMMUNITY WITHOUT PAGE RANK\n")
	params['use_pagerank']= False
	C_local, Dendo_local = local_C(params,plot_results=True,G=G)
	print('\n\nDendo Local = ', Dendo_local)

	print("\n\nRUNNING LOCAL COMMUNITY\n")
	params['use_pagerank']= True
	C_local2, Dendo_local2 = local_C(params,plot_results=True,G=G)
	print('\n\nDendo Local with pagerank = ', Dendo_local2)

	locals_shared = []
	local_pr_shared = []
	for c in C_local:
		if c in C_local2:
			locals_shared += [c]
		if c in top_k_nodes:
			local_pr_shared += [c]

	print("\nlocals shared:",locals_shared)
	print('\npr local shared:',local_pr_shared)


def all(params):
	G = init.build(params)  

	print("\n\nLOCAL COMMUNITY WITHOUT PAGE RANK\n")
	params['use_pagerank']= False
	C_local, Dendo_local = local_C(params,plot_results=False,G=G)
	print('\n\nDendo Local = ', Dendo_local)
			
	print("\n\nRUNNING PAGE RANK\n")
	top_k_nodes = page_rank.surf(G,params)
	#plot.drawG_by_rank(G,params,top_k_nodes)
	print('\nPagerank topk = ', top_k_nodes)

	if False:
		print("\n\nRUNNING LOCAL COMMUNITY\n")
		params['use_pagerank']= True
		C_local2, Dendo_local2 = local_C(params,plot_results=True,G=G)
		print('\n\nDendo Local = ', Dendo_local2)

		locals_shared = []
	local_pr_shared = []
	for c in C_local:
		#if c in C_local2:
		#	locals_shared += [c]
		if c in top_k_nodes:
			local_pr_shared += [c]

	print("\nRUNNING GLOBAL COMMUNITY\n")
	Dendo_global, C, node_C  = global_C(params,plot_results=False)

	seed = params['seeds'][0] #assume just 1
	C_global = C[node_C[seed]]
	print("\nResulting size of global community = ",len(C_global))

	plot.drawG_by_comm_global(G.subgraph(C_global),node_C,params)

	global_local_shared, all_shared, global_PR_shared = [],[],[]
	for c in C_global:
		if c in C_local:
			global_local_shared += [c]
		#if c in locals_shared:
		#	all_shared += [c]
		if c in top_k_nodes:
			global_PR_shared += [c]

	print("Dendo local:",Dendo_local)
	print("Dendo global:",Dendo_global)
	H = G.subgraph(global_local_shared)
	plot.drawG_by_dendos(H,Dendo_local, Dendo_global[seed], params)
	#print("\nlocals shared:",locals_shared)
	print('\npr local shared:',local_pr_shared)
	print("\nglobal and local shared:",global_local_shared)
	print('\nglobal PR shared:',global_PR_shared)
	#print('\nall shared:',all_shared)



def both(params):
	seed = params['seeds'][0] #assume just 1

	print("\nRUNNING GLOBAL COMMUNITY\n")
	Dendo_global, C, node_C  = global_C(params,plot_results=False)


	print("\n\nRUNNING LOCAL COMMUNITY\n")
	C_local, Dendo_local = local_C(params,plot_results=False)

	C_global = C[node_C[seed]]

	#print("Local C: ", C)
	#print("Global Dendo[C]", Dendo[seed],'\n\n')

	i=0
	print('\nMatching genes:')
	for c in C_local:
		if c in C_global:
			i+=1
			print(c)
	
	percent = round(i/min(len(C_local),len(C_global))*100,1)
	print("Matching percent = ",i/max(len(C_local),len(C_global)),'%')
	print('Total # matches =',i)
	print('lng globalC=',len(C_global),'lng localC=',len(C_local))

	print('\n\nDendo Global = ', Dendo_global)
	print('\n\nDendo Local = ', Dendo_local)

	print("\n\nDone.\n\n")


def local_C(params,plot_results=True, G=None):

	if G is None:
		G = init.build(params)  

	#page_rank.surf(G,params)
	#plot.drawG_by_rank(G,params)

	C, U, Rs, Dendo = local_community.clauset(G,params)
	if plot_results:
		subG = G.subgraph(C+U)
		plot.drawG_by_comm_local(subG,params)
		plot.Rs_over_time(subG,params,Rs)

	return C, Dendo

	


def global_C(params, plot_results=True):

	C, node_C, G, Dendo = global_community.clauset(params) 
	if plot_results: 
		plot.drawG_by_comm_global(G,node_C,params)

	return Dendo, C, node_C


main()