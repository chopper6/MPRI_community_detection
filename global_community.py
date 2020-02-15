import init
from util import *
import numpy as npl, time


SUPERSLOWCHECK = False

def assign_to_nodes(C):
	node_communities = {}
	for c in C.keys():
		for n in C[c]:
			node_communities[n]=c
	return node_communities


def clauset(params):

	t0 = time.time()
	#dQ, H, a, A, nodeMap, G = init.import_sparse(params, filename)
	dQ, H, a, A, nodeMap, G = init.build_global(params)
	num_nodes = len(nodeMap)
	reverse_nodeMap = {nodeMap[key]:key for key in nodeMap.keys()}
	Communities = {nodeMap[key]:[key] for key in nodeMap.keys()}

	Dendo = {key:[[key]] for key in params['seeds']} 
	has_seed = [0 for i in range(num_nodes)]

	for s in params['seeds']:
		has_seed[nodeMap[s]] = 1

	itr=0
	Q = 0
	picked = []
	t1 = time.time()
	if params['timeit']:
		print('Time to init algo = ',t1-t0,'secs.')
		tt,ta,tb,tc,td,te,tz = 0,0,0,0,0,0,0

	while len(Communities) > params['num_communities']:

		if params['debug'] and SUPERSLOWCHECK:
			for i in range(num_nodes):
				for j in range(num_nodes):
					assert(dQ[i,j]==dQ[j,i])

		if params['global_seed']:
			s,size = params['seeds'][0],0
			for c in Communities.keys():
				if has_seed[c]:
					size = max(len(Communities[c]),size)
			if size >= params['community_size']:
				break

		t8 = time.time()

		i = np.argmax(H)

		if params['timeit']:
			t7 = time.time()
			tt += t7-t8

		j = dict_argmax(dQ, i)

		# mess of prev attempts:
		#j = dQ[i].argmax()
		#j = np.array([dQ[i,k] for k in range(num_nodes)]).argmax()
		#j = np.array([dQ[i][k] for k in dQ[i].keys()]).argmax()

		#if params['debug']:
		#	print('\nvals at the max:',dQ[i,j],dQ[j,i])
		
		if params['timeit']:
			t0 = time.time()
			tz += t0-t7

		Q += dQ[i,j]

		if params['debug']:
			assert(i not in picked and j not in picked)
			picked += [i]

		if params['timeit']:
			t1 = time.time()
			ta += t1-t0

		i_name, j_name, seed = reverse_nodeMap[i], reverse_nodeMap[j], params['seeds'][0]
		if has_seed[i]==1:
			Dendo[seed] += [Communities[j].copy()]
			has_seed[j]=1
		elif has_seed[j]==1:
			Dendo[seed] += [Communities[i].copy()]
			has_seed[i]=1		

		Communities[j] += Communities[i].copy()

		del Communities[i]

		if params['timeit']:
			t2 = time.time()
			tb += t2-t1

		ki = dQ[i].nonzero()[1] 
		ki = [k for k in ki if k not in [i,j]]
		kj = dQ[j].nonzero()[1]
		kj = [k for k in kj if k not in [i,j]] #[kj != j]

		jdels = [i]
		reH = [i]

		if params['timeit']:
			t3 = time.time()
			tc += t3-t2

		for k in ki:
			if k in kj:
				dQ[j,k] += dQ[i,k]
				dQ[k,j] += dQ[i,k]
				jdels += [k]
			else:
				if params['debug']:
					assert(dQ[j,k]==0)
				change = dQ[i,k] - 2*a[j]*a[i]
				if dQ[j,k]==H[k] and change < 0:
					reH += [k]
				dQ[j,k] = change
				dQ[k,j] = change

			dQ[i,k] = dQ[k,i] = 0 #faster than del'g the row and col

		kj = [k for k in kj if k not in jdels]
		for k in kj: #have rm'd eles that are in ki too
			if dQ[j,k]==H[k]:
				reH += [k]
			if params['debug']:
				assert(dQ[i,k]==0)
			dQ[j,k] -= 2*a[j]*a[i] 
			dQ[k,j] -= 2*a[j]*a[i] 

		dQ[i,j] = dQ[j,i] = 0
		dQ[i,i] = 0

		if params['timeit']:
			t4 = time.time()
			td += t4-t3

		for k in reH:
			H[k] = dict_max(dQ,k)

		if params['debug']:
			assert(dQ[i,j] == dQ[j,i] == 0)

			ki = dQ[i].nonzero()[1] 
			for k in ki:
				if not dQ[i,k]==0:
					print('err:',dQ[i,k],k)
				assert(dQ[i,k]==0)
				assert(dQ[k,i]==0)
			for k in range(num_nodes):
				assert(dQ[k,k]==0)

			assert(H[i]==0)

		itr+=1
		if params['verbose']:
			if itr%10==0:
				print('\r# of Communities =',len(Communities),'Q=',Q,'\t\t\t',end='')
		
		if params['timeit']:
			t5 = time.time()
			te += t5-t4

	print("\nTimes for global clauset algo: ",tt,tz,ta,tb,tc,td,te)
	return Communities, assign_to_nodes(Communities), G, Dendo