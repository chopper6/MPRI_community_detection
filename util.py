import os, math
import numpy as np
from datetime import datetime, date

# JUST SOME GENERALLY USEFUL FUNCTIONS

def dict_argmax(dQ,i):
	keys = list(dQ[i].keys())
	j = np.array([dQ[i][k] for k in keys]).argmax()
	return keys[j][1]

def dict_max(dQ,i):
	a = [dQ[i][k] for k in dQ[i].keys()]
	if a != []:
		return max(a)
	else:
		return 0



def bool(x):
	if x in [0,'0','False',False,'false','unuh','noway','gtfofh']:
		return False
	elif x in [1,'1','True',True,'true','yeaya','fosho','nodoubt']:
		return True
	else: assert(False) #unknown value

def timestamp():
	now = datetime.now()
	curr_date = str(date.today()).strip('2020-')
	curr_time = str(datetime.now().strftime("%H-%M-%S"))
	tstamp = curr_date+'_'+curr_time
	return tstamp

def removes(chars,string):
	for char in chars:
		string = string.replace(char,'')
	return string

def rng(x):
    return range(len(x))

def avg(x):
	return sum(x)/len(x)

def var(x):
	the_avg = avg(x)
	var = avg([math.pow(the_avg-x[i],2) for i in rng(x)])
	return math.pow(var,1/2)

def L1(x):
	the_avg = avg(x)
	L1 = avg([abs(the_avg-x[i]) for i in rng(x)])
	return L1

def sqavg(x):
	return sum([math.pow(x[i],2) for i in range(len(x))])/len(x)

def powavg(x,power):
	a = sum([math.pow(x[i],power) for i in range(len(x))])/len(x)
	return math.pow(a,1/power)


def check_build_dir(dirr):
    if not os.path.exists(dirr):
        print("\nCreating new directory for output at: " + str(dirr) + '\n')
        os.makedirs(dirr)

def sort_a_by_b(A,B,reverse=False):
	#sorts by min
	a,b=A.copy(), B.copy() #otherwise will change in place
	not_done, iters = True,0
	while not_done:
		not_done = False
		for i in range(len(b)-1):
			if b[i] < b[i+1]:
				b = swap(b,i, i+1)
				a = swap(a,i, i+1)
				not_done=True
		iters += 1
		if iters > 10000: assert(False)
	if reverse:
		a.reverse(), b.reverse()
	return a,b

def swap(array,i,j):
	z=array[i]
	array[i] = array[j]
	array[j] = z
	return array

def safe_div_array(A,B):
	# a is numerator, b is divisor
	assert(len(A) == len(B))
	z=[]
	for i in rng(A):
		if B[i] == 0: z+=[0]
		else: z+=[A[i]/B[i]]
	return z