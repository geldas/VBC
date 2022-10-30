from numpy.random import randint
from numpy.random import rand
import numpy as np
import sys
from objfunc import rastrigin
from objfunc import rosenbrock
from objfunc import schwefel
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt


class GA:
	def __init__(self,NP,pS,pC,pM,nBitParam,nParam,dodParam):
		self.NP = NP
		self.pS = pS
		self.pC = pC
		self.pM = pM
		self.nBitParam = nBitParam
		self.nParam = nParam
		self.dodParam = dodParam
		self.k = round(pS*NP)
		self.xs = list()
		self.fvals = list()

	def run(self,nRuns,maxGener,obj_func,epsilonBw=None,epsilonRm=None,stallGener=sys.maxsize,maxTime=None,evaluateOptions=None):
		"""Call this function to start the genetic algorithm to find the minimum of the continuous function.
		Parameters:
		nRuns (int): number of restarts of the algorithm
		maxGener (int): number of maximum generation in each run
		obj_func (func): objective continous function that returns fval
		epsilonBw (float): parameter for stopping criterion Best-worst, when not given the criterion is not checked
		epsilonRm (float): parameter for stopping criterion Running mean, when not given the criterion is not checked
		stallGener (int): number of generations whose best result is checked for avarage change
		evaluateOptions (dict): dictionary with the options for evaluating results
		"""
		func = obj_func
		best_scores = list()
		best_evals = list()
		best_decoded = list()
		best_times = list()
		generations = np.zeros((nRuns,))
		running_mean = 0
		best_worst = 0
		# do nRuns of restarts to find minimum of the function
		for run in range(nRuns):
			population = [randint(0,2,self.nBitParam*self.nParam).tolist() for _ in range(self.NP)]
			best_scores.append(sys.maxsize)
			best_evals.append([sys.maxsize,sys.maxsize])
			best_decoded.append(sys.maxsize)
			best_times.append(timer())
			gen_best = sys.maxsize*np.ones(maxGener)
			gen_evals = list()
			reason = "Maximal number of generations."
			for gen in range(maxGener):
				#print(gen)
				# decode the value from binary to real
				decoded = np.array([self.decode(p) for p in population])
				# score the fitness function
				scores = [func(val) for val in decoded]
				best_scores[run] = scores[0]
				#gen_worst = -sys.maxsize
				# check for best score in nRuns and in generations in actual run
				# for j in range(len(scores)):
				# 	if scores[j] < best_scores[run]:
				# 		best_scores[run] = scores[j]
				# 		best_evals[run] = population[j]
				# 		best_decoded[run] = decoded[j]
				# 		gen_best[gen] = scores[j]
				# 		generations[run] = gen
				# 	elif scores[j] < gen_best[gen]:
				# 		gen_best[gen] = scores[j]
				# 	if scores[j] > gen_worst:
				# 		gen_worst = scores[j]
						#print(">%d, new best f(%s) = %f" % (gen,  decoded[j], scores[j]))
				# select the parents (Elite tournament selection)
				selected = [self.ets(population, scores,k) for k in range(self.NP)]
				# create the next generation (crossover, mutation)
				children = []
				for j in range(0, self.NP, 2):
					# get selected parents in pairs
					p1, p2 = selected[j], selected[j+1]
					# crossover and mutation
					for c in self.crossover(p1, p2):
						# mutation
						self.mutation(c)
						# store for next generation
						children.append(c)
				population = children
				
				idx = np.argmin(scores)
				gen_best[gen] = scores[idx]
				gen_evals.append(decoded[idx])
				gen_worst = np.max(scores)
				
				self.xs.append(decoded)
				self.fvals.append(scores)

				# check for termination criterions
				if (gen >= stallGener) & (epsilonRm != None):
					if self.running_mean_criterion(gen_best[gen],gen_best[gen-stallGener:gen-1],epsilonRm):
						reason = "Running Mean criterion."
						running_mean += 1
						break
				if epsilonBw !=None:
					if self.best_worst_criterion(gen_worst,gen_best[gen],epsilonBw):
						reason = "Best-Worst criterion."
						best_worst += 1
						break
				# if maxTime != None:
				# 	if timer()-best_times[run] > maxTime:
				# 		reason = "Maximal Time Budget."
				#rel_change = np.average(gen_best[max(0,gen-stallGener):gen])
			generations[run] = gen+1
			idx = np.argmin(gen_best)
			best_scores[run] = gen_best[idx]
			best_decoded[run] = gen_evals[idx]
			# print the best value for actual run
			best_times[run] = timer()-best_times[run]
			print("Run: %d> Termination reason: %s> best f(%s) = %f" % (run,reason,best_decoded[run], best_scores[run]))

		print("DONE!!!")
		# evaluate the results
		if evaluateOptions != None:
			max_gen = nRuns-(running_mean + best_worst)
			stops = [max_gen, running_mean, best_worst]
			self.evaluate_results(evals=best_decoded,scores=best_scores,times=best_times,generations=generations,options=evaluateOptions,nRuns=nRuns,stopping=stops)

	def best_worst_criterion(self,worst,best,epsilon):
		"""Criterion that checks whether the difference between best and worst score in actual generation is lower then epsilon
		Parameters:
		worst (float): the worst score in generation
		best (float): the best score in generation
		epsilon (float): the epsilon parameter
		"""
		if abs(worst-best)<epsilon:
			return True
		else:
			return False

	def running_mean_criterion(self,best_act,last_gener,epsilon):
		"""Criterion that checks whether the difference between best score in actual generation and mean value in last stall 
		generations is lower then epsilon.
		Parameters:
		best (float): the best score in generation
		last_gener (list): scores of the last generations
		epsilon (float): the epsilon parameter
		"""
		if abs(best_act-np.mean(last_gener)) < epsilon:
			return True
		else:
			return False

	def evaluate_results(self,evals,scores,times,generations,options,nRuns,stopping):
		"""Evaluate the results, create graphs and compute statistics for best run, computation time and number of generations.
		Parameters:
		evals (list): list of the x1..xn best parameters for each run
		scores (list): list of the best fvals for each run
		times (list): list of the times of each run
		generations (list): list of number of number of generations of each run
		options (dict): options for evaluating
		nRuns (int): number of restarts of the algorithm
		stopping (list): list containing the number of number of stopping criteria for nRuns
		"""
		min_i = np.argmin(scores)
		max_i = np.argmax(scores)
		min = scores[min_i]
		max = scores[max_i]
		gens_min = np.min(generations)+1
		gens_max = np.max(generations)+1
		gens_mean = np.mean(generations)+1
		min_time = times[min_i]
		min_evals = evals[min_i]
		mean = np.mean(scores)
		time_max = np.max(times)
		time_min = np.min(times)
		time_mean = np.mean(times)

		print(f'Statistics fval of RUNs: min fval = {min}, max fval = {max}, mean fval = {min}')
		print(f'Statistics generation number of RUN: min gen_n = {gens_min}, max gen_n = {gens_max}, mean gens_n = {gens_mean}')
		print(f'Statistics computation time of RUN: min time = {time_min}, max time = {time_max}, mean time = {time_mean}')
		print(f'Best parameters: {min_evals}')

		name = options["name"] + " " + str(self.nParam) + "D"
		save_name = options["name"]+str(self.nParam)+"D"

		fig,ax = plt.subplots(figsize=(12,12))
		ax.plot(scores,marker='o',linestyle='None')
		ax.hlines(y=mean,xmin=0,xmax=nRuns,linewidth=2,color='r',label="mean")
		ax.set_xlabel("run")
		ax.set_ylabel("fval")
		ax.set_title(name)
		ax.plot()
		plt.savefig(save_name+".png")
		plt.close()

		fig,ax = plt.subplots(figsize=(6,6))
		ax.plot(generations,marker='o',linestyle='None')
		ax.hlines(y=gens_mean,xmin=0,xmax=nRuns,linewidth=2,color='r',label="mean")
		ax.set_xlabel("run")
		ax.set_ylabel("#Generations")
		ax.set_title(name+" - #Generations")
		ax.plot()
		plt.savefig(save_name+"evals.png")
		plt.close()

		file = open(save_name+".txt","w")
		file.write(name+"\n")
		file.write("fval max:\n"+str(max)+"\n")
		file.write("fval min:\n"+str(min)+"\n")
		file.write("fval mean:\n"+str(mean)+"\n")
		file.write("best generations:\n"+str(gens_min)+"\n")
		file.write("best time:\n"+str(min_time)+"\n")
		file.write("best evals:\n"+str(min_evals)+"\n")
		file.write("time max:\n"+str(time_max)+"\n")
		file.write("time min:\n"+str(time_min)+"\n")
		file.write("time mean:\n"+str(time_mean)+"\n")
		file.write("generations max:\n"+str(gens_max)+"\n")
		file.write("generations min:\n"+str(gens_min)+"\n")
		file.write("generations mean:\n"+str(gens_mean)+"\n")
		file.write("stopping criterions (max generation/running mean/best-worst):\n"+str(stopping[0])+"/"+str(stopping[1])+"/"+str(stopping[2]))
		#fig = plt.figure(figsize=(12, 12))
		# ax = fig.add_subplot(111, projection='3d')
		# Z = func(X,Y,self.nParam)
		# surf = ax.plot_surface(X, Y, Z)
		# ax.scatter(min_evals[0],min_evals[1],min,s=30,c='red')
		# ax.set_xlim((self.dodParam[0],self.dodParam[1]))
		# ax.set_ylim((self.dodParam[0],self.dodParam[1]))
		#plt.show()

		if self.nParam == 2:
			self.plot_graph(min_evals, min, evals, save_name, options["func"])

	def plot_graph(self, best_x, best_val, xs, save_name, func):
		"""Saves graphs of the best result and the progress of finding optimum.
		Parameters:
		best_x (list): list of the coordinates of the minimum
		best_val (float): value of global minimum
		xs (list): list of the coordinates of values in each generation during the best run
		save_name (string): the name of the file
		"""
		
		X1 = np.linspace(self.dodParam[0], self.dodParam[1], 100)     
		X2 = np.linspace(self.dodParam[0], self.dodParam[1], 100)     
		X1, X2 = np.meshgrid(X1, X2) 
		Y = func(X1, X2)

		x = list(zip(*xs))

		fig = plt.figure()
		ax = plt.axes(projection ="3d")  
		ax.plot_surface(X1, X2, Y, linewidth=0.1, antialiased=True, alpha = 0.3) 
		ax.scatter3D(best_x[0], best_x[1], best_val, color='r', marker='o', s=50, label = 'Optimal value')
		ax.set_title(save_name + ' - Optimal value')
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
		ax.set_zlabel('fval')
		ax.legend()
		plt.savefig(save_name+"OptimVal.png")
		plt.close()

		fig, ax = plt.subplots()
		CS = ax.contour(X1,X2,Y)
		ax.set_title(save_name + ' - Progress of solution')
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
		ax.plot(x[0],x[1],linewidth=3,c="r")
		ax.scatter(best_x[0],best_x[1],marker="o",color="g")
		plt.grid(True)
		plt.savefig(save_name+"Progress.png")


	def selection(self, pop, scores):
		"""Tournament selection.
		Parameters:
		pop (list): list of the population
		scores (list): list of the scores of the fitness function
		"""
		selection_ix = randint(len(pop))
		for ix in randint(0, len(pop), self.k-1):
			if scores[ix] < scores[selection_ix]:
				selection_ix = ix
		return pop[selection_ix]

	def ets(self,pop,scores,ai):
		"""Elite tournament selection. Choose the first element as actual index in population and then k-1 random individuals.
		Parameters:
		pop (list): list of the population
		scores (list): list of the scores of the fitness function
		ai (int): index of actual individual
		"""
		selection_ix = ai
		for ix in randint(0,len(pop), self.k-1):
			if scores[ix] < scores[selection_ix]:
				selection_ix = ix
		return pop[selection_ix]
 
	def crossover(self, p1, p2):
		"""1-point crossover. From two parents create two children if the crossover rate is hihger than random number.
		Parameters:
		p1()
		scores (list): list of the scores of the fitness function
		ai (int): index of actual individual
		"""
		c1, c2 = p1.copy(), p2.copy()
		if rand() < self.pC:
			pt = randint(1, len(p1)-2)
			c1 = p1[:pt] + p2[pt:]
			c2 = p2[:pt] + p1[pt:]
		return [c1, c2]
	
	def mutation(self, bitstring):
		"""Mutation operator. Perform mutation if the random number is lower than mutation rate.
		Parameters:
		bitstring (list): list of binary values representing the individual
		"""
		for i in range(len(bitstring)):
			if rand() < self.pM:
				bitstring[i] = 1 - bitstring[i]		

	def decode(self,encoded):
		"""Decode the list representing the individual in binary values to real number.
		Parameters:
		encoded (list): list of binary values representing the individual.
		"""
		decoded = []
		largest = 2**self.nBitParam		
		for i in range(self.nParam):
			start = i*self.nBitParam
			end = start+self.nBitParam
			integer = sum(val*(2**idx) for idx, val in enumerate(reversed(encoded[start:end])))
			value = self.dodParam[0] + (integer/largest) * (self.dodParam[1] - self.dodParam[0])
			decoded.append(value)
		return decoded


def eval_rastrigin(X1, X2):
    return (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (X2**2 - 10 * np.cos(2 * np.pi * X2)) + 20

def eval_schwefel(X1, X2):
    return 418.9829*2 - (X1*np.sin(np.sqrt(np.absolute(X1)))+X2*np.sin(np.sqrt(np.absolute(X2))))

def eval_rosenbrock(X1, X2):
    return  100*(X2-X1**2)**2+(X1-1)**2


def main():
	n_params = 2
	n_iter = 100
	n_bits = 15
	n_pop = 100
	r_cross = 0.9
	r_mut = 1.0 / (float(n_bits) * n_params)

	p_s = 0.1

	bounds = [-5.12, 5.12] #rastrigin
	options = {
		"name":"Rastrigin",
		"func":eval_rastrigin
	}

	ga = GA(n_pop,p_s,r_cross,r_mut,n_bits,n_params,bounds)
	ga.run(nRuns=10,maxGener=100,obj_func=rastrigin,epsilonBw=2,epsilonRm=0.00001,stallGener=50,evaluateOptions=options)

	
	options = {
		"name":"Rosenbrock",
		"func":eval_rosenbrock
	}
	bounds = [-2.048, 2.048] #rosenbrock
	ga.run(nRuns=10,maxGener=100,obj_func=rosenbrock,epsilonBw=2,epsilonRm=0.00001,stallGener=50,evaluateOptions=options)

	
	bounds = [-500, 500] #schwefel
	options = {
		"name":"Schwefel",
		"func":eval_schwefel
	}
	
	ga.run(nRuns=10,maxGener=100,obj_func=schwefel,epsilonBw=2,epsilonRm=0.00001,stallGener=50,evaluateOptions=options)



if __name__ == "__main__":
    main()