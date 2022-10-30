import numpy as np
import time
import sys
from objfunc import rastrigin
from objfunc import rosenbrock
from objfunc import schwefel
from timeit import default_timer as timer
import matplotlib
import matplotlib.pyplot as plt



class HC12():
    #getter a setter - method for entering and obtaining rank
    @property
    def dodParam(self):
        return self._dodParam

    @dodParam.setter
    def dodParam(self, dodParam):
        # if len(dodParam) == self.nParam:
        #     self._dodParam = np.array(dodParam)
        # else:
        self._dodParam = np.array([dodParam for _ in range(self.nParam)])
    
    def __init__(self, nParam, nBitParam, dodParam =None, float_type = np.float64):
        super(HC12,self).__init__()

        self.nParam = nParam
        self.nBitParam = np.array([nBitParam for _ in range(self.nParam)], dtype = np.uint16)
        self.dodParam = dodParam
        self.uint_type = np.uint16
        self.float_type = float_type
        self.total_bits = int(np.sum(self.nBitParam))

        # rows of matrix M0
        self.__M0_rows = 1
        self.__M1_rows = self.total_bits
        self.__M2_rows = self.total_bits*(self.total_bits-1)//2
        self.rows = self.__M0_rows + self.__M1_rows + self.__M2_rows
        # matrix K - kernel
        self.K = np.zeros((1,self.nParam), dtype = self.uint_type)
        # matrix M - matrix of numbers for masks
        self.M = np.zeros((self.rows,self.nParam), dtype = self.uint_type) 
        # matrix B - binary
        self.B = np.zeros((self.rows,self.nParam), dtype = self.uint_type)
        # matrix I - integer
        self.I = np.zeros((self.rows,self.nParam), dtype = self.uint_type)
        # matrix R - real value
        self.R = np.zeros((self.rows,self.nParam), dtype = self.float_type)
        # matrix F - functional value
        self.F = np.zeros((self.rows,self.nParam), dtype = self.float_type) 
        self.__init_M() 

    def __init_M(self):
        """Initializes matrix M.
        """
        bit_lookup = []
        for p in range(self.nParam):
            for b in range(self.nBitParam[p]):
                bit_lookup.append((p,b))

        for j in range(1, 1+self.__M1_rows):
            # bit shift
            p, bit = bit_lookup[j-1] 
            self.M[j,p] |= 1 << bit

        j = self.__M0_rows+ self.__M1_rows

        for bit in range(self.total_bits-1):
            # bit shift
            for bit2 in range (bit+1, self.total_bits):
                self.M[j,bit_lookup[bit][0]] |= 1 << bit_lookup[bit][1]
                self.M[j,bit_lookup[bit2][0]] |= 1 << bit_lookup[bit2][1]
                j += 1

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
        save_name = "HC12" + options["name"]+str(self.nParam)+"D"

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
        file.write("stopping criterions (max generation/running mean/best-worst):\n"+str(stopping[0])+"/"+str(stopping[1]))
        
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
        X1 = np.linspace(self.dodParam[0,0], self.dodParam[0,1], 100)     
        X2 = np.linspace(self.dodParam[0,0], self.dodParam[0,1], 100)     
        X1, X2 = np.meshgrid(X1, X2) 
        Y = func(X1, X2)
        x = list(zip(*xs))

        fig = plt.figure()
        ax = plt.axes(projection ="3d")  
        ax.plot_surface(X1, X2, Y, linewidth=0.1, antialiased=True, alpha = 0.3) 
        ax.scatter3D(best_x[0], best_x[1], best_val, color='r', marker='o', s=50, label = 'Optimal value')
        ax.set_title(save_name[4:] + ' - Optimal value')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('fval')
        ax.legend()
        plt.savefig(save_name+"OptimVal.png")
        plt.close()

        fig, ax = plt.subplots()
        CS = ax.contour(X1,X2,Y)
        ax.set_title(save_name[4:] + ' - Progress of solution')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.plot(x[0],x[1],linewidth=3,c="r")
        ax.scatter(best_x[0],best_x[1],marker="o",color="g")
        plt.grid(True)
        plt.savefig(save_name+"Progress.png")


    def hc12(self, func, times, max_iter,epsilonBw=None,epsilonRm=None,stallGener=sys.maxsize,maxTime=None,evaluateOptions=None):    
        """Runs the HC12 algorithm.
        Parameters:
        func (func): the function to be optimized
        times (int): number of runs
        max_iter (int): number of generations
        epsilonBw (float): parameter for stopping criterion Best-worst, when not given the criterion is not checked
		epsilonRm (float): parameter for stopping criterion Running mean, when not given the criterion is not checked
		stallGener (int): number of generations whose best result is checked for avarage change
		evaluateOptions (dict): dictionary with the options for evaluating results
        """
        dod = self.dodParam
        n_bit = self.nBitParam
        
        print(evaluateOptions["name"])
        print(self.nParam)

        x_out = np.zeros((times,self.nParam),dtype = self.float_type)
        fval = np.full(times, float('inf'))

        def interval_to_float(int_i, a, b, n_bits):
            return(b-a)/(2**n_bits-1)*int_i + a
        
        iterations = np.zeros((times, 1))
        winning_run = 0
        t = []

        best_scores = list()
        best_evals = list()
        best_times = list()
        generations = np.zeros((times,))
        running_mean = 0


        for run_i in range(times):
            best_scores.append(sys.maxsize)
            best_evals.append([sys.maxsize,sys.maxsize])
            best_times.append(timer())
            gen_best = sys.maxsize*np.ones(max_iter)
            start = time.time()
            reason = "Maximal iteration."

            # prepare K
            self.K[:] = [np.random.randint(0, 2**n_bit[i]) for i in range(self.nParam)]
            run_fval = float('inf')

            for iter_i in range(max_iter):
                np.bitwise_xor(self.K, self.M, out = self.B)
                np.bitwise_and(self.B, 1 << n_bit, out=self.I)
                for par in range(self.nParam):
                    for bit in range(n_bit[par], 0, -1):
                        self.I[:,par] |= np.bitwise_xor((self.I[:,par] & 1<<bit)>>1,self.B[:,par] & 1<<(bit-1))
                        self.R[:,par] = interval_to_float(self.I[:,par], dod[par,0], dod[par,1], n_bit[par])

                self.F = [func(c) for c in self.R]
                
                best_idx = np.argmin(self.F)
                worst_idx = np.argmax(self.F)
                
                run_fval = self.F[best_idx]
                if run_fval < best_scores[run_i]:
                    best_scores[run_i] = run_fval
                    best_evals[run_i] = self.R[best_idx,:]
                    gen_best[iter_i] = run_fval
                    generations[run_i] = iter_i
                gen_best[iter_i] = run_fval
                

                # check for termination criterion
                if (iter_i >= stallGener) & (epsilonRm != None):
                    if self.running_mean_criterion(gen_best[iter_i],gen_best[iter_i-stallGener:iter_i-1],epsilonRm):
                        reason = "Running Mean criterion."
                        running_mean += 1
                        break

                self.K = self.B[best_idx, :]

            iterations[run_i] = iter_i
            x_out[run_i,:] = self.R[best_idx,:]

            if run_fval < min(fval):
                winning_run = run_i

            fval[run_i] = run_fval
            t.append(time.time() - start)

            print(f'x = {x_out[run_i]}, fval = {fval[run_i]}')

        if evaluateOptions != None:
            max_gen = times-running_mean
            stops = [max_gen, running_mean]
            self.evaluate_results(evals=x_out,scores=best_scores,times=best_times,generations=generations,options=evaluateOptions,nRuns=run_i,stopping=stops)
                
        return x_out, fval, t



def eval_rastrigin(X1, X2):
    return (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (X2**2 - 10 * np.cos(2 * np.pi * X2)) + 20

def eval_schwefel(X1, X2):
    return 418.9829*2 - (X1*np.sin(np.sqrt(np.absolute(X1)))+X2*np.sin(np.sqrt(np.absolute(X2))))

def eval_rosenbrock(X1, X2):
    return  100*(X2-X1**2)**2+(X1-1)**2


def main():
    hc12_instance = HC12(nParam=2 ,nBitParam = 8, dodParam=[-2.048, 2.048]) # rosenbrock
    x, fx, t = hc12_instance.hc12(func=rosenbrock,times=1,max_iter=50,epsilonRm=0.001,stallGener=10,evaluateOptions={"name":"Rosenbrock","func":eval_rosenbrock})

    hc12_instance = HC12(nParam=2 ,nBitParam = 8, dodParam=[-5.12,5.12]) # rastrigin
    x, fx, t = hc12_instance.hc12(func=rastrigin,times=5,max_iter=100,epsilonRm=0.001,stallGener=10,evaluateOptions={"name":"Rastrigin","func":eval_rastrigin})

    hc12_instance = HC12(nParam=2 ,nBitParam = 8, dodParam=[-500, 500]) # schwefel
    x, fx, t = hc12_instance.hc12(func=schwefel,times=3,max_iter=100,epsilonRm=0.001,stallGener=10,evaluateOptions={"name":"Schwefel","func":eval_schwefel})


if __name__ == "__main__":
    main()