import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from objfunc import rastrigin
from objfunc import rosenbrock
from objfunc import schwefel
import time
from timeit import default_timer as timer

class GSO:
    def __init__(self, n_p, sensor_range, range_min, gamma, rho, step, beta, neighbour_n, dod_param:np.array):
        self.npop = n_p
        self.sensor_range = sensor_range
        self.range_min = range_min
        self.gamma = gamma
        self.rho = rho
        self.step = step
        self.beta = beta
        self.neighbour_n = neighbour_n
        self.decision_range = self.sensor_range*np.ones((self.npop,1))
        self.luciferin = 5*np.ones((self.npop,))
        self._dod_param = dod_param
        #self.glowworms = np.zeros((self.npop,self.dim))
        self.dim = len(dod_param)
        self.glowworms = np.random.uniform(self._dod_param[:,0],self._dod_param[:,1],[self.npop,self.dim])
        # self.glowworms = np.array([[-1.5,-1.5], [-2.5,-2.5],[-1.7,-1.7],[-1.9,-1.9],[-2.3,-2.3],[-2.6,-2.6],[-1.1,-1.1],\
        #     [-2.4,-2.4],[-1.45,-1.45],[-3.8,-3.8]])
        self.fx = np.zeros(self.npop)
        

    def run(self,nRuns,func,eval_options=None):
        """Run the algorithm for nRuns and find optims on given func.
        Parameters:
        nRuns (int): number of restarts
        func (func): objective function
        """
        run_i = 0
        times = list()
        self.glowworm_history = list()
        self.fx_history = list()
        self.glowworm_history.append(self.glowworms)
        
        while run_i<nRuns:
            times.append(timer())
            self.luciferin_update(func)
            self.glowworm_history.append(np.copy(self.glowworms))
            self.fx_history.append(np.copy(self.fx))

            t1 = timer()
            self.movement()
            times[run_i] = timer()-times[run_i]

            run_i += 1
        if eval_options != None:
            print("Optimization done! Please refer to the graph.")
            print(f'Min time of run = {np.min(times)}, max time of run = {np.max(times)}')
            print(f'Mean time = {np.mean(times)}, Computation time = {np.sum(times)}')
            fc = eval_options["func"]
            name = eval_options["name"]
            glowworms = list(zip(*self.glowworm_history))
            X1 = np.linspace(self._dod_param[0][0], self._dod_param[0][1], 100)     
            X2 = np.linspace(self._dod_param[0][0], self._dod_param[0][1], 100)     
            X1, X2 = np.meshgrid(X1, X2) 
            Y = fc(X1, X2)
            fig, ax = plt.subplots()
            ax.set_title(name + ' - Solution')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.contour(X1,X2,Y)
            ax.plot()
            ax.scatter(self.glowworms[:,0],self.glowworms[:,1],marker="x",color="g")
            plt.savefig(name+"_solution.png")
            plt.show()

            fig, ax = plt.subplots()
            for i in range(len(glowworms)):
                gw = list(zip(*glowworms[i]))
                ax.plot(gw[0], gw[1])
            ax.set_title(name + ' - Progress')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            
            plt.scatter(self.glowworms[:,0],self.glowworms[:,1],marker="x",s=10)
            plt.savefig(name+"_progress.png")
            plt.grid(True)
            plt.show()


    def luciferin_update(self,func):
        """Update luciferin level.
        Parameters:
        func (func): function for scoring the glowworms and updating their luciferin level
        """
        n = np.size(self.glowworms,1)
        self.fx = np.array([-func(val) for val in self.glowworms])
        self.luciferin = (1-self.rho)*self.luciferin+self.gamma*self.fx

    def find_neighbour(self,agent_i):
        """Find the neighbours of given glowworm.
        Parameters:
        agent_i (int): index of actual agent
        """
        neighbours = []
        n_a = 0
        for i in range(self.npop):
            if i != agent_i:
                if self.luciferin[agent_i] < self.luciferin[i]:
                    square = np.power((self.glowworms[agent_i]-self.glowworms[i]),2)
                    sum_square = np.sum(square)
                    dist = np.sqrt(sum_square)
                    if dist<self.decision_range[agent_i]:
                        neighbours.append(i)
                        n_a += 1
        return neighbours,n_a

    def movement(self):
        """Move the glowwworms.
        """
        agent = []
        n_a = np.zeros((self.npop))
        for i in range(self.npop):
            neighbours,n_a[i] = self.find_neighbour(i)
            p = self.find_probabilities(i,neighbours)
            agent.append(self.glowworm_selection(p,neighbours))
        for i in range(self.npop):
            if (agent[i] > -1) & (self.luciferin[i]<self.luciferin[agent[i]]):
                self.move_glowworm(i,agent[i])
            updated_decision_range = min(self.sensor_range,self.decision_range[i] + self.beta*(self.neighbour_n-n_a[i])) # upravit
            self.decision_range[i] = max(0,updated_decision_range)
           
    def find_probabilities(self,agent_i,neighbours):
        """Find probabilities to of actual glowworm to move towards its neighbour its neighbours according to their luciferin level.
        Parameters:
        agent_i (int): the index of actual agent
        neighbours (list): list of neighbours in agents range
        """
        luciferin_sum = 0
        p = []
        for n in neighbours:
            luciferin_sum += self.luciferin[agent_i] - self.luciferin[n]
        for n in neighbours:
            p_n = (self.luciferin[agent_i] - self.luciferin[n])/luciferin_sum
            p.append(p_n)
        return p

    def glowworm_selection(self,prob,neighbours):
        """Select glowworm for actual agent to move closer to it.
        Parameters:
        prob (list): list of probabilities
        neighbours (list): list of neighbours
        """
        bound_lower = 0
        bound_upper = 0
        toss = np.random.rand()
        #toss = 0.35
        ag = -5
        for i in range(len(prob)):
            bound_lower = bound_upper
            bound_upper += prob[i]
            if (toss>bound_lower) & (toss<bound_upper):
                ag = neighbours[i]
                break
        return ag
        
    def move_glowworm(self, glowworm, agent):
        """Move the glowworm closer to selected agent.
        Parameters:
        glowworm (int): index of actual agent
        agent (int): index of neighbour to which the glowworms should move closer
        """
        square = np.power(self.glowworms[glowworm]-self.glowworms[agent],2)
        sum_square = np.sum(square)
        dist = np.sqrt(sum_square)
        path = (self.glowworms[agent]-self.glowworms[glowworm])/dist

        moved = self.glowworms[glowworm] + self.step*path
        flag = 0
        for i in range(self.dim):
            if ((moved[i]) < self._dod_param[i][0]) | (moved[i] > self._dod_param[i][1]):
                flag = 1
                break

        if flag == 0:
            self.glowworms[glowworm] = np.copy(moved)


def eval_rastrigin(X1, X2):
    return (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (X2**2 - 10 * np.cos(2 * np.pi * X2)) + 20

def eval_schwefel(X1, X2):
    return 418.9829*2 - (X1*np.sin(np.sqrt(np.absolute(X1)))+X2*np.sin(np.sqrt(np.absolute(X2))))

def eval_rosenbrock(X1, X2):
    return  100*(X2-X1**2)**2+(X1-1)**2


def main():
    a = 1
    for i in range(a):
        gso_instance = GSO(100,3,0,0.6,0.4,0.03,0.08,5,dod_param=np.array([(-2.048, 2.048),(-2.048, 2.048)])) # rosenbrock
        gso_instance.run(10,func=rosenbrock,eval_options={"name":"Rosenbrock","func":eval_rosenbrock})
    # for i in range(a):
    #     gso_instance = GSO(100,400,0,0.6,0.4,3,0.08,5,dod_param=np.array([(-500,500),(-500,500)])) # schwefel
    #     gso_instance.run(1000,func=schwefel,eval_options={"name":"Schwefel","func":eval_schwefel}) 
    # for i in range(a):
    #     a = GSO(500,4,0,0.6,0.4,0.1,0.08,5,dod_param=np.array([(-5.12,5.12),(-5.12,5.12)])) # rastrigin
    #     a.run(100,func=rastrigin,eval_options={"name":"Rastrigin","func":eval_rastrigin})

if __name__ == "__main__":
    main()

    