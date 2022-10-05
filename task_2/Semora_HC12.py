import numpy as np
import time
import statistics 
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
from objfunc import rastrigin
from objfunc import schwefel
from objfunc import rosenbrock

class HC12():
    #getter a setter - method for entering and obtaining rank
    @property
    def dod_param(self):
        return self._dod_param

    @dod_param.setter
    def dod_param(self, dod_param):
        # if len(dod_param) == self.n_param:
        #     self._dod_param = np.array(dod_param)
        # else:
        self._dod_param = np.array([dod_param for _ in range(self.n_param)])
    
    def __init__(self, n_param, n_bit_param, dod_param =None, float_type = np.float64):
        super(HC12,self).__init__()

        self.n_param = n_param
        self.n_bit_param = np.array([n_bit_param for _ in range(self.n_param)], dtype = np.uint16)
        self.dod_param = dod_param
        self.uint_type = np.uint32
        self.float_type = float_type
        self.total_bits = int(np.sum(self.n_bit_param))

        # rows of matrix M0
        self.__M0_rows = 1
        # rows of matrix M0
        self.__M1_rows = self.total_bits
        # rows of matrix M0
        self.__M2_rows = self.total_bits*(self.total_bits-1)//2
        # total number of rows
        self.rows = self.__M0_rows + self.__M1_rows + self.__M2_rows
        # matrix K - kernel
        self.K = np.zeros((1,self.n_param), dtype = self.uint_type)
        # matrix M - matrix of numbers for masks
        self.M = np.zeros((self.rows,self.n_param), dtype = self.uint_type) 
        # matrix B - binary
        self.B = np.zeros((self.rows,self.n_param), dtype = self.uint_type)
        # matrix I - integer
        self.I = np.zeros((self.rows,self.n_param), dtype = self.uint_type)
        # matrix R - real value
        self.R = np.zeros((self.rows,self.n_param), dtype = self.float_type)
        # matrix F - functional value
        self.F = np.zeros((self.rows,self.n_param), dtype = self.float_type) 
        self.__init_M() 

    def __init_M(self):
        #matrix M
        bit_lookup = []
        for p in range(self.n_param):
            for b in range(self.n_bit_param[p]):
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

    def hc12(self, func, times, max_iter):    
        # func = self.func
        # times = self.nRuns 
        # max_iter = self.maxGener

        dod = self.dod_param
        n_bit = self.n_bit_param

        x_out = np.zeros((times,self.n_param),dtype = self.float_type)
        fval = np.full(times, float('inf'))

        def interval_to_float(int_i, a, b, n_bits):
            return(b-a)/(2**n_bits-1)*int_i + a
        
        iterations = np.zeros((times, 1))
        winning_run = 0
        t = []

        for run_i in range(times):
            decoded_list = []
            scores_list = []
            start = time.time()
            # prepare K
            self.K[:] = [np.random.randint(0, 2**n_bit[i]) for i in range(self.n_param)]
            run_fval = float('inf')

            for iter_i in range(max_iter):
                #print(iter_i)
                # K xor M -> result B
                np.bitwise_xor(self.K, self.M, out = self.B)
                # Decode Graye B to I -> result I 
                np.bitwise_and(self.B, 1 << n_bit, out=self.I)
                for par in range(self.n_param):
                    for bit in range(n_bit[par], 0, -1):
                        self.I[:,par] |= np.bitwise_xor((self.I[:,par] & 1<<bit)>>1,self.B[:,par] & 1<<(bit-1))
                        #self.I[:, par] |= np.bitwise_xor((self.I[:, par] & 1 << bit) >>1, self.B[:,par]&1<<(bit-1))
                    # Convert I to real numbers -> result R
                    self.R[:,par] = interval_to_float(self.I[:,par], dod[par,0], dod[par,1], n_bit[par])

                # Calculating the chase of the objective function -> result F
                self.F = [func(c) for c in self.R]
                

                # select the best one and then either terminate it or declare it a new K
                best_idx = np.argmin(self.F)
                
                run_fval = self.F[best_idx]
                
                
                decoded_list.append(self.R[:,par])
                scores_list.append(run_fval)

                # 1. basic quiting
                if best_idx == 0:
                    break                
                # if run_fval < self.fitness:
                #    break

                self.K = self.B[best_idx, :]
            iterations[run_i] = iter_i
            x_out[run_i,:] = self.R[best_idx,:]

            if run_fval < min(fval):
                winning_run = run_i

            fval[run_i] = run_fval
            t.append(time.time() - start)

            print(f'RUN: {run_i+1} of {times}, min fx = {fval[run_i]}')
        return x_out, fval, t, decoded_list, scores_list

#DEFINE FUNCTION
def rastrigin_plot(X1, X2):
    return (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (X2**2 - 10 * np.cos(2 * np.pi * X2)) + 20

def schwefel_plot(X1, X2):
    return 418.9829*2 - (X1*np.sin(np.sqrt(np.absolute(X1)))+X2*np.sin(np.sqrt(np.absolute(X2))))

def rosenbrock_plot(X1, X2):
    return  100*(X2-X1**2)**2+(X1-1)**2

#PLOT FUNCTION
def plot_function():
    if fce == rastrigin:
        X1 = np.linspace(-5.12, 5.12, 100)     
        X2 = np.linspace(-5.12, 5.12, 100)     
        X1, X2 = np.meshgrid(X1, X2) 
        Y = rastrigin_plot(X1, X2)

    elif fce == schwefel:
        X1 = np.linspace(-500, 500, 100)     
        X2 = np.linspace(-500, 500, 100)     
        X1, X2 = np.meshgrid(X1, X2) 
        Y = schwefel_plot(X1, X2)

    elif fce == rosenbrock:
        X1 = np.linspace(-10, 10, 100)     
        X2 = np.linspace(-10, 10, 100)     
        X1, X2 = np.meshgrid(X1, X2) 
        Y = rosenbrock_plot(X1, X2)
    
    fig = plt.figure()
    ax = plt.axes(projection ="3d")  
    ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=cm.nipy_spectral, linewidth=0.1, antialiased=True, alpha = 0.1) 
    x_list = list(zip(*x))
    ax.scatter3D(x_list[0], x_list[1], fx, color='r', marker='x', s=80, label = 'Final minimums')
    x_list2 = list(zip(*decoded_list))
    #ax.scatter3D(x_list2[0], x_list2[1], scores_list, color='g', label = 'Interim minimums') 
    ax.set_title('Plot Function')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.legend()
    plt.show()


if  __name__ == "__main__":
    fce = schwefel
    RUNs = 10
    D = 2
    nBitParam = 10
    maxGener = 10

    if fce == rastrigin:
        dodParam=[-5.12,5.12] # rastrigin

    elif fce == schwefel:
        dodParam=[-500, 500] # schwefel
        
    elif fce == rosenbrock:
        dodParam=[-10, 10] # rosenbrock

    hc12_instance = HC12(n_param=D,n_bit_param = nBitParam, dod_param=dodParam) 
    x, fx, t, decoded_list, scores_list = hc12_instance.hc12(func=fce,times=RUNs,max_iter=maxGener)
    

    print(f'\nMaximum y: {max(fx)}')
    print(f'Minimum y: {min(fx)}')
    print(f'Průměr y: {statistics.mean(fx)}')
    print(f'Median y: {statistics.median(fx)}')
    print("Nalezená minimální hodnota: X= {0}, Y= {1}".format(x[np.argmin(fx)], min(fx)))

    print(f'\nMaximum t: {max(t)}')
    print(f'Minimum t: {min(t)}')
    print(f'Průměr t: {statistics.mean(t)}')
    print(f'Median t: {statistics.median(t)}')

    if D == 2:
       plot_function()