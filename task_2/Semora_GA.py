import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 
from numpy.random import randint
from numpy.random import rand
import statistics 

#DEFINE FUNCTION
def rastrigin(X1, X2):
    return (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (X2**2 - 10 * np.cos(2 * np.pi * X2)) + 20

def schwefel(X1, X2):
    return 418.9829*2 - (X1*np.sin(np.sqrt(np.absolute(X1)))+X2*np.sin(np.sqrt(np.absolute(X2))))

def rosenbrock(X1, X2):
    return  100*(X2-X1**2)**2+(X1-1)**2

#PLOT FUNCTION
def plot_function():
    if fce == rastrigin:
        X1 = np.linspace(-5.12, 5.12, 100)     
        X2 = np.linspace(-5.12, 5.12, 100)     
        X1, X2 = np.meshgrid(X1, X2) 
        Y = rastrigin(X1, X2)

    elif fce == schwefel:
        X1 = np.linspace(-500, 500, 100)     
        X2 = np.linspace(-500, 500, 100)     
        X1, X2 = np.meshgrid(X1, X2) 
        Y = schwefel(X1, X2)

    elif fce == rosenbrock:
        X1 = np.linspace(-10, 10, 100)     
        X2 = np.linspace(-10, 10, 100)     
        X1, X2 = np.meshgrid(X1, X2) 
        Y = rosenbrock(X1, X2)
    
    fig = plt.figure()
    ax = plt.axes(projection ="3d")  
    ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=cm.nipy_spectral, linewidth=0.1, antialiased=True, alpha = 0.1) 
    x_list = list(zip(*x_final))
    ax.scatter3D(x_list[0], x_list[1], y_final, color='r', marker='x', s=80, label = 'Final minimums')
    x_list2 = list(zip(*decoded_list))
    #ax.scatter3D(x_list2[0], x_list2[1], scores_list, color='g', label = 'Interim minimums') 
    ax.set_title('Plot Function')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.legend()
    plt.show()

#GENETIC ALGORITHM
# objective function
def objective(X):
    x = np.asarray_chkfinite(X)
    A = 10
    if fce == rastrigin: 
        if len(x.shape) ==2:
            axis=1
        else:
            axis = 0
        n = np.size(x,axis)
        y = A*n + np.sum(x**2-A*np.cos(2*np.pi*x))
        return y

    elif fce == schwefel:
        if len(x.shape) == 2:
            axis = 1
        else:
            axis = 0
        n = np.size(x,axis)
        y = 418.9829*n - np.sum(x*np.sin(np.sqrt(np.abs(x))))
        return y


    elif fce == rosenbrock:
        if len(x.shape) == 2:
            axis = 1
        else:
            axis = 0
        n = np.size(x,axis)
        y = 0
        for i in range(n-1):
            y += 100*np.power((x[i+1] - np.power(x[i],2)),2) + np.power((x[i]-1),2)
    return y
        

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded

# tournament selection
def selection(pop, scores):
    # first random selection
    k = int(pS*len(pop)) # pravděpodobnost selekce
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

 
# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
 
# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(d) for d in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                #print("Gen:%d, new best f = %f" % (gen,  scores[i])) #Průběh optimalizace 
                decoded_list.append(decoded[i])
                scores_list.append(scores[i])
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]

if  __name__ == "__main__":
    decoded_list = []
    scores_list = []
    t = []
    x_final = list()
    y_final = list()

    fce = rastrigin     
    RUNs = 10
    D = 2

    # define range for input
    if fce == rastrigin:
        bounds = [[-5.12, 5.12]]*D #rastrigin 
    elif fce == schwefel:
        bounds = [[-500, 500]]*D #schwefel
    elif fce == rosenbrock:
        bounds = [[-10, 10]]*D #rosenbrock

    NP = 100 #Velikost populace
    pS = 0.5 #Síla selekce
    n_bits = 16 # bits per variable
    pC = 0.5 #pravděpodobnost křížení
    pM = 1 / (float(n_bits) * len(bounds)) #pravděpodobnost mutace
    maxGener = 100 #Počet iterací

    
    # perform the genetic algorithm search
    for i in range(RUNs):
        start = time.time()
        best, score = genetic_algorithm(objective, bounds, n_bits, maxGener, NP, pC, pM)
        decoded = decode(bounds, n_bits, best)
        t.append(time.time() - start)
        print(f'RUN: {i+1} of {RUNs}: min fx = {score}')
        x_final.append(decoded)
        y_final.append(score)

    print(f'\nMaximum y: {max(y_final)}')
    print(f'Minimum y: {min(y_final)}')
    print(f'Průměr y: {statistics.mean(y_final)}')
    print(f'Median y: {statistics.median(y_final)}')
    print("Nalezená minimální hodnota: X= {0}, Y= {1}".format(x_final[np.argmin(y_final)], min(y_final)))

    print(f'\nMaximum t: {max(t)}')
    print(f'Minimum t: {min(t)}')
    print(f'Průměr t: {statistics.mean(t)}')
    print(f'Median t: {statistics.median(t)}')

    if D == 2:
        plot_function()