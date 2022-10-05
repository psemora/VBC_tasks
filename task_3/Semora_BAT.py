import random
from MyBat import BatAlgorithm
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 
import statistics 

def rastrigin(D, X):
    y =  10*D + sum(X**2 - 10 * np.cos(2 * math.pi * X))
    #y = (X[0]**2 - 10 * np.cos(2 * np.pi * X[0])) + (X[1]**2 - 10 * np.cos(2 * np.pi * X[1])) + 20
    return y

def schwefel(D,X):
    y =  418.9829*D - sum( X * np.sin( np.sqrt( abs( X ))))
    #y =  418.9829*2 - (X[0]*np.sin(np.sqrt(np.absolute(X[0])))+X[1]*np.sin(np.sqrt(np.absolute(X[1])))) #schwefel
    return y

def rosenbrock(D,X):
    #y = 100*(X[1]-X[0]**2)**2+(X[0]-1)**2 #rosenbrock
    x0 = X[:-1]
    x1 = X[1:]
    y = sum(100*(x1-x0**2)**2 + (x0-1)**2)
    return y

def rastrigin_plot(X1, X2):
    return (X1**2 - 10 * np.cos(2 * np.pi * X1)) + (X2**2 - 10 * np.cos(2 * np.pi * X2)) + 20

def schwefel_plot(X1, X2):
    return 418.9829*2 - (X1*np.sin(np.sqrt(np.absolute(X1)))+X2*np.sin(np.sqrt(np.absolute(X2))))

def rosenbrock_plot(X1, X2):
    return  100*(X2-X1**2)**2+(X1-1)**2

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
    x_list2 = list(zip(*x_final))
    ax.scatter3D(x_list2[0], x_list2[1], y_final, color='r', marker='x', s=80, label = 'Final minimums')  
    x_decoded_list = list(zip(*decoded_list))
    #ax.scatter3D(x_decoded_list[0], x_decoded_list[1], scores_list, color='g', label = 'Interim minimums') 
    ax.set_title('Plot Function')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.legend()
    plt.show()


if  __name__ == "__main__":
    t = []
    RUNs = 10
    NP = 100
    D = 2

    x_final = list()
    y_final = list()
    print("Choose Function:\n1. Rastrigin\n2. Schwefel\n3. Rosenbrock\n0. Exit")
    opt = int(input())
    if opt == 0:
        exit(0)
    elif opt == 1:
        fce = rastrigin
        Algorithm = BatAlgorithm(D, NP, NP*D, 0.5, 0.5, 0.0, 2.0, -5.12, 5.12, rastrigin)
        print("optimal Values: x = [0, 0], y = 0")

    elif opt == 2:
        fce = schwefel
        Algorithm = BatAlgorithm(D, NP, NP*D, 0.5, 0.5, 0.0, 2.0, -500, 500, schwefel)
        print("optimal Values: x = [420.9687, 420.9687], y = 0")

    elif opt == 3:
        fce = rosenbrock
        Algorithm = BatAlgorithm(D, NP, NP*D, 0.5, 0.5, 0.0, 2.0, -10, 10, rosenbrock)
        print("optimal Values: x = [1, 1], y = 0")
    

    for i in range(RUNs):
        start = time.time()
        x, y, decoded_list, scores_list = Algorithm.move_bats()
        t.append(time.time() - start)
        print("RUN : %d of %d, Min Y = %f" % (i+1, RUNs, y))
        #print("Position: ", x)
        #print("Minimized value = ", y)
        x_final.append(x)
        y_final.append(y)
    print("\nMaximum y: %f" % (max(y_final)))
    print("Minimum y: %f" % (min(y_final)))
    print("Průměr y:  %f" % (statistics.mean(y_final)))
    print("Median y:  %f"% (statistics.median(y_final)))
    print("Nalezená minimální hodnota: X = {0}, Y= {1}".format(x_final[y_final.index(min(y_final))], min(y_final)))

    print(f'\nMaximum t: {max(t)}')
    print(f'Minimum t: {min(t)}')
    print(f'Průměr t: {statistics.mean(t)}')
    print(f'Median t: {statistics.median(t)}')

    if D == 2:
       plot_function()