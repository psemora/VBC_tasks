import random
import math, copy
import numpy as np


class Bat:
    def __init__(self, D, A, r, F_min, F_max, X_min, X_max, fun):
        self.D = D	
        self.X_min = X_min
        self.X_max = X_max
        self.X = []
        for i in range(self.D):
            rnd = random.random()
            self.X += [self.X_min + (self.X_max - self.X_min) * rnd]
        self.X = np.array(self.X)
        self.A = A
        self.r = r	
        self.F_min = F_min
        self.F_max = F_max
        self.F = self.F_min + (self.F_max - self.F_min) * random.uniform(0, 1)		
        self.fun = fun
        self.sol = self.fun(self.D, self.X)

    def updateFrequency(self, rand_num):
        self.F = self.F_min + rand_num * (self.F_max - self.F_min)	
    
    def adjust_range(self):
        while True:
            for i in range(self.D):
                if self.X[i] > self.X_max:
                    self.X[i] = self.X_max
                if self.X[i] < self.X_min:
                    self.X[i] = self.X_min
                self.randomize_position()
            else:
                break
        self.sol = self.fun(self.D, self.X)

    def randomize_position(self):
        for i in range(0, self.D):
            self.X[i] = self.X_min + (self.X_max - self.X_min)*random.random()

        
    def jump(self, b):	
        for i in range(self.D):
            self.X[i] = b.X[i] + 0.3 * random.gauss(0, 1)
        self.adjust_range()
        
    
    def changeA(self, a):
        self.A = self.A * a
    
    def changeR(self, g):
        self.r = self.r * (1 - math.exp(-g))
    
    def getCopy(self):
        return copy.deepcopy(self)


class BatAlgorithm:
    def __init__(self, D, NP, N_gen, A_min, A_max, F_min, F_max, X_min, X_max, fitness_fun):
        self.D = D	#Dimension
        self.NP = NP	#Population
        self.N_gen = N_gen	# Number of generations
        self.A_min = A_min	#min Loudness
        self.A_max = A_max	#max Loudness
        
        self.F_min = F_min	#min frequency
        self.F_max = F_max	#max frequency
        self.fitness_fun = fitness_fun	#fitness function

        self.X_min = X_min	#min X
        self.X_max = X_max	#max X
        
        self.alpha = 0.95
        self.gamma = 0.05
        
        self.bats = []
        self.decoded_list = list()
        self.scores_list = list()


        for i in range(NP):
            b = Bat(D, A_min + (A_max - A_min)*random.random(), random.random(), F_min, F_max, X_min, X_max, fitness_fun)
            b.sol = self.fitness_fun(self.D, b.X)
            self.bats += [b]
            self.decoded_list.append(b.X)
            self.scores_list.append(b.sol)
        self.best_bat = self.get_best_bat()

    def get_best_bat(self):
        i = 0
        for j in range(0, self.NP):
            if(self.bats[i].sol > self.bats[j].sol):
                i = j
        return self.bats[i]

    def move_bats(self):
        const_sol = self.best_bat.sol
        counter = 0
        for b in  self.bats:
            b.sol = b.fun(b.D, b.X)

        t = 0
        while True:
            t += 1
            cnt = 1
            x_graph, y_graph = [], []
            
            for index, bat in enumerate(self.bats):
                    rnd_num = random.uniform(0, 1)	
                    self.bats[index].updateFrequency(rnd_num)
                    tmp_bat = self.bats[index].getCopy()
                    tmp_bat.sol = self.fitness_fun(tmp_bat.D, tmp_bat.X)
                    rnd_num = random.random()	

                    if rnd_num > self.bats[index].r:
                        tmp_best_bat = self.get_best_bat()
                        tmp_bat.jump(tmp_best_bat)
                    
                    tmp_bat.sol = self.fitness_fun(self.D, tmp_bat.X)
                    
                    rnd_num = random.random()
                    if rnd_num < tmp_bat.A and tmp_bat.sol < self.bats[index].sol:	
                        self.bats[index] = tmp_bat.getCopy()	
                        self.bats[index].changeA(self.alpha)
                        self.bats[index].changeR(self.gamma*t)
                    if self.bats[index].sol < self.best_bat.sol:
                        self.best_bat = bat.getCopy()

                    
                    cnt += 1	
                    y_graph += [self.bats[index].sol]
                    x_graph += [self.bats[index].X[0]]
            if const_sol == self.best_bat.sol:
                counter += 1
            else:
                const_sol= self.best_bat.sol
                counter = 0
            if counter > 200:
                break

        return self.best_bat.X, self.best_bat.sol, self.decoded_list, self.scores_list