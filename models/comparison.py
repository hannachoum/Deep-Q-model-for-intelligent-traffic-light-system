import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cross import *
from city import *

class No_learning_coordinator:

    def __init__(self, N:int, M:int, mini_thru:int, maxi_thru:int, mini_in:int,  maxi_in:int):

        self.time=0

        self.possible=[[0,4],[1,5],[2,6],[3,7],[0,5],[1,4],[2,7],[3,6]]

        self.N=N
        self.M=M

        self.mini_thru=mini_thru
        self.maxi_thru=maxi_thru

        self.mini_in=mini_in
        self.maxi_in=maxi_in

        self.city=City(self.N, self.M, self.mini_in, self.maxi_in)

        #Intersections, the dictionary of all the intersections depends on coordinator (as both City and Q_Learner2 use it)
        #We also initialise the dictionary of all the Q_Learners:

        self.all_intersections={}

        for i in range(self.N):
            for j in range(self.M):

                self.all_intersections[(i,j)]=Intersection(self.mini_thru, self.maxi_thru)

    def reset_game(self):

        #In case one agent loses the game, reset the intersections:
        self.all_intersections={}

        for i in range(self.N):
            for j in range(self.M):
                self.all_intersections[(i,j)]=Intersection(self.mini_thru, self.maxi_thru)

        #Then need to harmonise to make each intersection aware of its neighbors:
        self.city.set_harmony(self.all_intersections)

    def episodes_random_agent(self):

        print("RANDOM")

        self.time+=1
        rewards=[0 for _ in range(self.N*self.M)]
        episodes=[]

        for iteration in range(1000):
            print(iteration)
            episode=0
            while np.min(rewards)>-80:
                for i in range(self.N):
                    for j in range(self.M):

                        self.action=np.random.randint(0,8)
                        
                        self.all_intersections[(i,j)].new_round(self.action)
                        self.all_intersections[(i,j)].turning_right()
                        #BOUNDARY:
                        #1. exit
                        self.city.update_exit_boundary(self.all_intersections,i,j)

                        #2. new cars
                        self.city.add_at_boundary(self.all_intersections, i, j)
                        self.all_intersections[(i,j)].rewarding()
                        rewards[i*self.M+j]=self.all_intersections[(i,j)].reward
                        

                for i in range(self.N):
                    for j in range(self.M):       

                        self.city.harmonise(self.all_intersections, i, j)

                episode+=1

            self.reset_game()
            rewards=[0 for _ in range(self.N*self.M)]
            episodes.append(episode)

        print(episodes)
        return np.mean(episodes)

    def episodes_cyclic_agent(self):

        print("CYCLIC")

        self.time+=1
        rewards=[0 for _ in range(self.N*self.M)]
        episodes=[]

        for iteration in range(1000):
            print("iteration",iteration)
            actions=[np.random.randint(0,8) for _ in range(self.N*self.M)]
            episode=0
            while np.min(rewards)>-80:
                for action in range(len(actions)):
                    actions[action]+=1
                    actions[action]%=8
                for i in range(self.N):
                    for j in range(self.M):
                        
                        self.all_intersections[(i,j)].new_round(actions[self.M*i+j])
                        self.all_intersections[(i,j)].turning_right()
                        
                        #BOUNDARY:
                        #1. exit
                        self.city.update_exit_boundary(self.all_intersections,i,j)
                        #2. new cars
                        self.city.add_at_boundary(self.all_intersections, i, j)

                        self.all_intersections[(i,j)].rewarding()
                        rewards[i*self.M+j]=self.all_intersections[(i,j)].reward
                        

                for i in range(self.N):
                    for j in range(self.M):       

                        self.city.harmonise(self.all_intersections, i, j)

                episode+=1

            self.reset_game()
            rewards=[0 for _ in range(self.N*self.M)]
            episodes.append(episode)

        print(episodes)
        return np.mean(episodes)

    def episodes_greedy_agent(self):

        print("GREEDY")

        self.time+=1
        rewards=[0 for _ in range(self.N*self.M)]
        episodes=[]

        for iteration in range(1000):
            print("iteration",iteration)
            episode=0
            action=0
            while np.min(rewards)>-80:
                
                for i in range(self.N):
                    for j in range(self.M):

                        phase_number=[0 for _ in range(8)]
                        for phase in range(8):
                            for lane in range(2):
                                phase_number[phase]+=self.all_intersections[(i,j)].waiting[self.possible[phase][lane]]
                        action=np.argmax(phase_number)
                        
                        self.all_intersections[(i,j)].new_round(action)
                        self.all_intersections[(i,j)].turning_right()
                        #BOUNDARY:
                        #1. exit
                        self.city.update_exit_boundary(self.all_intersections,i,j)

                        #2. new cars
                        self.city.add_at_boundary(self.all_intersections, i, j)
                        #print(self.all_intersections[(i,j)].waiting)
                        self.all_intersections[(i,j)].rewarding()
                        rewards[i*self.M+j]=self.all_intersections[(i,j)].reward
                        

                for i in range(self.N):
                    for j in range(self.M):       

                        self.city.harmonise(self.all_intersections, i, j)

                episode+=1

            self.reset_game()
            rewards=[0 for _ in range(self.N*self.M)]
            episodes.append(episode)

        print(episodes)
        return np.mean(episodes)

        