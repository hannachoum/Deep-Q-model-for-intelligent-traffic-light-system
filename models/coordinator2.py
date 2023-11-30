import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from frap2 import *
from cross import *
from city import *
from qlearner import *
from comparison import *

#Loss function:
loss=nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

run_number="3"

min_reward=-100

#Replay_buffer
length_D=100000
use_buffer=True
batch_size=128

#Epsilon decay
epsilon_decay=0.99999

class Coordinator:

    def __init__(self, N:int, M:int, gamma:float, mini_thru:int, maxi_thru:int, mini_in:int,  maxi_in:int):

        self.gamma=gamma
        self.time=0

        #Size of the city
        self.N=N
        self.M=M

        #Cars going through when given green light
        self.mini_thru=mini_thru
        self.maxi_thru=maxi_thru

        #Cars adding up at every round
        self.mini_in=mini_in
        self.maxi_in=maxi_in

        #Define the city:
        self.city=City(self.N, self.M, self.mini_in, self.maxi_in)

        #If at the last iteration of an episode
        self.last={} 

        #Intersections and Q_learners:
        self.all_intersections={}
        self.learners={}

        for i in range(self.N):
            for j in range(self.M):

                self.all_intersections[(i,j)]=Intersection(self.mini_thru, self.maxi_thru)
                self.learners[(i,j)]=Q_Learner(self.gamma)
                self.last[(i,j)]=False

        #FRAP: (shared by all learners)

        #1. FRAP:
        self.frap=FRAP2().to(device)

        #2. FRAP_target (not updated as often)
        self.frap_target=copy.deepcopy(self.frap).to(device)

        #Initialise the replay buffer replay_D
        self.replay_D_s=np.zeros((length_D,10,8)) #To store the state variables
        self.replay_D_ar=np.zeros((length_D,4)) #To store int (action number and reward)

    def next_round(self, epsilon:float): #epsilon is for the epsilon greedy policy

        self.time+=1

        #Make each learner find an appropriate action and play it in its intersection
        for i in range(self.N):
            for j in range(self.M):

                #REPLAY BUFFER: (initial pressure, initial phase)
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 1,:]=self.learners[(i,j)].phase

                if self.N-1!=0:
                    self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 2]=i/(self.N-1)
                    i_position=i/(self.N-1)
                else:
                    self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 2]=i
                    i_position=i
                if self.M-1!=0:
                    self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 3]=j/(self.M-1)
                    j_position=j/(self.M-1)
                else:
                    self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 3]=j
                    j_position=j

                #Define STATE:
                self.learners[(i,j)].make_state(self.all_intersections[(i,j)])

                #REPLAY BUFFER: (wtime: mu, sigma, max)
                #print(self.learners[(i,j)].wtime_mu[0], self.learners[(i,j)].wtime_max[0])
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 4,:]=self.learners[(i,j)].wtime_mu[0]
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 5,:]=self.learners[(i,j)].wtime_sigma[0]
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 6,:]=self.learners[(i,j)].wtime_max[0]

                print(self.learners[(i,j)].wtime_mu[0], self.learners[(i,j)].wtime_sigma[0], self.learners[(i,j)].wtime_max[0])

                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 0,:]=self.learners[(i,j)].pressure[0]
                
                #NEXT ROUND

                self.learners[(i,j)].next_round(self.all_intersections[(i,j)], epsilon, self.frap, i_position, j_position)

                self.all_intersections[(i,j)].turning_right()

                #BOUNDARY:
                #1. exit
                self.city.update_exit_boundary(self.all_intersections,i,j)

                #2. new cars
                self.city.add_at_boundary(self.all_intersections, i, j)

                #REWARD:
                self.learners[(i,j)].get_reward(self.all_intersections[(i,j)])

                #REPLAY BUFFER: (action, new phase)
                self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 0]=self.learners[(i,j)].action
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 3,:]=self.learners[(i,j)].phase
                self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 1]=self.learners[(i,j)].reward


        #HARMONISE the city after:
        for i in range(self.N):
            for j in range(self.M):       

                self.city.harmonise(self.all_intersections, i, j)

                #Check if last episode:
                if self.learners[(i,j)].reward<min_reward:
                    self.last[(i,j)]=True


        for i in range(self.N):
            for j in range(self.M):

                if self.N-1!=0:
                    i_position=i/(self.N-1)
                else:
                    i_position=i
                if self.M-1!=0:
                    j_position=j/(self.M-1)
                else:
                    j_position=j

                #define NEW STATE:
                self.learners[(i,j)].make_new_state(self.all_intersections[(i,j)])

                #REPLAY BUFFER for new state: (wtime: mu, sigma, max) & waiting
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 7,:]=self.learners[(i,j)].wtime_mu[0]
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 8,:]=self.learners[(i,j)].wtime_sigma[0]
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 9,:]=self.learners[(i,j)].wtime_max[0]

                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 2,:]=self.learners[(i,j)].pressure[0]

                #COMPUTE next
                self.learners[(i,j)].target(self.frap, self.frap_target, i_position, j_position, self.last[(i,j)])

                self.last[(i,j)]=False