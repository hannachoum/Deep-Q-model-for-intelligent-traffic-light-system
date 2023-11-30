import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from NN import *
from intersection2 import *
from city2 import *

#Loss function:
loss=nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

class RL_agent:

    def __init__(self, gamma: float, i:int, j:int, N:int, M:int):

        self.gamma=gamma

        #Find the position according to max:
        if N-1!=0:
            new_i=i/(N-1)
        else:
            new_i=i
        if M-1!=0:
            new_j=j/(M-1)
        else:
            new_j=j
            
        self.i_position=torch.full((1,1),new_i)
        self.j_position=torch.full((1,1),new_j)

        self.possible_phase=[[0,4],[1,5],[2,6],[3,7],[0,5],[1,4],[2,7],[3,6]]

        #Arbitrarily set the initial phase to phase (=action) 0:
        self.action=0
        self.reward=0
        self.phase=torch.zeros((1,8))

        for lane in self.possible_phase[0]:
            self.phase[0,lane]=1

        #Initialise but will be modified later:
        self.wtime_mu=None
        self.wtime_sigma=None
        self.wtime_max=None
        self.y=None
        self.y_hat=None

    def make_state(self, intersection:Crossing5):

        #1. Prepare state for network:
        #Phase is already known, need to find the pressure to get the state of the intersection:
        #1.1 PRESSURE
        waiting=copy.deepcopy(intersection.waiting)
        for lane in range(8):
            waiting[lane]/=30
        
        self.pressure=waiting.view((1,12))

        #1.2 WAITING TIME (mu, sigma, max):
        wtime_mu=torch.zeros((1,12))
        wtime_sigma=torch.zeros((1,12))
        wtime_max=torch.zeros((1,12))

        for lane in range(12):

            if intersection.wtime[lane].shape[0]!=0:
                wtime_mu[0,lane]=torch.mean(intersection.wtime[lane])
                wtime_max[0,lane]=torch.max(intersection.wtime[lane])

                if intersection.wtime[lane].shape[0]!=1:
                    wtime_sigma[0,lane]=torch.std(intersection.wtime[lane])

        #print(intersection.wtime)
        #print(self.wtime_mu, self.wtime_sigma, self.wtime_max)

        self.wtime_mu=torch.div(wtime_mu, 5)
        self.wtime_sigma=torch.div(wtime_sigma, 5)
        self.wtime_max=torch.div(wtime_max, 10)

    def next_round(self, intersection:Crossing5, epsilon:float, frap:FRAP5):

        #y_hat:
        #print("HAT")
        out=frap.forward( self.pressure.to(device), self.phase.to(device), self.wtime_mu.to(device), self.wtime_sigma.to(device), self.wtime_max.to(device), self.i_position.to(device), self.j_position.to(device))
        self.out=out
        #print(self.out)

        #POLICY
        proba=np.random.uniform(0,1)

        if proba>epsilon:
            action=out.argmax()
        else:
            action=int(np.random.randint(0,8))

        self.action=action
        self.y_hat=out[:,action][0]

        #Now MAKE A STEP in the environment with s_t and a_t
        intersection.next_round(action)

        #NEW STATE
        #NEW PHASE (rest is computed in make_new_state as we do not need the harmony for phase and we need the action taken unlike the rest of state features)
        self.phase=torch.zeros((1,8))
        for lane in self.possible_phase[action]:
            self.phase[0,lane]=1

    def get_reward(self, intersection:Crossing5):

        #REWARD r_t
        intersection.rewarding()
        self.reward=torch.tensor(5*(intersection.reward+120)/300).float()


    def target(self, frap:FRAP5, frap_target:FRAP5, last:bool):

        get_a=frap.forward( self.pressure.to(device), self.phase.to(device), self.wtime_mu.to(device), self.wtime_sigma.to(device), self.wtime_max.to(device),  self.i_position.to(device), self.j_position.to(device))
        action2=get_a.max(dim=1).indices

        out2=frap_target.forward( self.pressure.to(device), self.phase.to(device), self.wtime_mu.to(device), self.wtime_sigma.to(device), self.wtime_max.to(device),  self.i_position.to(device), self.j_position.to(device))
        self.out2=out2
        Q_next=out2[:,action2][0][0]

        #y:
        #If we are NOT at the end of an episode:
        if last==False:
            self.y=self.reward+ self.gamma*Q_next
            
        #If we are at the end of an episode:
        else:
            #self.y=self.reward
            self.y=self.reward+ self.gamma*Q_next

    
    def get_loss(self):

        return loss(self.y, self.y_hat)
