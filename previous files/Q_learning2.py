import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from FRAP import *
from city_cross2 import *
from city2 import *

#Loss function:
loss=nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

class Q_Learner3:

    def __init__(self, gamma: float):

        self.gamma=gamma
        self.Q={} #dictionary of all Q with (SARS) as a key

        self.reward=0
        self.gain=0
        self.time=0

        #Define initial state of the intersection:
        #For the pressure, the is according to the intersection given (intersection depends on coordinator for clarity between classes)
        #PHASES needs to be turned into an 8-vector with 0 where traffic light is red and 1 where traffic light is green
        self.possible_phase=[[0,4],[1,5],[2,6],[3,7],[0,5],[1,4],[2,7],[3,6]]
        phase_vec=torch.zeros((1,8))

        #Arbitrarily set the initial phase to 0:
        for lane in self.possible_phase[0]:
            phase_vec[0,lane]=1

        self.phase=phase_vec
        self.action=0
        self.out=0

    def next_round(self, intersection:Intersection4, epsilon:float, frap:FRAP):

        #Phase is already known, need to find the pressure to get the state of the intersection:
        #PRESSURE
        waiting=intersection.n_waiting
        for lane in range(8):
            waiting[lane]/=50
        
        self.pressure=torch.Tensor([waiting])

        #y_hat:
        out=frap.forward(self.pressure.to(device), self.phase.to(device))
        self.out=out

        #POLICY
        proba=np.random.uniform(0,1)

        if proba>epsilon:
            action=out.argmax()
        else:
            action=int(np.random.randint(0,8))

        self.action=action
        self.y_hat=out[action]

        #Now MAKE A STEP in the environment with s_t and a_t
        intersection.new_round(action)

        #SAVE CHANGES FOR STATE s_{t+1}:
        #1. No need for new pressure, pressure is computed at the beginning of next_round
        #2. NEW PHASE
        phase_vec=torch.zeros((1,8))
        for lane in self.possible_phase[action]:
            phase_vec[0,lane]=1

        self.phase=phase_vec


    def rewards(self, intersection:Intersection4, frap_target:FRAP):

        #REWARD r_t
        intersection.rewarding()
        self.reward=intersection.reward

        #y
        out2=frap_target.forward(self.pressure.to(device), self.phase.to(device))
        self.out2=out2
        Q_next=out2.max()
        self.y=self.reward/300 + self.gamma*Q_next
        
        self.time+=1

    def get_loss(self):

        return loss(self.y, self.y_hat)