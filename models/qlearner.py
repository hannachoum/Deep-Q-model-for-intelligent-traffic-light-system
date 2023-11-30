import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from frap2 import *
from cross import *
from city import *

#Loss function:
loss=nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

class Q_Learner:

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

        #Initialise but will be modified later:
        self.wtime_mu=None
        self.wtime_sigma=None
        self.wtime_max=None


    def make_state(self, intersection:Intersection):

        #1. Prepare state for network:
        #Phase is already known, need to find the pressure to get the state of the intersection:
        #1.1 PRESSURE
        waiting=intersection.waiting.copy()
        for lane in range(8):
            waiting[lane]/=30
        
        self.pressure=torch.Tensor([waiting])

        #1.2 WAITING TIME (mu, sigma, max):
        wtime_mu=[]
        wtime_sigma=[]
        wtime_max=[]

        for lane in range(8):

            if intersection.wtime[lane]!=[]:
                wtime_mu.append(np.mean(intersection.wtime[lane]))
                wtime_sigma.append(np.std(intersection.wtime[lane]))
                wtime_max.append(np.max(intersection.wtime[lane]))

            else:
                wtime_mu.append(0)
                wtime_sigma.append(0)
                wtime_max.append(0)

            wtime_mu[-1]/=5
            wtime_sigma[-1]/=5
            wtime_max[-1]/=10

        self.wtime_mu=torch.Tensor([wtime_mu])
        self.wtime_sigma=torch.Tensor([wtime_sigma])
        self.wtime_max=torch.Tensor([wtime_max])


    def next_round(self, intersection:Intersection, epsilon:float, frap:FRAP2, i:int, j:int):

        i_position=torch.full((1,1),i)
        j_position=torch.full((1,1),j)


        #y_hat:
        out=frap.forward( self.pressure.to(device), self.phase.to(device), self.wtime_mu.to(device), self.wtime_sigma.to(device), self.wtime_max.to(device), i_position.to(device), j_position.to(device))
        self.out=out

        #POLICY
        proba=np.random.uniform(0,1)

        if proba>epsilon:
            action=out.argmax()
        else:
            action=int(np.random.randint(0,8))

        self.action=action
        self.y_hat=out[:,action][0]

        #Now MAKE A STEP in the environment with s_t and a_t
        intersection.new_round(action)

        #NEW STATE
        #NEW PHASE (rest is computed in make_new_state as we do not need the harmony for phase and we need the action taken unlike the rest of state features)
        phase_vec=torch.zeros((1,8))
        for lane in self.possible_phase[action]:
            phase_vec[0,lane]=1

        self.phase=phase_vec

    def get_reward(self, intersection:Intersection):

        #REWARD r_t
        intersection.rewarding()
        self.reward=intersection.reward


    def make_new_state(self, intersection:Intersection):

        #2.SAVE CHANGES FOR STATE s_{t+1}:
        #2.1 NEW PRESSURE:
        waiting=intersection.waiting.copy()
        for lane in range(8):
            waiting[lane]/=30

        
        self.pressure=torch.Tensor([waiting])

        #2.2 WAITING TIME (mu, sigma, max):
        wtime_mu=[]
        wtime_sigma=[]
        wtime_max=[]

        for lane in range(8):

            if intersection.wtime[lane]!=[]:
                wtime_mu.append(np.mean(intersection.wtime[lane]))
                wtime_sigma.append(np.std(intersection.wtime[lane]))
                wtime_max.append(np.max(intersection.wtime[lane]))

            else:
                wtime_mu.append(0)
                wtime_sigma.append(0)
                wtime_max.append(0)

            wtime_mu[-1]/=5
            wtime_sigma[-1]/=5
            wtime_max[-1]/=10

        self.wtime_mu=torch.Tensor([wtime_mu])
        self.wtime_sigma=torch.Tensor([wtime_sigma])
        self.wtime_max=torch.Tensor([wtime_max])


    def target(self, frap:FRAP2, frap_target:FRAP2, i:int, j:int, last:bool):

        i_position=torch.full((1,1),i)
        j_position=torch.full((1,1),j)

        get_a=frap.forward( self.pressure.to(device), self.phase.to(device), self.wtime_mu.to(device), self.wtime_sigma.to(device), self.wtime_max.to(device),  i_position.to(device), j_position.to(device))
        action2=get_a.max(dim=1).indices

        out2=frap_target.forward( self.pressure.to(device), self.phase.to(device), self.wtime_mu.to(device), self.wtime_sigma.to(device), self.wtime_max.to(device),  i_position.to(device), j_position.to(device))
        self.out2=out2
        Q_next=out2[:,action2][0][0]

        #y:
        #If we are NOT at the end of an episode:
        if last==False:
            self.y=10*(self.reward+120)/150+ self.gamma*Q_next
            
        #If we are at the end of an episode:
        else:
            self.y=torch.tensor(10*(self.reward+120)/150).float()

        # print((self.reward+100)/130-0.5, Q_next, self.out2)
        
        self.time+=1

    def get_loss(self):

        return loss(self.y, self.y_hat)