import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy

#CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

#DROPOUT
drop=0.2

#PHASE COMPETITION MASK:
competition=torch.zeros((8,8)).to(device)

competition[0,4]=1
competition[0,5]=1
competition[1,4]=1
competition[1,5]=1
competition[2,6]=1
competition[2,7]=1
competition[3,6]=1
competition[3,7]=1

for i in range(8):
    for j in range(8):
        if competition[i,j]==1:
            competition[j,i]=1

comp=torch.zeros((1,8,7)).to(device) #need this matrix to remove the diagonal parameters to get the final vector to be (8,7)
#We must have depth 1

for i in range(8):
    for  j in range(8):
        if j<i:
            comp[0,i,j]=competition[i,j]
        elif j>i:
            comp[0,i,j-1]=competition[i,j]

"""
class NN_phase(nn.Module):

    def __init__(self):
        super(NN_phase, self).__init__()

        self.network=nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 32),
            nn.ReLU(),
            #nn.Dropout(drop),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self,phase_vec):
        return self.network(phase_vec)

"""

class NN_demand(nn.Module):

    def __init__(self):
        super(NN_demand, self).__init__()

        self.network=nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 32),
            nn.ReLU(),
            #nn.Dropout(drop),
            nn.Linear(32, 4)
        )

    def forward(self,conc_demand):
        return self.network(conc_demand)

class NN_coupled_demand(nn.Module):

    def __init__(self):
        super(NN_coupled_demand, self).__init__()

        self.network=nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 32),
            nn.ReLU(),
            #nn.Dropout(drop),
            nn.Linear(32, 4)
        )

    def forward(self,conc_demand):
        return self.network(conc_demand)



class Conv_D(nn.Module):

    def __init__(self):
        super(Conv_D, self).__init__()

        self.conv1=nn.Conv2d(8,256,1)
        self.conv2=nn.Conv2d(256,32,1)
        self.conv3=nn.Conv2d(32,2,1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.dropout = nn.Dropout(drop)

    def forward(self,D):
        conv1=self.dropout(self.relu(self.conv1(D)))
        conv2=self.dropout(self.relu(self.conv2(conv1)))
        return self.conv3(conv2)

class Conv_C(nn.Module):

    def __init__(self):
        super(Conv_C, self).__init__()

        self.conv1=nn.Conv2d(3,256,1)
        self.conv2=nn.Conv2d(256,32,1)
        self.conv3=nn.Conv2d(32,8,1)
        self.conv4=nn.Conv2d(8,1,(1,7))
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
        self.dropout = nn.Dropout(drop)

    def forward(self,H_c):
        conv1=self.dropout(self.relu(self.conv1(H_c)))
        conv2=self.dropout(self.relu(self.conv2(conv1)))
        conv3=self.dropout(self.relu(self.conv3(conv2)))
        return self.conv4(conv3)

class FRAP(nn.Module):

    def __init__(self):
        
        super(FRAP, self).__init__()

        #define all possible phases (combinations of lanes that can get green light together)
        self.possible_phase=[[0,4],[1,5],[2,6],[3,7],[0,5],[1,4],[2,7],[3,6]]
        #phase is a couple of 2 lanes that can get green light together
        #phase NEEDS to be an 8-vector with 0 for red light, 1 for green

        #Define the neural networks
        #self.NN_phase=NN_phase()
        self.NN_demand=NN_demand()
        self.NN_coupled_demand=NN_coupled_demand()
        self.Conv_D=Conv_D()
        self.Conv_C=Conv_C()

    def forward(self, waiting:torch.Tensor, phase:torch.Tensor, wtime_mu:torch.Tensor, wtime_sigma:torch.Tensor, wtime_max:torch.Tensor, i_position:torch.Tensor, j_position:torch.Tensor):

        #DEMAND vector:
        phase=phase.to(device)

        demand=[]
        #n_phase=self.NN_phase.forward(phase)

        wait=waiting
        mu_list=wtime_mu
        sigma_list=wtime_sigma
        max_list=wtime_max

        for lane in range(8):

            demand.append(self.NN_demand(torch.cat((i_position[:,:], j_position[:,:], mu_list[:,lane:lane+1], sigma_list[:,lane:lane+1], max_list[:,lane:lane+1], wait[:,lane:lane+1],phase),1))) #make a demand vector for each of the possible lanes
            #in the demand vector, we get the result after feeding the NN_demand neural network

        #Now we need to create the matrix D (paired demand embedding, for each possible phase couple)

        #First find out the demand for each of the phase
        demand_phase=[]
        for phase in range(8): #iterate on the phases
            vec=torch.zeros((1,8)).to(device)
            #for lane_number in self.possible_phase[phase]: #for a given phase, iterate on the 2 lanes of that phase
                #vec+=demand[lane_number] #sum the result obtained in demand together
            for lane in range(2):
                vec[0,lane*4:lane*4+4]=demand[self.possible_phase[phase][lane]]

            vec2=self.NN_coupled_demand(vec)
            demand_phase.append(vec2)

        #Then build the cube D
        D=torch.zeros((8, 8, 7)).to(device) #DO NOT FORGET!!!
        #(a,b,c) means a is depth, b is n_rows and c is n_columns in torch

        for i in range(8):
            for j in range(8):
                if j<i:
                    D[0:4,i,j]=demand_phase[i]
                    D[4:8,i,j]=demand_phase[j]
                elif j>i:
                    D[0:4,i,j-1]=demand_phase[i]
                    D[4:8,i,j-1]=demand_phase[j]

        #Now we do the convolution
        H_d=self.Conv_D.forward(D)
        H_c=torch.cat((H_d,comp),0)

        #Now do a convolution on H_c
        C=self.Conv_C(H_c)

        """
        #Output the Q_values vector:
        Q_values=torch.zeros((1,8)).to(device)
        for i in range(8):
            n=0
            for j in range(7):
                n+=C[0,i,j]

            Q_values[0,i]=n

        return Q_values[0,:]
        """

        return C[0,:,0]

class FRAP2(nn.Module):

    def __init__(self):
        super(FRAP2, self).__init__()

        self.network=nn.Sequential(
            nn.Linear(42, 512),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 8)
        )

    def forward(self, waiting:torch.Tensor, phase:torch.Tensor, wtime_mu:torch.Tensor, wtime_sigma:torch.Tensor, wtime_max:torch.Tensor, i_position:torch.Tensor, j_position:torch.Tensor):
        phase=phase.to(device)
        waiting=waiting.to(device)
        wtime_max=wtime_max.to(device)
        wtime_mu=wtime_mu.to(device)
        wtime_sigma=wtime_sigma.to(device)
        i_position=i_position.to(device)
        j_position=j_position.to(device)

        return self.network(torch.cat((i_position[:,:], j_position[:,:], wtime_mu[:,:], wtime_sigma[:,:], wtime_max[:,:], waiting[:,:],phase),1))[0]