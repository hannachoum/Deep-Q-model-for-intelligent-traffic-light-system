import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy


#CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation



#DROPOUT IS IMPLEMENTED HERE WITH 0.3
drop=0.05

class NN_vehicles(nn.Module):

    def __init__(self):
        super(NN_vehicles, self).__init__()

        self.network=nn.Sequential(
            nn.Linear(1, 12),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(12, 4),
        )

    def forward(self,n_cars):
        return self.network(n_cars)

class NN_phase(nn.Module):

    def __init__(self):
        super(NN_phase, self).__init__()

        self.network=nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(16, 4),
        )

    def forward(self,phase_vec):
        return self.network(phase_vec)

class NN_demand(nn.Module):

    def __init__(self):
        super(NN_demand, self).__init__()

        self.network=nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(16, 4),
        )

    def forward(self,conc_demand):
        return self.network(conc_demand)

class Conv_D(nn.Module):

    def __init__(self):
        super(Conv_D, self).__init__()

        self.conv1=nn.Conv2d(8,4,1)
        self.conv2=nn.Conv2d(4,2,1)
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(drop)

    def forward(self,D):
        conv1=self.dropout(self.relu(self.conv1(D)))
        #maybe later put activation function for the last layer (tanh, sigmoid...)
        return self.conv2(conv1)

class Conv_E(nn.Module):

    def __init__(self):
        super(Conv_E, self).__init__()

        self.conv1=nn.Conv2d(1,8,1)
        self.conv2=nn.Conv2d(8,2,1)
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(drop)

    def forward(self,E):
        conv1=self.dropout(self.relu(self.conv1(E)))
        #maybe later put activation function for the last layer (tanh, sigmoid...)
        return self.conv2(conv1)

class Conv_C(nn.Module):

    def __init__(self):
        super(Conv_C, self).__init__()

        self.conv1=nn.Conv2d(2,4,1)
        self.conv2=nn.Conv2d(4,1,1)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
        self.dropout = nn.Dropout(drop)

    def forward(self,H_c):
        conv1=self.dropout(self.relu(self.conv1(H_c)))
        #maybe later put activation function for the last layer (tanh, sigmoid...)
        return self.conv2(conv1)

class FRAP3(nn.Module):

    def __init__(self):
        
        super(FRAP3, self).__init__()

        #define all possible phases (combinations of lanes that can get green light together)
        self.possible_phase=[[0,4],[1,5],[2,6],[3,7],[0,5],[1,4],[2,7],[3,6]]
        #phase is a couple of 2 lanes that can get green light together
        #phase NEEDS to be an 8-vector with 0 for red light, 1 for green

        #Define the neural networks
        self.NN_cars=NN_vehicles()
        self.NN_phase=NN_phase()
        self.NN_demand=NN_demand()
        self.Conv_D=Conv_D()
        self.Conv_E=Conv_E()
        self.Conv_C=Conv_C()


    def forward(self, waiting:torch.Tensor, phase:torch.Tensor):

        phase=phase.to(device)

        #Make a list after feeding forward into the NN_cars network (to reduce computation effort later)
        demand=[]
        n_phase=self.NN_phase.forward(phase)

        wait=copy.deepcopy(waiting)

        for lane in range(8):
            #LATER MAYBE ADD LOCATION
            n_cars=self.NN_cars.forward(wait[:,lane:lane+1])#find a demand vector (feed the network with a torch tensor!!)
            demand.append(self.NN_demand(torch.cat((n_cars,n_phase),1))) #make a demand vector for each of the possible lanes
            #in the demand vector, we get the result after feeding the NN_demand neural network

        #Now we need to create the matrix D (paired demand embedding, for each possible phase couple)

        #First find out the demand for each of the phase
        demand_phase=[]
        for phase in range(8): #iterate on the phases
            vec=torch.zeros((1,4)).to(device)
            for lane_number in self.possible_phase[phase]: #for a given phase, iterate on the 2 lanes of that phase
                vec+=demand[lane_number] #sum the result obtained in demand together
            demand_phase.append(vec)

        #Then build the cube D
        D=torch.zeros((8, 8, 7)).to(device) #DO NOT FORGET!!!
        #(a,b,c) means a is depth, b is n_rows and c is n_columns in torch

        for i in range(8):
            for j in range(7):
                if i!=j: #do not take the same phase
                    #Here we fill in D, to concatenate couples of phases:
                    D[0:4,i,j]=demand_phase[i]
                    D[4:8,i,j]=demand_phase[j]

        #Now we do the convolution
        H_d=self.Conv_D.forward(D)

        #Define the phase competition mask:
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
            for  j in range(7):
                if j<i:
                    comp[0,i,j]=competition[i,j]
                else:
                    comp[0,i,j]=competition[i,j+1]

        #Make E, the relation embedding volume, from the competition matrix:
        H_r=self.Conv_E(comp)

        #Finally, make H_c:
        H_c=torch.mul(H_d, H_r)

        #Now do a convolution on H_c
        C=self.Conv_C(H_c)

        #Output the Q_values vector:
        Q_values=torch.zeros((1,8)).to(device)
        for i in range(8):
            n=0
            for j in range(7):
                n+=C[0,i,j]

            Q_values[0,i]=n

        return Q_values[0,:]