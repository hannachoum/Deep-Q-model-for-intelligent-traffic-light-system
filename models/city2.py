import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from intersection2 import *

class City5:

    def __init__(self, N:int, M:int, mini_in:int, maxi_in:int):

        self.N=N
        self.M=M
        self.mini_in=mini_in
        self.maxi_in=maxi_in

    def exit_boundary(self, intersections:dict, i:int, j:int):

        #1. Update exit boundary:
        #1.1 Set exits at boundary to 0:
        if i==0:
            intersections[(i,j)].exiting[0]=0
        if j==self.M-1:
            intersections[(i,j)].exiting[1]=0
        if i==self.N-1:
            intersections[(i,j)].exiting[2]=0
        if j==0:
            intersections[(i,j)].exiting[3]=0

        #1.2 Refill back with cars newly moving from the action that has just been undertaken
        for lane in intersections[(i,j)].action_summary.keys():

            moving=int(intersections[(i,j)].action_summary[lane])

            if i==0 and (lane==5 or lane==6 or lane==9):
                intersections[(i,j)].exiting[0]+=moving

            if j==self.M-1 and (lane==7 or lane==0 or lane==10):
                intersections[(i,j)].exiting[1]+=moving
                
            if i==self.N-1 and (lane==2 or lane==1 or lane==11):
                intersections[(i,j)].exiting[2]+=moving

            if j==0 and (lane==4 or lane==3 or lane==8):
                intersections[(i,j)].exiting[3]+=moving

    def harmonise(self, intersections:dict, i:int, j:int):

        #2. Harmonise actions undertaken by all intersections:
        for lane in intersections[(i,j)].action_summary.keys():

            moving=int(intersections[(i,j)].action_summary[lane]) #int

            #2.1 Update exiting:
            if lane==0 or lane==5 or lane==11:
                if j-1>=0: #If intersection (i,j-1) exists:
                    intersections[(i,j-1)].exiting[1]-=moving

            elif lane==3 or lane==6 or lane==10:
                if i-1>=0:
                    intersections[(i-1,j)].exiting[2]-=moving
                
            elif lane==1 or lane==4 or lane==9:
                if j+1<self.M:
                    intersections[(i,j+1)].exiting[3]-=moving

            else:
                if i+1<self.N:
                    intersections[(i+1,j)].exiting[0]-=moving

            #2.2 Update waiting:
            if lane==5 or lane==6 or lane==9:
                if i-1>=0: #In this case, cars are in exiting[0] of intersection (i,j)
                    #Now we need to put them into waiting in lane 6,7,9, in (i-1,j)
                    for _ in range(moving):
                        proba=np.random.randint(0,3)
                        if proba==0:
                            intersections[(i-1,j)].waiting[6]+=1
                            #Add a new car that has waiting 0 (same after)
                            intersections[(i-1,j)].wtime[6]=torch.cat((intersections[(i-1,j)].wtime[6], torch.zeros(1)))
                        elif proba==1:
                            intersections[(i-1,j)].waiting[3]+=1
                            intersections[(i-1,j)].wtime[3]=torch.cat((intersections[(i-1,j)].wtime[3], torch.zeros(1)))
                        else:
                            intersections[(i-1,j)].waiting[10]+=1
                            intersections[(i-1,j)].wtime[10]=torch.cat((intersections[(i-1,j)].wtime[10], torch.zeros(1)))

            elif lane==7 or lane==0 or lane==10:
                if j+1<self.M:
                    for _ in range(moving):
                        proba=np.random.randint(0,3)
                        if proba==0:
                            intersections[(i,j+1)].waiting[0]+=1
                            intersections[(i,j+1)].wtime[0]=torch.cat((intersections[(i,j+1)].wtime[0], torch.zeros(1)))
                        elif proba==1:
                            intersections[(i,j+1)].waiting[5]+=1
                            intersections[(i,j+1)].wtime[5]=torch.cat((intersections[(i,j+1)].wtime[5], torch.zeros(1)))
                        else:
                            intersections[(i,j+1)].waiting[11]+=1
                            intersections[(i,j+1)].wtime[11]=torch.cat((intersections[(i,j+1)].wtime[11], torch.zeros(1)))

            elif lane==1 or lane==2 or lane==11:
                if i+1<self.N:
                    for _ in range(moving):
                        proba=np.random.randint(0,3)
                        if proba==0:
                            intersections[(i+1,j)].waiting[2]+=1
                            intersections[(i+1,j)].wtime[2]=torch.cat((intersections[(i+1,j)].wtime[2], torch.zeros(1)))
                        elif proba==1:
                            intersections[(i+1,j)].waiting[7]+=1
                            intersections[(i+1,j)].wtime[7]=torch.cat((intersections[(i+1,j)].wtime[7], torch.zeros(1)))
                        else:
                            intersections[(i+1,j)].waiting[8]+=1
                            intersections[(i+1,j)].wtime[8]=torch.cat((intersections[(i+1,j)].wtime[8], torch.zeros(1)))

            else:
                if j-1>=0:
                    for _ in range(moving):
                        proba=np.random.randint(0,3)
                        if proba==0:
                            intersections[(i,j-1)].waiting[4]+=1
                            intersections[(i,j-1)].wtime[4]=torch.cat((intersections[(i,j-1)].wtime[4], torch.zeros(1)))
                        elif proba==1:
                            intersections[(i,j-1)].waiting[1]+=1
                            intersections[(i,j-1)].wtime[1]=torch.cat((intersections[(i,j-1)].wtime[1], torch.zeros(1)))
                        else:
                            intersections[(i,j-1)].waiting[9]+=1
                            intersections[(i,j-1)].wtime[9]=torch.cat((intersections[(i,j-1)].wtime[9], torch.zeros(1)))

    def waiting_boundary(self, intersections:dict, i:int, j:int):

        #3. ADD at each boundaries:
        if i==0:

            #3.1 How many cars are entering
            n_new1=np.random.randint(self.mini_in,self.maxi_in+1)
            n_new2=np.random.randint(self.mini_in,self.maxi_in+1)
            n_new3=np.random.randint(self.mini_in,self.maxi_in+1)

            #3.2 Add into the waiting
            intersections[(i,j)].waiting[2]+=n_new1
            intersections[(i,j)].waiting[7]+=n_new2
            intersections[(i,j)].waiting[8]+=n_new3

            #3.3 Add into the waiting time at the intersection
            if n_new1>0:
                intersections[(i,j)].wtime[2]=torch.cat((intersections[(i,j)].wtime[2], torch.zeros(n_new1)))
            if n_new2>0:
                intersections[(i,j)].wtime[7]=torch.cat((intersections[(i,j)].wtime[7], torch.zeros(n_new2)))
            if n_new3>0:
                intersections[(i,j)].wtime[8]=torch.cat((intersections[(i,j)].wtime[8], torch.zeros(n_new3)))

        if i==self.N-1:
            n_new1=np.random.randint(self.mini_in,self.maxi_in+1)
            n_new2=np.random.randint(self.mini_in,self.maxi_in+1)
            n_new3=np.random.randint(self.mini_in,self.maxi_in+1)

            intersections[(i,j)].waiting[6]+=n_new1
            intersections[(i,j)].waiting[3]+=n_new2
            intersections[(i,j)].waiting[10]+=n_new3

            if n_new1>0:
                intersections[(i,j)].wtime[6]=torch.cat((intersections[(i,j)].wtime[6], torch.zeros(n_new1)))
            if n_new2>0:
                intersections[(i,j)].wtime[3]=torch.cat((intersections[(i,j)].wtime[3], torch.zeros(n_new2)))
            if n_new3>0:
                intersections[(i,j)].wtime[10]=torch.cat((intersections[(i,j)].wtime[10], torch.zeros(n_new3)))

        if j==0:
            n_new1=np.random.randint(self.mini_in,self.maxi_in+1)
            n_new2=np.random.randint(self.mini_in,self.maxi_in+1)
            n_new3=np.random.randint(self.mini_in,self.maxi_in+1)

            intersections[(i,j)].waiting[0]+=n_new1
            intersections[(i,j)].waiting[5]+=n_new2
            intersections[(i,j)].waiting[11]+=n_new3

            if n_new1>0:
                intersections[(i,j)].wtime[0]=torch.cat((intersections[(i,j)].wtime[0], torch.zeros(n_new1)))
            if n_new2>0:
                intersections[(i,j)].wtime[5]=torch.cat((intersections[(i,j)].wtime[5], torch.zeros(n_new2)))
            if n_new3>0:
                intersections[(i,j)].wtime[11]=torch.cat((intersections[(i,j)].wtime[11], torch.zeros(n_new3)))
                
        if j==self.M-1:
            n_new1=np.random.randint(self.mini_in,self.maxi_in+1)
            n_new2=np.random.randint(self.mini_in,self.maxi_in+1)
            n_new3=np.random.randint(self.mini_in,self.maxi_in+1)

            intersections[(i,j)].waiting[1]+=n_new1
            intersections[(i,j)].waiting[4]+=n_new2
            intersections[(i,j)].waiting[9]+=n_new3

            if n_new1>0:
                intersections[(i,j)].wtime[1]=torch.cat((intersections[(i,j)].wtime[1], torch.zeros(n_new1)))
            if n_new2>0:
                intersections[(i,j)].wtime[4]=torch.cat((intersections[(i,j)].wtime[4], torch.zeros(n_new2)))
            if n_new3>0:
                intersections[(i,j)].wtime[9]=torch.cat((intersections[(i,j)].wtime[9], torch.zeros(n_new3)))

    def set_harmony(self,  intersections:dict):
    #See it as initial harmonisation
    #After resetting the game, we must make each intersection aware of the others:

        for i in range(self.N):
            for j in range(self.M):

                #Update exiting:
                if i!=0:
                    intersections[(i,j)].exiting[0]=intersections[(i-1,j)].waiting[3]+intersections[(i-1,j)].waiting[6]+intersections[(i-1,j)].waiting[10]
                if j!=self.M-1:
                    intersections[(i,j)].exiting[1]=intersections[(i,j+1)].waiting[0]+intersections[(i,j+1)].waiting[5]+intersections[(i,j+1)].waiting[11]
                if i!=self.N-1:
                    intersections[(i,j)].exiting[2]=intersections[(i+1,j)].waiting[2]+intersections[(i+1,j)].waiting[7]+intersections[(i+1,j)].waiting[8]
                if j!=0:
                    intersections[(i,j)].exiting[3]=intersections[(i,j-1)].waiting[1]+intersections[(i,j-1)].waiting[4]+intersections[(i,j-1)].waiting[9]
        