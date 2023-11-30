import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Crossing5:

    def __init__(self, mini_in:int, maxi_in:int, mini_thru:int, maxi_thru:int, i:int, j:int):

        self.mini_in=mini_in
        self.maxi_in=maxi_in
        self.mini_thru=mini_thru
        self.maxi_thru=maxi_thru
        self.position=(i,j)

        #1. Initialise waiting cars + age in lane:
        self.waiting=torch.zeros(12)
        self.wtime={}
        for lane in range(12):
            self.wtime[lane]=torch.zeros(0)
            self.waiting[lane]=np.random.randint(self.mini_in, self.maxi_in+1)

        #2. Exiting cars
        self.exiting=torch.zeros(4)

        #3. Lane combinations for action, summary of actions taken:
        self.possible=[[0,4],[1,5],[2,6],[3,7],[0,5],[1,4],[2,7],[3,6]]
        self.action_summary={}

        #4. Initialise reward:
        self.reward=0
    

    def next_round(self, action:int):

        self.action_summary={}

        for lane in self.possible[action]:

            #Need to define the lane cars will go to when light turn green for their lane
            if lane==5 or lane==6:
                exit_lane=0
            elif lane==0 or lane==7:
                exit_lane=1
            elif lane==1 or lane==2:
                exit_lane=2
            else:
                exit_lane=3

            #1. 3 different cases:
            #1.1 there are less cars waiting than mini: all leave to exiting
            if self.waiting[lane]<self.mini_thru:

                n=int(self.waiting[lane])

                self.exiting[exit_lane]+=n
                self.action_summary[lane]=n
                self.waiting[lane]=0
                self.wtime[lane]=torch.zeros(0)
            
            #1.2 cars waiting in lane are between mini and maxi: randomly pick a number of cars moving through between mini and maxi
            elif self.waiting[lane]<self.maxi_thru:

 
                n=np.random.randint(self.mini_thru, self.waiting[lane]+1)

                self.exiting[exit_lane]+=n
                self.action_summary[lane]=n
                self.waiting[lane]-=n

                self.wtime[lane]=torch.add(self.wtime[lane][n:], 1) 
                
            #1.3 just pick a random number if there are more than maxi
            else:


                n=np.random.randint(self.mini_thru, self.maxi_thru+1)

                self.exiting[exit_lane]+=n
                self.action_summary[lane]=n
                self.waiting[lane]-=n

                self.wtime[lane]=torch.add(self.wtime[lane][n:], 1)

        #2. Add age +1 to all other cars that did not get the green light
        for lane in range(8):
            if lane not in self.possible[action]:
                self.wtime[lane]=torch.add(self.wtime[lane], 1)

        #3. Make cars turn right
        for l in range(4):

            lane=l+8

            if self.waiting[lane]<self.mini_thru:

                n=int(self.waiting[lane])

                self.exiting[(l-1)%4]+=n
                self.action_summary[lane]=n
                self.waiting[lane]=0
                self.wtime[lane]=torch.zeros(0)

            elif self.waiting[lane]<self.maxi_thru:

                n=np.random.randint(self.mini_thru, self.waiting[lane]+1)

                self.exiting[(l-1)%4]+=n
                self.action_summary[lane]=n
                self.waiting[lane]-=n
                self.wtime[lane]=torch.add(self.wtime[lane][n:], 1) 

            else:

                n=np.random.randint(self.mini_thru, self.maxi_thru+1)

                self.exiting[(lane-1)%4]+=n
                self.action_summary[lane]=n
                self.waiting[lane]-=n
                self.wtime[lane]=torch.add(self.wtime[lane][n:], 1)

        #print(self.exiting)
        #print(self.waiting)
        #print(self.action_summary)



    def rewarding(self):

        #REWARD:
        #1. PRESSURE
        #pressure=torch.sum(self.exiting)-torch.sum(self.waiting)

        #2. MAX TIME a car has been waiting
        total_wait=0

        for lane in range(12):
            if self.wtime[lane].shape[0]!=0:
                total_wait+=torch.sum(self.wtime[lane])

        self.reward=-total_wait*0.3333+torch.sum(self.exiting)*0.05

        if self.reward<-100:

            self.reward=-120
        




