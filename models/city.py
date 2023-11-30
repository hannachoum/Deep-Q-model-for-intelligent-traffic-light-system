import numpy as np
import copy
from cross import *

class City:

    def __init__(self, N:int, M:int, mini:int, maxi:int):
        self.N=N
        self.M=M

        #To add new cars entering into the city
        self.mini=mini
        self.maxi=maxi+1 #To have it included

    def update_exit_boundary(self, intersections:dict, i, j):

        #Needs to be located after the next_round() of intersection
        #1. Previous cars need to leave the city:
        if i==0:
            intersections[(i,j)].exiting[0]=0
        if j==self.M-1:
            intersections[(i,j)].exiting[1]=0
        if i==self.N-1:
            intersections[(i,j)].exiting[2]=0
        if j==0:
            intersections[(i,j)].exiting[3]=0
            

        for lane in intersections[(i,j)].action_summary.keys():

            #2. Refill the exiting lanes with the cars entering the exit lanes:
            moving=intersections[(i,j)].action_summary[lane]

            #Then put back the cars leaving the city from the action summary:
            
            if i==0 and (lane==5 or lane==6 or lane==9):
                intersections[(i,j)].exiting[0]+=moving

            if j==self.M-1 and (lane==7 or lane==0 or lane==10):
                intersections[(i,j)].exiting[1]+=moving
                
            if i==self.N-1 and (lane==2 or lane==1 or lane==11):
                intersections[(i,j)].exiting[2]+=moving

            if j==0 and (lane==4 or lane==3 or lane==8):
                intersections[(i,j)].exiting[3]+=moving


    def harmonise(self, intersections:dict, i, j):
        #In this function we harmonise the results of next round for one intersection in the city (update waiting and exiting)
        
        for lane in intersections[(i,j)].action_summary.keys():

            moving=intersections[(i,j)].action_summary[lane]
        
            #Using moving and n, update exiting and waiting of adjacent intersections
            #This only matters if the lane is not at the edge of the city:

            #1. Update the exiting lane:

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

            #2. Update the waiting lane:

            if lane==5 or lane==6 or lane==9:
                if i-1>=0: #In this case, cars are in exiting[0] of intersection (i,j)
                    #Now we need to put them into waiting in lane 6,7,9, in (i-1,j)
                    for _ in range(moving):
                        proba=np.random.randint(0,3)
                        if proba==0:
                            intersections[(i-1,j)].waiting[6]+=1
                            intersections[(i-1,j)].wtime[6].append(0) #Add a new car that has waiting 0 (same after)
                        elif proba==1:
                            intersections[(i-1,j)].waiting[3]+=1
                            intersections[(i-1,j)].wtime[3].append(0)
                        else:
                            intersections[(i-1,j)].right[2]+=1
                            intersections[(i-1,j)].wtime[10].append(0)

            elif lane==7 or lane==0 or lane==10:
                if j+1<self.M:
                    for _ in range(moving):
                        proba=np.random.randint(0,3)
                        if proba==0:
                            intersections[(i,j+1)].waiting[0]+=1
                            intersections[(i,j+1)].wtime[0].append(0)
                        elif proba==1:
                            intersections[(i,j+1)].waiting[5]+=1
                            intersections[(i,j+1)].wtime[5].append(0)
                        else:
                            intersections[(i,j+1)].right[3]+=1
                            intersections[(i,j+1)].wtime[11].append(0)

            elif lane==1 or lane==2 or lane==11:
                if i+1<self.N:
                    for _ in range(moving):
                        proba=np.random.randint(0,3)
                        if proba==0:
                            intersections[(i+1,j)].waiting[2]+=1
                            intersections[(i+1,j)].wtime[2].append(0)
                        elif proba==1:
                            intersections[(i+1,j)].waiting[7]+=1
                            intersections[(i+1,j)].wtime[7].append(0)
                        else:
                            intersections[(i+1,j)].right[0]+=1
                            intersections[(i+1,j)].wtime[8].append(0)

            else:
                if j-1>=0:
                    for _ in range(moving):
                        proba=np.random.randint(0,3)
                        if proba==0:
                            intersections[(i,j-1)].waiting[4]+=1
                            intersections[(i,j-1)].wtime[4].append(0)
                        elif proba==1:
                            intersections[(i,j-1)].waiting[1]+=1
                            intersections[(i,j-1)].wtime[1].append(0)
                        else:
                            intersections[(i,j-1)].right[1]+=1
                            intersections[(i,j-1)].wtime[9].append(0)


    def add_at_boundary(self, intersections:dict, i, j):

        #1. ADD at each boundaries:
        if i==0:

            #1.1 How many cars are entering
            n_new1=np.random.randint(self.mini,self.maxi)
            n_new2=np.random.randint(self.mini,self.maxi)
            n_new3=np.random.randint(self.mini,self.maxi)

            #1.2 Add into the waiting
            intersections[(i,j)].waiting[2]+=n_new1
            intersections[(i,j)].waiting[7]+=n_new2
            intersections[(i,j)].right[0]+=n_new3

            #1.3 Add into the waiting time at the intersection
            for _ in range(n_new1):
                intersections[(i,j)].wtime[2].append(0)
            for _ in range(n_new2):
                intersections[(i,j)].wtime[7].append(0)
            for _ in range(n_new3):
                intersections[(i,j)].wtime[8].append(0)

        if i==self.N-1:
            n_new1=np.random.randint(self.mini,self.maxi)
            n_new2=np.random.randint(self.mini,self.maxi)
            n_new3=np.random.randint(self.mini,self.maxi)

            intersections[(i,j)].waiting[6]+=n_new1
            intersections[(i,j)].waiting[3]+=n_new2
            intersections[(i,j)].right[2]+=n_new3

            for _ in range(n_new1):
                intersections[(i,j)].wtime[6].append(0)
            for _ in range(n_new2):
                intersections[(i,j)].wtime[3].append(0)
            for _ in range(n_new3):
                intersections[(i,j)].wtime[10].append(0)

        if j==0:
            n_new1=np.random.randint(self.mini,self.maxi)
            n_new2=np.random.randint(self.mini,self.maxi)
            n_new3=np.random.randint(self.mini,self.maxi)

            intersections[(i,j)].waiting[0]+=n_new1
            intersections[(i,j)].waiting[5]+=n_new2
            intersections[(i,j)].right[3]+=n_new3

            for _ in range(n_new1):
                intersections[(i,j)].wtime[0].append(0)
            for _ in range(n_new2):
                intersections[(i,j)].wtime[5].append(0)
            for _ in range(n_new3):
                intersections[(i,j)].wtime[11].append(0)
                
        if j==self.M-1:
            n_new1=np.random.randint(self.mini,self.maxi)
            n_new2=np.random.randint(self.mini,self.maxi)
            n_new3=np.random.randint(self.mini,self.maxi)

            intersections[(i,j)].waiting[1]+=n_new1
            intersections[(i,j)].waiting[4]+=n_new2
            intersections[(i,j)].right[1]+=n_new3

            for _ in range(n_new1):
                intersections[(i,j)].wtime[1].append(0)
            for _ in range(n_new2):
                intersections[(i,j)].wtime[4].append(0)
            for _ in range(n_new3):
                intersections[(i,j)].wtime[9].append(0)

    def set_harmony(self,  intersections:dict):
    #See it as initial harmonisation
    #After resetting the game, we must make each intersection aware of the others:

        for i in range(self.N):
            for j in range(self.M):

                #Update exiting:
                if i!=0:
                    intersections[(i,j)].exiting[0]=intersections[(i-1,j)].waiting[3]+intersections[(i-1,j)].waiting[6]+intersections[(i-1,j)].right[2]
                if j!=self.M-1:
                    intersections[(i,j)].exiting[1]=intersections[(i,j+1)].waiting[0]+intersections[(i,j+1)].waiting[5]+intersections[(i,j+1)].right[3]
                if i!=self.N-1:
                    intersections[(i,j)].exiting[2]=intersections[(i+1,j)].waiting[2]+intersections[(i+1,j)].waiting[7]+intersections[(i+1,j)].right[0]
                if j!=0:
                    intersections[(i,j)].exiting[3]=intersections[(i,j-1)].waiting[1]+intersections[(i,j-1)].waiting[4]+intersections[(i,j-1)].right[1]