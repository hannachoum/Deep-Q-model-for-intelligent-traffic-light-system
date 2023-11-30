import numpy as np
import copy
from city_cross2 import *

class City2:

    def __init__(self, N:int, M:int, mini:int, maxi:int):
        self.N=N
        self.M=M

        #To add new cars entering into the city
        self.mini=mini
        self.maxi=maxi+1 #To have it included

    def update_exit_boundary(self, intersections:dict, i, j):

        #1. Previous cars need to leave the city:
        if i==0:
            intersections[(i,j)].exiting[0]=[]
        if j==self.M-1:
            intersections[(i,j)].exiting[1]=[]
        if i==self.N-1:
            intersections[(i,j)].exiting[2]=[]
        if j==0:
            intersections[(i,j)].exiting[3]=[]
            

        #print(intersections[(i,j)].action_summary)
        for lane in intersections[(i,j)].action_summary.keys():

            #2. Refill the exiting lanes with the cars entering the exit lanes:
            moving=intersections[(i,j)].action_summary[lane]

            #Then put back the cars leaving the city from the action summary:
            
            if i==0 and (lane==5 or lane==6 or lane==9):
                for car in moving:
                    intersections[(i,j)].exiting[0].append(car)

            if j==self.M-1 and (lane==7 or lane==0 or lane==10):
                for car in moving:
                    intersections[(i,j)].exiting[1].append(car)
                
            if i==self.N-1 and (lane==2 or lane==1 or lane==11):
                for car in moving:
                    intersections[(i,j)].exiting[2].append(car)

            if j==0 and (lane==4 or lane==3 or lane==8):
                for car in moving:
                    intersections[(i,j)].exiting[3].append(car)


    def harmonise(self, intersections:dict, i, j):
        #In this function we harmonise the results of next round for one intersection in the city (update waiting and exiting)
        

        for lane in intersections[(i,j)].action_summary.keys():

            #n is the number of cars moving through the intersection from lane lane:
            moving=intersections[(i,j)].action_summary[lane]
            n=len(moving)
        
            #Using moving and n, update exiting and waiting of adjacent intersections
            #This only matters if the lane is not at the edge of the city:

            #1. Update the exiting lane:

            if lane==0 or lane==5 or lane==11:
                if j-1>=0: #If intersection (i,j-1) exists:
                    exiting=intersections[(i,j-1)].exiting[1][n:] #n first cars are going away through moving so keep the last ones still waiting in (i,j)
                    #And thus still exiting in (i,j-1)
                    intersections[(i,j-1)].exiting[1]=exiting

            elif lane==3 or lane==6 or lane==10:
                if i-1>=0:
                    exiting=intersections[(i-1,j)].exiting[2][n:]
                    intersections[(i-1,j)].exiting[2]=exiting
                
            elif lane==1 or lane==4 or lane==9:
                if j+1<self.M:
                    exiting=intersections[(i,j+1)].exiting[3][n:]
                    intersections[(i,j+1)].exiting[3]=exiting

            else:
                if i+1<self.N:
                    exiting=intersections[(i+1,j)].exiting[0][n:]
                    intersections[(i+1,j)].exiting[0]=exiting

            #2. Update the waiting lane:

            if lane==5 or lane==6 or lane==9:
                if i-1>=0: #In this case, cars are in exiting[0] of intersection (i,j)
                    #Now we need to put them into waiting in lane 6,7,9, in (i-1,j)
                    for car in moving:
                        car.choose_next()
                        if car.next==0:
                            intersections[(i-1,j)].waiting[6].append(car)
                        elif car.next==1:
                            intersections[(i-1,j)].waiting[3].append(car)
                        else:
                            intersections[(i-1,j)].right[2].append(car)

            elif lane==7 or lane==0 or lane==10:
                if j+1<self.M:
                    for car in moving:
                        car.choose_next()
                        if car.next==0:
                            intersections[(i,j+1)].waiting[0].append(car)
                        elif car.next==1:
                            intersections[(i,j+1)].waiting[5].append(car)
                        else:
                            intersections[(i,j+1)].right[3].append(car)

            elif lane==1 or lane==2 or lane==11:
                if i+1<self.N:
                    for car in moving:
                        car.choose_next()
                        if car.next==0:
                            intersections[(i+1,j)].waiting[2].append(car)
                        elif car.next==1:
                            intersections[(i+1,j)].waiting[7].append(car)
                        else:
                            intersections[(i+1,j)].right[0].append(car)

            else:
                if j-1>=0:
                    for car in moving:
                        car.choose_next()
                        if car.next==0:
                            intersections[(i,j-1)].waiting[4].append(car)
                        elif car.next==1:
                            intersections[(i,j-1)].waiting[1].append(car)
                        else:
                            intersections[(i,j-1)].right[1].append(car)



    def add_at_boundary(self, intersections:dict, i, j):
        #1. ADD at each boundaries:
        if i==0:
            n_new1=np.random.randint(self.mini,self.maxi)
            n_new2=np.random.randint(self.mini,self.maxi)
            n_new3=np.random.randint(self.mini,self.maxi)

            for new in range(n_new1):
                intersections[(i,j)].waiting[2].append(Car())
            for new in range(n_new2):
                intersections[(i,j)].waiting[7].append(Car())
            for new in range(n_new3):
                intersections[(i,j)].right[0].append(Car())


        if i==self.N-1:
            n_new1=np.random.randint(self.mini,self.maxi)
            n_new2=np.random.randint(self.mini,self.maxi)
            n_new3=np.random.randint(self.mini,self.maxi)

            for new in range(n_new1):
                intersections[(i,j)].waiting[6].append(Car())
            for new in range(n_new2):
                intersections[(i,j)].waiting[3].append(Car())
            for new in range(n_new3):
                intersections[(i,j)].right[2].append(Car())

        if j==0:
            n_new1=np.random.randint(self.mini,self.maxi)
            n_new2=np.random.randint(self.mini,self.maxi)
            n_new3=np.random.randint(self.mini,self.maxi)

            for new in range(n_new1):
                intersections[(i,j)].waiting[0].append(Car())
            for new in range(n_new2):
                intersections[(i,j)].waiting[5].append(Car())
            for new in range(n_new3):
                intersections[(i,j)].right[3].append(Car())

                
        if j==self.M-1:
            n_new1=np.random.randint(self.mini,self.maxi)
            n_new2=np.random.randint(self.mini,self.maxi)
            n_new3=np.random.randint(self.mini,self.maxi)

            for new in range(n_new1):
                intersections[(i,j)].waiting[1].append(Car())
            for new in range(n_new2):
                intersections[(i,j)].waiting[4].append(Car())
            for new in range(n_new3):
                intersections[(i,j)].right[1].append(Car())
            

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


    def get_neighbors(self, i, j):

        neighb=[1 for _ in range(8)]
        if i==0:
            neighb[2]=0
            neighb[7]=0
        elif i==self.N-1:
            neighb[3]=0
            neighb[6]=0

        if j==0:
            neighb[0]=0
            neighb[5]=0
        elif j==self.M-1:
            neighb[1]=0
            neighb[4]=0

        return neighb