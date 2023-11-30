import numpy as np
import copy

#We need the class Cars to keep track of how long each car will wait overall to measure the performances of our model
class Car: 

    def __init__(self):

        self.time=0

    def choose_next(self):

        self.time+=1
        proba=np.random.uniform(0,1)

        #These probabilities are observed in real life as stated in the Chacha paper
        #Turning right is not really interesting for now (as it does not require a traffic light)
        if proba<0.33:
            self.next=1 #left
        elif proba<0.67:
            self.next=0 #straight
        else: #More complicated, deal with turn right, they don't need the green light:
            self.next=2 #right

class Intersection4:

    def __init__(self,  mini:int, maxi:int):

        #Intersection must have 8 traffic lights and 4 turn right lanes: overall 12 lanes
        #1. Initialise basics
        self.time=0
        self.possible=[[0,4],[1,5],[2,6],[3,7],[0,5],[1,4],[2,7],[3,6]]

        self.mini=mini
        self.maxi=maxi

        #2. WAITING, EXITING & RIGHT:
        self.waiting={}
        self.n_waiting=[]
        self.exiting={}
        self.n_exiting=[0 for _ in range(4)]
        self.right={}
        self.n_right=[]

        for lane in range(8):
            n=np.random.randint(0, 3)
            self.waiting[lane]=[Car() for _ in range(n)]
            self.n_waiting.append(n)
        
        for lane in range(4):
            self.exiting[lane]=[]

        for lane in range(4):
            n=np.random.randint(0, 3)
            self.right[lane]=[Car() for _ in range(n)]
            self.n_right.append(n)

        #3. INITIATE ARBITRARILY:
        self.action=0
        self.action_summary={} 
        self.reward=0

    def new_round(self, action: int):

        self.time+=1

        #1. Initialise actions recording:
        self.action=action
        self.action_summary={}

        L=[]
        for i in range(4):
            L.append(len(self.exiting[i]))
        #print("beginning", L)
        
        #2. UPDATE: from waiting to exiting for lanes with green light:
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

            #3. 3 different cases:
            #3.1 there are less cars waiting than mini: all leave to exiting
            if self.n_waiting[lane]<self.mini:
                #print("before 0", len(self.exiting[exit_lane]))
                self.exiting[exit_lane]+=self.waiting[lane]
                
                self.action_summary[lane]=self.waiting[lane]
                self.waiting[lane]=[]
                #print("after 0", len(self.exiting[exit_lane]))
            
            #3.2 cars waiting in lane are between mini and maxi: randomly pick a number of cars moving through between mini and maxi
            elif self.n_waiting[lane]<self.maxi:
                #print("before 1", len(self.exiting[exit_lane]))
                n=np.random.randint(self.mini, self.n_waiting[lane]+1)

                self.exiting[exit_lane]+=self.waiting[lane][0:n]

                self.action_summary[lane]=self.waiting[lane][0:n]

                waiting=copy.deepcopy(self.waiting[lane][n:])
                self.waiting[lane]=waiting
                #print("after 1", len(self.exiting[exit_lane]))
            
            #3.3 just pick a random number if there are more than maxi
            else:
                #print("before 2", len(self.exiting[exit_lane]))

                n=np.random.randint(self.mini, self.maxi+1)

                self.exiting[exit_lane]+=self.waiting[lane][0:n]

                self.action_summary[lane]=self.waiting[lane][0:n]

                waiting=copy.deepcopy(self.waiting[lane][n:])
                self.waiting[lane]=waiting

                #print("after 2", len(self.exiting[exit_lane]))



    def turning_right(self):

        #1. Make cars turn right

        for lane in range(4):

            if self.n_right[lane]<self.mini:

                self.exiting[(lane-1)%4]+=self.right[lane]

                self.action_summary[lane+8]=self.right[lane]
                self.right[lane]=[]

            elif self.n_right[lane]<self.maxi:

                n=np.random.randint(self.mini, self.n_right[lane]+1)

                self.exiting[(lane-1)%4]+=self.right[lane][0:n]

                self.action_summary[lane+8]=self.right[lane][0:n]

                right=copy.deepcopy(self.right[lane][n:])
                self.right[lane]=right

            else:

                n=np.random.randint(self.mini, self.maxi)

                self.exiting[(lane-1)%4]+=self.right[lane][0:n]

                self.action_summary[lane+8]=self.right[lane][0:n]

                right=copy.deepcopy(self.right[lane][n:])
                self.right[lane]=right


    def rewarding(self):

        self.reward=-np.sum(self.n_waiting)+np.sum(self.n_exiting)

        #Penalty if reward is too low: LOSE THE GAME (individually)

        if self.reward<-80:
            self.reward=-150

    def equilibrate(self):

        for lane in range(4):

            self.n_exiting[lane]=len(self.exiting[lane])
            self.n_right[lane]=len(self.right[lane])

        for lane in range(8):

            self.n_waiting[lane]=len(self.waiting[lane])