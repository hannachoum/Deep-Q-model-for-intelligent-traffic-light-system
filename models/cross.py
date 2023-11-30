import numpy as np
import copy


class Intersection:

    def __init__(self,  mini:int, maxi:int):

        #Intersection must have 8 traffic lights and 4 turn right lanes: overall 12 lanes
        #1. Initialise basics
        self.time=0
        self.possible=[[0,4],[1,5],[2,6],[3,7],[0,5],[1,4],[2,7],[3,6]]

        self.mini=mini
        self.maxi=maxi

        #2. WAITING, EXITING & RIGHT:
        self.waiting=[0 for _ in range(8)]
        self.exiting=[0 for _ in range(4)]
        self.right=[0 for _ in range(4)]

        #3. WAITING TIME AT INTERSECTION:
        self.wtime={}
        for lane in range(12):
            self.wtime[lane]=[] #Account for all possible lanes

        #4. INITIATE ARBITRARILY:
        self.action=0
        self.action_summary={} 
        self.reward=0

    def new_round(self, action: int):

        self.time+=1

        #1. Initialise actions recording:
        self.action=action
        self.action_summary={}
        
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
            if self.waiting[lane]<self.mini:
          
                self.exiting[exit_lane]+=self.waiting[lane]
                self.action_summary[lane]=self.waiting[lane]
                self.waiting[lane]=0

                self.wtime[lane]=[]
            
            #3.2 cars waiting in lane are between mini and maxi: randomly pick a number of cars moving through between mini and maxi
            elif self.waiting[lane]<self.maxi:
 
                n=np.random.randint(self.mini, self.waiting[lane]+1)

                self.exiting[exit_lane]+=n
                self.action_summary[lane]=n
                self.waiting[lane]-=n

                left=copy.deepcopy(self.wtime[lane][n:])
                for car_time in range(len(left)):
                    left[car_time]+=1

                self.wtime[lane]=left
            
            #3.3 just pick a random number if there are more than maxi
            else:

                n=np.random.randint(self.mini, self.maxi+1)

                self.exiting[exit_lane]+=n
                self.action_summary[lane]=n
                self.waiting[lane]-=n

                left=copy.deepcopy(self.wtime[lane][n:].copy())
                for car_time in range(len(left)):
                    left[car_time]+=1

                self.wtime[lane]=left

        for lane in range(8):
            if lane not in self.possible[action]:
                for car_time in range(len(self.wtime[lane])):
                    self.wtime[lane][car_time]+=1


    def turning_right(self):

        #1. Make cars turn right

        for lane in range(4):

            if self.right[lane]<self.mini:

                self.exiting[(lane-1)%4]+=self.right[lane]
                self.action_summary[lane+8]=self.right[lane]
                self.right[lane]=0

                self.wtime[lane+8]=[]

            elif self.right[lane]<self.maxi:

                n=np.random.randint(self.mini, self.right[lane]+1)

                self.exiting[(lane-1)%4]+=n
                self.action_summary[lane+8]=n
                self.right[lane]-=n

                left=copy.deepcopy(self.wtime[lane+8][n:])
                for car_time in range(len(left)):
                    left[car_time]+=1

                self.wtime[lane+8]=left

            else:

                n=np.random.randint(self.mini, self.maxi)

                self.exiting[(lane-1)%4]+=n
                self.action_summary[lane+8]=n
                self.right[lane]-=n

                left=copy.deepcopy(self.wtime[lane+8][n:])
                for car_time in range(len(left)):
                    left[car_time]+=1

                self.wtime[lane+8]=left


    def rewarding(self):

        #REWARD:
        #1. PRESSURE
        pressure=np.sum(self.exiting)-np.sum(self.waiting)

        #2. MAX TIME a car has been waiting
        total_wait=0
        max_waiting=0
        for lane in range(12):
            if self.wtime[lane]!=[]:
                total_wait=np.sum(self.wtime[lane])
                
                if np.max(self.wtime[lane])>max_waiting:
                    max_waiting=np.max(self.wtime[lane])

        self.reward=pressure-4*total_wait

        #Penalty if reward is too low: LOSE THE GAME (individually)

        """
        if self.reward<=-100:
            self.reward=-120
        """