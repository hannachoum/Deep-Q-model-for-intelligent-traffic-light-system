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
from Q_learning2 import *

#Loss function:
loss=nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

#Replay_buffer
length_D=10000000

#Epsilon decay
epsilon_decay=0.99999
starting_decay=5000

class Coordinator:

    def __init__(self, N:int, M:int, gamma: float, mini_thru:int, maxi_thru:int, mini_in:int,  maxi_in:int):

        self.gamma=gamma
        self.time=0

        self.N=N
        self.M=M

        self.mini_thru=mini_thru
        self.maxi_thru=maxi_thru

        self.mini_in=mini_in
        self.maxi_in=maxi_in

        self.city=City2(self.N, self.M, self.mini_in, self.maxi_in)

        #Intersections, the dictionary of all the intersections depends on coordinator (as both City and Q_Learner2 use it)
        #We also initialise the dictionary of all the Q_Learners:

        self.all_intersections={}
        self.learners={}

        for i in range(self.N):
            for j in range(self.M):

                self.all_intersections[(i,j)]=Intersection4(self.mini_thru, self.maxi_thru)
                self.learners[(i,j)]=Q_Learner3(self.gamma)

        #INITIALISE FRAP:
        #Same FRAP network for all the learners
        self.frap=FRAP().to(device)
        #We need a different network for the target (also shared by all the learners)
        self.frap_target=FRAP().to(device)
        #This one is to remember the target network as this network should be uptated at each iteration
        self.frap_target_remember=copy.deepcopy(self.frap_target)

        #Initialise the replay buffer replay_D
        self.replay_D_s=np.zeros((length_D,4,8)) #To store the state variables
        self.replay_D_ar=np.zeros((length_D,2)) #To store int (action number and reward, and position of intersection)


    def next_round(self, epsilon:float): #epsilon is for the epsilon greedy policy

        self.time+=1

        #Make each learner find an appropriate action and play it in its intersection
        for i in range(self.N):
            for j in range(self.M):

                #REPLAY BUFFER: (initial pressure, initial phase)
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 0,:]=self.all_intersections[(i,j)].n_waiting
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 1,:]=self.learners[(i,j)].phase
                
                #DO the next round 
                self.all_intersections[(i,j)].equilibrate() #Need to equilibrate often to make sure n_waiting corresponds to the numerical version of waiting and so on

                self.learners[(i,j)].next_round(self.all_intersections[(i,j)], epsilon, self.frap)

                self.all_intersections[(i,j)].equilibrate()

                self.all_intersections[(i,j)].turning_right()

                self.all_intersections[(i,j)].equilibrate()

                #Update exit boundary:
                self.city.update_exit_boundary(self.all_intersections,i,j)

                self.all_intersections[(i,j)].equilibrate()

                #ADD NEW CARS AT BOUNDARY:
                self.city.add_at_boundary(self.all_intersections, i, j)

                #REPLAY BUFFER: (action, new phase)
                self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 0]=self.learners[(i,j)].action
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 3,:]=self.learners[(i,j)].phase

                #COMPUTE REWARD for the corresponding action for each learner

                self.all_intersections[(i,j)].equilibrate()

                self.learners[(i,j)].rewards(self.all_intersections[(i,j)], self.frap_target)

                self.all_intersections[(i,j)].equilibrate()
                
                #REPLAY BUFFER: (new pressure (needed harmonise function), reward):
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 2,:]=self.all_intersections[(i,j)].n_waiting
                self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 1]=self.learners[(i,j)].reward

                #Harmonise the city:
                self.city.harmonise(self.all_intersections, i, j)

                


    def backprop(self, optimisers, optimisers_target, batch_size:int, C_target:int):

        #We have the same frap and frap_target for all the agents:
        L=None

        for i in range(self.N):
            for j in range(self.M):

                if L is None:
                    L=self.learners[(i,j)].get_loss()
                    #print(L.shape, L,"None")
                else:
                    L+=self.learners[(i,j)].get_loss()
                    #print(L.shape, L)

        L/=(self.N*self.M)

        #Backpropagate on both networks (target and non target)
        optimisers.zero_grad()
        optimisers_target.zero_grad()

        L.backward()
        optimisers.step()
        optimisers_target.step()

        #BUT cancel the effects of backpropagation on target network if it is not the right time
        self.update_target_network(C_target, self.frap_target, self.frap_target_remember)

        #LATER REPLAY BUFFER

        if self.time*self.N*self.M>length_D:

            for _ in range(batch_size):

                row=np.random.randint(0,length_D)

                #1. Get the corresponding phase
                phase=torch.zeros((1,8))
                for j in range(8):
                    phase[0,j]=self.replay_D_s[row, 1,j]
                
                waiting = self.replay_D_s[row, 0,:]
                for lane in range(8):
                    waiting[lane] /= 50
                
                out=self.frap.forward(torch.Tensor(np.array([waiting])).to(device), phase.to(device))
                y_hat=out[int(self.replay_D_ar[row, 0])]

                phase2=torch.zeros((1,8))
                for j in range(8):
                    phase2[0,j]=self.replay_D_s[row, 3,j]

                waiting = self.replay_D_s[row, 2,:]
                for lane in range(8):
                    waiting[lane] /= 50

                out2=self.frap_target.forward(torch.Tensor(np.array(([waiting]))).to(device), phase2.to(device))
                y=self.replay_D_ar[row, 1]+self.gamma*out2.max()

                L+=loss(y, y_hat)

            L/=(self.N*self.M+batch_size)

        else:
            L/=self.N*self.M


    def reset_game(self):

        #In case one agent loses the game, reset the intersections:
        self.all_intersections={}

        for i in range(self.N):
            for j in range(self.M):
                self.all_intersections[(i,j)]=Intersection4(self.mini_thru, self.maxi_thru)

        #Then need to harmonise to make each intersection aware of its neighbors:
        self.city.set_harmony(self.all_intersections)
        

    def update_target_network(self, C:int, frap_target:FRAP, frap_target_remember:FRAP):
        #This function allows to update the target network every C iterations
        #If it's not the right time (not modulo C): target network comes back to the target remember network which cancels the effect of backpropagation
        #THUS NEED TO PUT THIS FUNCTION AFTER BACKPROPAGATION IN TRAINING LOOP!!!!
        if self.time%C!=0:
            frap_target=copy.deepcopy(frap_target_remember)

        #Otherwise update:
        else:
            frap_target_remember=copy.deepcopy(frap_target)

    def train(self, iterations:int, epsilon:float, batch_size:int, C_target:int):

        #Re-initialise:
        self.city=City2(self.N, self.M, self.mini_in,  self.maxi_in)
        self.learners={}

        #Define the optimisers:
        optimisers=optim.Adam(self.frap.parameters())
        optimisers_target=optim.Adam(self.frap_target.parameters())

        #PREPARE PLOTTING: (make dictionaries of lists to keep track of several intersections)
        R_history={}
        WAIT_history={}
        EXIT_history={}
        WAIT_position={}
        RIGHT_position={}
        LENGTH_EPISODE_history={}
        dead={}              

        EPISODES_OVERALL_history=[0]
        deadd=0

        OUT={}
        OUT2={}

        Q_value={}
        Q_value2={}

        #FILL IN:
        for i in range(self.N):
            for j in range(self.M):
                self.learners[(i,j)]=Q_Learner3(self.gamma)

                #NOW FOR THE PLOTTING:
                R_history[(i,j)]=[] #reward, collected from the Q_Learner2
                WAIT_history[(i,j)]=[] #Max number of cars waiting at an intersection,, also collected from Q_Learner2
                EXIT_history[(i,j)]=[]
                LENGTH_EPISODE_history[(i,j)]=[0] #Collected with length of episodes computed later for each intersection
                dead[(i,j)]=0 #Keep track for the length of episode

                OUT[(i,j)]=[]
                OUT2[(i,j)]=[]

                Q_value[(i,j)]=[]
                Q_value2[(i,j)]=[]

        #TRAINING:
        self.need_reset=0

        for iteration in range(iterations):

            #Exploration decay:
            if iteration>starting_decay and epsilon>0.001:
                epsilon*=epsilon_decay

            print("iteration",iteration)

            self.next_round(epsilon)
            self.backprop(optimisers, optimisers_target, batch_size, C_target)
            

            #Need to measure the episodes
            for i in range(self.N):
                for j in range(self.M):

                    OUT[(i,j)]=[float(self.learners[(i,j)].out[k]) for k in range(8)]
                    OUT2[(i,j)]=[float(self.learners[(i,j)].out2[k]) for k in range(8)]

                    Q_value[(i,j)]=[float(self.learners[(i,j)].y_hat),float(self.learners[(i,j)].y_hat)]
                    Q_value2[(i,j)]=[float(self.learners[(i,j)].y),float(self.learners[(i,j)].y)]

                    R_history[(i,j)].append(self.learners[(i,j)].reward)
                    
                    L=[]
                    for k in self.all_intersections[(i,j)].waiting.keys():
                        L.append(len(self.all_intersections[(i,j)].waiting[k]))
                    WAIT_history[(i,j)].append(max(L))
                    WAIT_position[(i,j)]=L

                    L=[]
                    for k in self.all_intersections[(i,j)].exiting.keys():
                        L.append(len(self.all_intersections[(i,j)].exiting[k]))
                    EXIT_history[(i,j)]=L

                    L=[]
                    for k in self.all_intersections[(i,j)].right.keys():
                        L.append(len(self.all_intersections[(i,j)].right[k]))
                    RIGHT_position[(i,j)]=L

                    if self.learners[(i,j)].reward==-150: 
                        #NEED TO RESET
                        #Wait for all intersections to finish their round
                        self.need_reset=1
                        
                        LENGTH_EPISODE_history[(i,j)].append(0)
                        dead[(i,j)]=0

                    else:
                        dead[(i,j)]+=1
                        LENGTH_EPISODE_history[(i,j)][-1]+=1
                        #print(len(LENGTH_EPISODE_history[(i,j)]),LENGTH_EPISODE_history[(i,j)][-1])
                        

            #RESET THE GAME if needed:
            if self.need_reset==1:
                self.reset_game()
                self.need_reset=0

                #To keep track of the length of episodes overall:
                EPISODES_OVERALL_history.append(0)
                deadd=0
            else:
                deadd+=1
                EPISODES_OVERALL_history[-1]+=1

            #PLOTTING:
            if iteration%100==0:

                #Initialise plots:
                fig1, ax1 = plt.subplots()
                ax1.set_title("Rewards")
                fig2, ax2 = plt.subplots()
                ax2.set_title("Length of episodes")
                fig3, ax3 = plt.subplots()
                ax3.set_title("MAX waiting")
                fig4, ax4 = plt.subplots()
                ax4.set_title("Episodes overall")
                fig5, ax5 = plt.subplots()
                ax5.set_title("OUT")
                fig6, ax6 = plt.subplots()
                ax6.set_title("Q_values")
                fig7, ax7 = plt.subplots()
                ax7.set_title("exiting position")
                fig8, ax8 = plt.subplots()
                ax8.set_title("waiting position")
                fig9, ax9 = plt.subplots()
                ax9.set_title("right position")

                COLOR_LIST=['black','gold','crimson', 'green', 'olive', 'limegreen', 'dodgerblue', 'deeppink','yellow','lawngreen','darkturquoise','red',
                'darkred','darkorange','forestgreen','teal']

                Q_color=['cyan', 'darkturquoise', 'dodgerblue', 'blue', 'lime', 'forestgreen', 'darkgreen', 'black', 'cornflowerblue', 'chartreuse', 'mediumturquoise',
                'royalblue', 'slateblue', 'palegreen','olive', 'teal', 'aqua']
                OUT2_color=["red", 'crimson','orangered','orange', 'tomato','peru','rosybrown','darkred','grey','chocolate', "deeppink", "rosybrown", "darkviolet",
                "violet",  "lightsalmon", 'coral','fuchsia']

                for i in range(self.N):
                    for j in range(self.M):

                        ax1.plot(R_history[(i,j)][-100:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")
                        ax2.plot(LENGTH_EPISODE_history[(i,j)][:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")
                        ax3.plot(WAIT_history[(i,j)][-20:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")
                        
                        ax7.plot(EXIT_history[(i,j)][:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")
                        ax8.plot(WAIT_position[(i,j)][:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")
                        ax9.plot(RIGHT_position[(i,j)][:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")

                        ax5.plot(OUT[(i,j)], color=Q_color[i*self.M+j], label="("+str(i)+","+str(j)+")")
                        ax5.plot(OUT2[(i,j)], color=OUT2_color[i*self.M+j], label="("+str(i)+","+str(j)+")_2")

                        ax6.plot(Q_value[(i,j)], color=Q_color[i*self.M+j], label="("+str(i)+","+str(j)+")_2")
                        ax6.plot(Q_value2[(i,j)], color=OUT2_color[i*self.M+j], label="("+str(i)+","+str(j)+")_2")

                    
                ax4.plot(EPISODES_OVERALL_history[:], color='red')

                ax2.legend()
                ax5.legend()
                ax6.legend()
                ax7.legend()

                fig1.savefig("Rewards.png")
                fig2.savefig("Length of episodes.png")
                fig3.savefig("MAX_waiting_history.png")
                fig4.savefig("Overall episodes.png")
                fig5.savefig("OUT_plot.png")
                fig6.savefig("Q_values.png")
                fig7.savefig("exiting_position.png")
                fig8.savefig("wait_position.png")
                fig9.savefig("right_position.png")

                plt.close(fig1)
                plt.close(fig2)
                plt.close(fig3)
                plt.close(fig4)
                plt.close(fig5)
                plt.close(fig6)
                plt.close(fig7)
                plt.close(fig8)
                plt.close(fig9)

mini_thru=7
maxi_thru=9

mini_in=1
maxi_in=3

coord=Coordinator(1,2,0.95, mini_thru, maxi_thru, mini_in, maxi_in)
coord.train(100000, 0.05, 128, 500)