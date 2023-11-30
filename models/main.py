import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from NN import *
from intersection2 import *
from city2 import *
from agent import *
from comparison import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

run_number="1"

mini_reward=-120
min_reward=5*(mini_reward+120)/300

#Replay_buffer
length_D=100000
use_buffer=False
batch_size=128

#Epsilon decay
epsilon_decay=0.99995

class Coordinator:

    def __init__(self, N:int, M:int, gamma: float, mini_thru:int, maxi_thru:int, mini_in:int,  maxi_in:int):

        self.time=0
        self.gamma=gamma
        
        #size of the city
        self.N=N
        self.M=M

        #Cars going through when given green light
        self.mini_thru=mini_thru
        self.maxi_thru=maxi_thru

        #Cars adding up at every round
        self.mini_in=mini_in
        self.maxi_in=maxi_in

        #Define the city:
        self.city=City5(self.N, self.M, self.mini_in, self.maxi_in)

        #If at the last iteration of an episode
        self.last={}

        #Intersections and Q_learners:
        self.all_intersections={}
        self.learners={}

        for i in range(self.N):
            for j in range(self.M):

                self.all_intersections[(i,j)]=Crossing5(self.mini_in, self.maxi_in, self.mini_thru, self.maxi_thru, i,j)
                self.learners[(i,j)]=RL_agent(self.gamma, i,j,N,M)
                self.last[(i,j)]=False

        self.city.set_harmony(self.all_intersections)

        #1. FRAP:
        self.frap=FRAP5().to(device)

        #2. FRAP_target (not updated as often)
        self.frap_target=copy.deepcopy(self.frap).to(device)
        self.frap_remember=copy.deepcopy(self.frap).to(device)

        #Initialise the replay buffer replay_D
        self.replay_Dphase=torch.zeros((length_D,2,8)) #To store the state variables
        self.replay_Dstate=torch.zeros((length_D,8,12))
        self.replay_Dint=torch.zeros((length_D,5)) #To store int (action number and reward)

    def next_round(self, epsilon:float, choice:int): #epsilon is for the epsilon greedy policy

        #Make each learner find an appropriate action and play it in its intersection
        for i in range(self.N):
            for j in range(self.M):

                #REPLAY BUFFER: (initial pressure, initial phase)
                self.replay_Dphase[(self.time*self.N*self.M+i*self.M+j)%length_D, 0,:]=self.learners[(i,j)].phase[0]

                #Define STATE:
                self.learners[(i,j)].make_state(self.all_intersections[(i,j)])
                self.replay_Dint[(self.time*self.N*self.M+i*self.M+j)%length_D, 2]=self.learners[(i,j)].i_position[0,0]
                self.replay_Dint[(self.time*self.N*self.M+i*self.M+j)%length_D, 3]=self.learners[(i,j)].j_position[0,0]

                #REPLAY BUFFER: (wtime: mu, sigma, max)
                #print(self.learners[(i,j)].wtime_mu[0], self.learners[(i,j)].wtime_max[0])
                self.replay_Dstate[(self.time*self.N*self.M+i*self.M+j)%length_D, 1,:]=self.learners[(i,j)].wtime_mu[0]
                self.replay_Dstate[(self.time*self.N*self.M+i*self.M+j)%length_D, 2,:]=self.learners[(i,j)].wtime_sigma[0]
                self.replay_Dstate[(self.time*self.N*self.M+i*self.M+j)%length_D, 3,:]=self.learners[(i,j)].wtime_max[0]

                self.replay_Dstate[(self.time*self.N*self.M+i*self.M+j)%length_D, 0,:]=self.learners[(i,j)].pressure[0]
                
                #NEXT ROUND (+ turning right)
                if choice==0:
                    self.learners[(i,j)].next_round(self.all_intersections[(i,j)], epsilon, self.frap)
                else:
                    self.learners[(i,j)].next_round(self.all_intersections[(i,j)], epsilon, self.frap_target)

                #BOUNDARY:
                #1. exit
                self.city.exit_boundary(self.all_intersections,i,j)
                #2. new cars
                self.city.waiting_boundary(self.all_intersections, i, j)

                #REWARD:
                self.learners[(i,j)].get_reward(self.all_intersections[(i,j)])

                #REPLAY BUFFER: (action, new phase)
                self.replay_Dint[(self.time*self.N*self.M+i*self.M+j)%length_D, 0]=self.learners[(i,j)].action
                self.replay_Dphase[(self.time*self.N*self.M+i*self.M+j)%length_D, 1,:]=self.learners[(i,j)].phase[0]
                self.replay_Dint[(self.time*self.N*self.M+i*self.M+j)%length_D, 1]=self.learners[(i,j)].reward


        #HARMONISE the city after:
        for i in range(self.N):
            for j in range(self.M):       

                self.city.harmonise(self.all_intersections, i, j)

                #Check if last episode:
                if self.learners[(i,j)].reward<min_reward:
                    self.last[(i,j)]=True
                    self.replay_Dint[(self.time*self.N*self.M+i*self.M+j)%length_D, 4]=1


        for i in range(self.N):
            for j in range(self.M):

                #define NEW STATE:
                self.learners[(i,j)].make_state(self.all_intersections[(i,j)])

                #REPLAY BUFFER for new state: (wtime: mu, sigma, max) & waiting
                self.replay_Dstate[(self.time*self.N*self.M+i*self.M+j)%length_D, 5,:]=self.learners[(i,j)].wtime_mu[0]
                self.replay_Dstate[(self.time*self.N*self.M+i*self.M+j)%length_D, 6,:]=self.learners[(i,j)].wtime_sigma[0]
                self.replay_Dstate[(self.time*self.N*self.M+i*self.M+j)%length_D, 7,:]=self.learners[(i,j)].wtime_max[0]

                self.replay_Dstate[(self.time*self.N*self.M+i*self.M+j)%length_D, 4,:]=self.learners[(i,j)].pressure[0]

                #COMPUTE next
                if choice==0:
                    self.learners[(i,j)].target(self.frap, self.frap_target, self.last[(i,j)])
                else:
                    self.learners[(i,j)].target(self.frap_target, self.frap, self.last[(i,j)])

                self.last[(i,j)]=False

        self.time+=1


    def backprop(self, optimiser, batch_size:int, C_target:int, choice:int):

        #We have the same frap and frap_target for all the agents:
        L=None

        for i in range(self.N):
            for j in range(self.M):

                if L is None:
                    L=self.learners[(i,j)].get_loss()

                else:
                    L+=self.learners[(i,j)].get_loss()

        L/=self.N*self.M

        #Start replaying buffer before it is fully filled up
        if self.time*self.N*self.M<length_D and self.time*self.N*self.M>2*batch_size and use_buffer==True:

            max_read=self.time*self.N*self.M

        #Once it is filled up, can replay any event from the buffer
        elif self.time*self.N*self.M>=length_D and use_buffer==True:

            max_read=length_D

        if self.time*self.N*self.M>2*batch_size and use_buffer==True:

            row_list=np.random.randint(0,max_read, size=batch_size).tolist()

            #STATE:
            #1. phase
            phase = self.replay_Dphase[row_list, 0, :]
            #print("phase0", phase.shape, phase)

            #phase=torch.tensor(phase)
            
            #2. waiting
            wait = self.replay_Dstate[row_list, 0,:]
            #print("wait0", wait.shape, wait)

            #3. wtime (mu, sigma, max):
            wtime_mu=self.replay_Dstate[row_list, 1,:]
            #print("mu0", wtime_mu.shape, wtime_mu)
            #wtime_mu=torch.tensor(wtime_mu)
            wtime_sigma=self.replay_Dstate[row_list, 2,:]
            #print("sigma0", wtime_sigma.shape, wtime_sigma)
            #wtime_sigma=torch.tensor(wtime_sigma)
            wtime_max=self.replay_Dstate[row_list, 3,:]
            #print("max0", wtime_max.shape, wtime_max)
            #wtime_max=torch.tensor(wtime_max)

            #4. retrieve position (i,j):
            i_position=torch.tensor(self.replay_Dint[row_list, np.newaxis, 2])
            j_position=torch.tensor(self.replay_Dint[row_list, np.newaxis, 3])

            #print(i_position.shape, i_position)
            #print(j_position.shape, j_position)

            #y_hat
            if choice==0:
                out=self.frap.forward(wait.to(device).float(), phase.to(device).float(), wtime_mu.to(device).float(), wtime_sigma.to(device).float(), wtime_max.to(device).float(), i_position.to(device).float(), j_position.to(device).float())
            else:
                out=self.frap_target.forward(wait.to(device).float(), phase.to(device).float(), wtime_mu.to(device).float(), wtime_sigma.to(device).float(), wtime_max.to(device).float(), i_position.to(device).float(), j_position.to(device).float())
            y_hat=torch.gather(out,1,self.replay_Dint[row_list, 0].to(torch.int64).view(-1,1))

            #NEW STATE:
            #1. phase
            phase2 = self.replay_Dphase[row_list, 1, :]
            #print("phase1", phase2.shape, phase2)
            #phase2=torch.tensor(phase2)

            #2. waiting
            wait2 = self.replay_Dstate[row_list, 4,:]
            #print("wait1", wait2.shape, wait2)
            #wait2=torch.tensor(waiting2)

            #3. wtime (mu, sigma, max):
            wtime_mu2=self.replay_Dstate[row_list, 5,:]
            #print("mu1", wtime_mu2.shape, wtime_mu2)
            wtime_sigma2=self.replay_Dstate[row_list, 6,:]
            #print("sigma1", wtime_sigma2.shape, wtime_sigma)
            wtime_max2=self.replay_Dstate[row_list, 7,:]
            #print("max1", wtime_max.shape, wtime_max)

            #y
            if choice==0:
                get_a=self.frap.forward(wait2.to(device).float(), phase2.to(device).float(), wtime_mu2.to(device).float(), wtime_sigma2.to(device).float(), wtime_max2.to(device).float(), i_position.to(device).float(), j_position.to(device).float())
            else:
                get_a=self.frap_target.forward(wait2.to(device).float(), phase2.to(device).float(), wtime_mu2.to(device).float(), wtime_sigma2.to(device).float(), wtime_max2.to(device).float(), i_position.to(device).float(), j_position.to(device).float())
            action2=get_a.max(dim=1).indices

            if choice==0:
                out2=self.frap_target.forward( wait2.to(device).float(), phase2.to(device).float(), wtime_mu2.to(device).float(), wtime_sigma2.to(device).float(), wtime_max2.to(device).float(), i_position.to(device).float(), j_position.to(device).float())
            else:
                out2=self.frap.forward( wait2.to(device).float(), phase2.to(device).float(), wtime_mu2.to(device).float(), wtime_sigma2.to(device).float(), wtime_max2.to(device).float(), i_position.to(device).float(), j_position.to(device).float())
            y=self.replay_Dint[row_list, 1]+self.gamma*torch.gather(out2,1,action2.view(-1,1))

            #for row in range(len(row_list)):
                #if self.replay_Dint[row_list[row],4]==1:
                    #y[row]=self.replay_Dint[row_list, 1]

            L+=loss(y.float(), y_hat.float())


        #Freeze the network you do not backpropagate
        if choice==0:
            self.frap_remember=copy.deepcopy(self.frap_target)
        else:
            self.frap_remember=copy.deepcopy(self.frap)

        #Backpropagate on both networks (estimation network)
        optimiser.zero_grad()
        L.backward()
        optimiser.step()

        if choice==0:
            self.frap_target=self.frap_remember
        else:
            self.frap=self.frap_remember

        #BUT cancel the effects of backpropagation on target network if it is not the right time
        #self.update_target_network(C_target)

    def update_target_network(self, C:int):
        #This function allows to update the target network every C iterations
        #If it's not the right time (not modulo C): target network comes back to the target remember network which cancels the effect of backpropagation
        #THUS NEED TO PUT THIS FUNCTION AFTER BACKPROPAGATION IN TRAINING LOOP!!!!

        if self.time%C==0:
            self.frap_target=copy.deepcopy(self.frap)

    def reset_game(self):

        #In case one agent loses the game, reset the intersections:
        self.all_intersections={}

        for i in range(self.N):
            for j in range(self.M):
                self.all_intersections[(i,j)]=Crossing5(self.mini_in, self.maxi_in, self.mini_thru, self.maxi_thru, i,j)

        #Then need to harmonise to make each intersection aware of its neighbors:
        self.city.set_harmony(self.all_intersections)

    def train(self, iterations:int, epsilon:float, batch_size:int, C_target:int):

        #Re-initialise:
        self.city=City5(self.N, self.M, self.mini_in,  self.maxi_in)
        self.learners={}

        #Define the optimisers:
        optimisers=optim.Adam(self.frap.parameters(), lr=1e-3)
        optimisers_target=optim.Adam(self.frap_target.parameters(), lr=1e-3)

        #PREPARE PLOTTING: (make dictionaries of lists to keep track of several intersections)
        R_history={}
        WAIT_history={}
        EXIT_history={}
        WAIT_position={}
        LENGTH_EPISODE_history={}
        dead={}

        age_history={}

        average=[]

        EPISODES_OVERALL_history=[0]
        deadd=0

        Q_value={}
        Q_value2={}
        Q_value3={}

        #FILL IN:
        for i in range(self.N):
            for j in range(self.M):
                self.learners[(i,j)]=RL_agent(self.gamma, i, j, self.N, self.M)

                #NOW FOR THE PLOTTING:
                age_history[(i,j)]=[]

                R_history[(i,j)]=[] #reward, collected from the Q_Learner2
                WAIT_history[(i,j)]=[] #Max number of cars waiting at an intersection,, also collected from Q_Learner2
                EXIT_history[(i,j)]=[]
                LENGTH_EPISODE_history[(i,j)]=[0] #Collected with length of episodes computed later for each intersection
                dead[(i,j)]=0 #Keep track for the length of episode

                Q_value[(i,j)]=[]
                Q_value2[(i,j)]=[]
                Q_value3[(i,j)]=[]

        #TRAINING:
        self.need_reset=0

        for iteration in range(iterations):    

            print("iteration",iteration)

            proba=0
            self.next_round(epsilon, proba)
            if proba==0:
                self.backprop(optimisers, batch_size, C_target, proba)
            else:
                self.backprop(optimisers_target, batch_size, C_target, proba)

            epsilon*=epsilon_decay

            age_history={}
            #Need to measure the episodes
            for i in range(self.N):
                for j in range(self.M):

                    Q_value[(i,j)]=[self.learners[(i,j)].y_hat.item(),self.learners[(i,j)].y_hat.item()]
                    Q_value2[(i,j)]=[self.learners[(i,j)].y.item(),self.learners[(i,j)].y.item()]
                    Q_value3[(i,j)]=[self.learners[(i,j)].reward,self.learners[(i,j)].reward]

                    age_history[(i,j)]=[]
                    for lane in range(8):
                        if self.all_intersections[(i,j)].wtime[lane].shape[0]==0:
                            age_history[(i,j)].append(0)
                        else:
                            age_history[(i,j)].append(int(torch.max(self.all_intersections[(i,j)].wtime[lane])))
                    
                    R_history[(i,j)].append(self.learners[(i,j)].reward)
                    
                    L=[]
                    for k in range(12):
                        L.append(self.all_intersections[(i,j)].waiting[k])
                    WAIT_history[(i,j)].append(max(L))
                    WAIT_position[(i,j)]=L

                    L=[]
                    for k in range(4):
                        L.append(self.all_intersections[(i,j)].exiting[k])
                    EXIT_history[(i,j)]=L

                    if self.learners[(i,j)].reward==0: 
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


            if iteration%500==0:
                with open("frap_checkpoint_"+run_number+".chkpt", "wb+") as f:
                    torch.save(self.frap.state_dict(), f)
                with open("frap-target_checkpoint_"+run_number+".chkpt", "wb+") as f:
                    torch.save(self.frap_target.state_dict(), f)
                with open("metrics_checkpoint_"+run_number+".chkpt", "wb+") as f:
                    torch.save(EPISODES_OVERALL_history, f)

            #PLOTTING:
            if iteration%500==0:

                print("run_number", run_number)

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
                ax5.set_title("Age")
                fig6, ax6 = plt.subplots()
                ax6.set_title("Q_values")
                fig7, ax7 = plt.subplots()
                ax7.set_title("exiting position")
                fig8, ax8 = plt.subplots()
                ax8.set_title("waiting position")
                

                COLOR_LIST=["black", "gray", "silver", "maroon", "red", "purple", "fuchsia", "green", "lime", "olive", "yellow",
                "navy", "blue", "teal", "aqua", "coral", "orange", "gold", "khaki", "indigo", "violet", "pink", "salmon",
                "crimson", "brown", "sienna", "tan", "beige", "rosybrown", "slategray", "darkslategray", "lightslategray",
                "darkgreen", "seagreen", "forestgreen", "darkturquoise", "turquoise", "skyblue", "deepskyblue", "steelblue",
                "royalblue", "mediumblue", "darkblue", "mediumslateblue", "darkorchid", "mediumorchid", "thistle", "plum"]


                for i in range(self.N):
                    for j in range(self.M):

                        ax1.plot(R_history[(i,j)][-100:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")
                        ax2.plot(LENGTH_EPISODE_history[(i,j)][-200:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")
                        ax3.plot(WAIT_history[(i,j)][-20:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")

                        ax5.plot(age_history[(i,j)], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")

                        ax6.plot(Q_value[(i,j)], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")_y_hat")
                        ax6.plot(Q_value2[(i,j)], color=COLOR_LIST[10+i*self.M+j], label="("+str(i)+","+str(j)+")_y")
                        ax6.plot(Q_value3[(i,j)], color=COLOR_LIST[15+i*self.M+j], label="("+str(i)+","+str(j)+")_reward")

                        ax7.plot(EXIT_history[(i,j)][:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")
                        ax8.plot(WAIT_position[(i,j)][:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")

                    
                ax4.plot(EPISODES_OVERALL_history[-200:], color='red')

                ax2.legend()
                ax5.legend()
                ax6.legend()
                ax7.legend()

                fig1.savefig("Rewards"+run_number+".png")
                fig2.savefig("Length of episodes"+run_number+".png")
                fig3.savefig("MAX_waiting_history"+run_number+".png")
                fig4.savefig("Overall episodes"+run_number+".png")
                fig5.savefig("age_plot"+run_number+".png")
                fig6.savefig("Q_values"+run_number+".png")
                fig7.savefig("exiting_position"+run_number+".png")
                fig8.savefig("wait_position"+run_number+".png")

                plt.close(fig1)
                plt.close(fig2)
                plt.close(fig3)
                plt.close(fig4)
                plt.close(fig5)
                plt.close(fig6)
                plt.close(fig7)
                plt.close(fig8)

            if len(EPISODES_OVERALL_history)%100==0 and iteration!=0 and check==1:

                check=0

                fig10, ax10 = plt.subplots()
                ax10.set_title("average_episodes"+str(C_target))

                print("epsilon=", epsilon)
                average_episode=np.sum(EPISODES_OVERALL_history[-100:])
                average_episode/=100
                average.append(average_episode)
                ax10.plot(average, color=COLOR_LIST[7])

                fig10.savefig("average_episodes_overall"+run_number+".png")
                plt.close(fig10)

            if len(EPISODES_OVERALL_history)%100!=0:

                check=1

mini_thru=7
maxi_thru=9

mini_in=1
maxi_in=3

#print(No_learning_coordinator(1,1,mini_thru, maxi_thru, mini_in, maxi_in).episodes_random_agent())
#print(No_learning_coordinator(1,1,mini_thru, maxi_thru, mini_in, maxi_in).episodes_cyclic_agent())
#print(No_learning_coordinator(1,1,mini_thru, maxi_thru, mini_in, maxi_in).episodes_greedy_agent())


coord=Coordinator(1,2,0.95, mini_thru, maxi_thru, mini_in, maxi_in)
coord.train(10000000000, 0.08, batch_size, 500)