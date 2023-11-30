import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from frap2 import *
from cross import *
from city import *
from qlearner import *
from comparison import *

#Loss function:
loss=nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if you have a GPU with CUDA installed, this may speed up computation

run_number="4"

min_reward=-100

#Replay_buffer
length_D=100000
use_buffer=True
batch_size=128

#Epsilon decay
epsilon_decay=0.99999

class Coordinator:

    def __init__(self, N:int, M:int, gamma: float, mini_thru:int, maxi_thru:int, mini_in:int,  maxi_in:int):

        self.gamma=gamma
        self.time=0

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
        self.city=City(self.N, self.M, self.mini_in, self.maxi_in)

        #If at the last iteration of an episode
        self.last={}

        #Intersections and Q_learners:
        self.all_intersections={}
        self.learners={}

        for i in range(self.N):
            for j in range(self.M):

                self.all_intersections[(i,j)]=Intersection(self.mini_thru, self.maxi_thru)
                self.learners[(i,j)]=Q_Learner(self.gamma)

                self.last[(i,j)]=False

        #FRAP: (shared by all learners)

        #1. FRAP:
        self.frap=FRAP2().to(device)

        #2. FRAP_target (not updated as often)
        self.frap_target=copy.deepcopy(self.frap).to(device)

        #Initialise the replay buffer replay_D
        self.replay_D_s=np.zeros((length_D,10,8)) #To store the state variables
        self.replay_D_ar=np.zeros((length_D,4)) #To store int (action number and reward)


    def next_round(self, epsilon:float): #epsilon is for the epsilon greedy policy

        #Make each learner find an appropriate action and play it in its intersection
        for i in range(self.N):
            for j in range(self.M):

                #REPLAY BUFFER: (initial pressure, initial phase)
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 1,:]=self.learners[(i,j)].phase

                if self.N-1!=0:
                    self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 2]=i/(self.N-1)
                    i_position=i/(self.N-1)
                else:
                    self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 2]=i
                    i_position=i
                if self.M-1!=0:
                    self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 3]=j/(self.M-1)
                    j_position=j/(self.M-1)
                else:
                    self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 3]=j
                    j_position=j

                #Define STATE:
                self.learners[(i,j)].make_state(self.all_intersections[(i,j)])

                #REPLAY BUFFER: (wtime: mu, sigma, max)
                #print(self.learners[(i,j)].wtime_mu[0], self.learners[(i,j)].wtime_max[0])
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 4,:]=self.learners[(i,j)].wtime_mu[0]
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 5,:]=self.learners[(i,j)].wtime_sigma[0]
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 6,:]=self.learners[(i,j)].wtime_max[0]

                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 0,:]=self.learners[(i,j)].pressure[0]
                
                #NEXT ROUND (+ turning right)
                self.learners[(i,j)].next_round(self.all_intersections[(i,j)], epsilon, self.frap, i, j)
                self.all_intersections[(i,j)].turning_right()

                #BOUNDARY:
                #1. exit
                self.city.update_exit_boundary(self.all_intersections,i,j)
                #2. new cars
                self.city.add_at_boundary(self.all_intersections, i, j)

                #REWARD:
                self.learners[(i,j)].get_reward(self.all_intersections[(i,j)])

                #REPLAY BUFFER: (action, new phase)
                self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 0]=self.learners[(i,j)].action
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 3,:]=self.learners[(i,j)].phase
                self.replay_D_ar[(self.time*self.N*self.M+i*self.M+j)%length_D, 1]=self.learners[(i,j)].reward


        #HARMONISE the city after:
        for i in range(self.N):
            for j in range(self.M):       

                self.city.harmonise(self.all_intersections, i, j)

                #Check if last episode:
                if self.learners[(i,j)].reward<min_reward:
                    self.last[(i,j)]=True


        for i in range(self.N):
            for j in range(self.M):

                if self.N-1!=0:
                    i_position=i/(self.N-1)
                else:
                    i_position=i
                if self.M-1!=0:
                    j_position=j/(self.M-1)
                else:
                    j_position=j

                #define NEW STATE:
                self.learners[(i,j)].make_new_state(self.all_intersections[(i,j)])

                #REPLAY BUFFER for new state: (wtime: mu, sigma, max) & waiting
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 7,:]=self.learners[(i,j)].wtime_mu[0]
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 8,:]=self.learners[(i,j)].wtime_sigma[0]
                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 9,:]=self.learners[(i,j)].wtime_max[0]

                self.replay_D_s[(self.time*self.N*self.M+i*self.M+j)%length_D, 2,:]=self.learners[(i,j)].pressure[0]

                #COMPUTE next
                self.learners[(i,j)].target(self.frap, self.frap_target, i_position, j_position, self.last[(i,j)])

                self.last[(i,j)]=False

        self.time+=1
                
                
    def backprop(self, optimisers, batch_size:int, C_target:int):

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
            phase = self.replay_D_s[row_list, 1, :]
            phase=torch.tensor(phase)
            
            #2. waiting
            waiting = self.replay_D_s[row_list, 0,:]
            wait=torch.tensor(waiting)

            #3. wtime (mu, sigma, max):
            wtime_mu=self.replay_D_s[row_list, 4,:]
            wtime_mu=torch.tensor(wtime_mu)
            wtime_sigma=self.replay_D_s[row_list, 5,:]
            wtime_sigma=torch.tensor(wtime_sigma)
            wtime_max=self.replay_D_s[row_list, 6,:]
            wtime_max=torch.tensor(wtime_max)

            #4. retrieve position (i,j):
            i_position=torch.tensor(self.replay_D_ar[row_list, np.newaxis, 2])
            j_position=torch.tensor(self.replay_D_ar[row_list, np.newaxis, 3])

            #y_hat
            out=self.frap.forward( wait.to(device).float(), phase.to(device).float(), wtime_mu.to(device).float(), wtime_sigma.to(device).float(), wtime_max.to(device).float(), i_position.to(device).float(), j_position.to(device).float())
            y_hat=out[:,self.replay_D_ar[row_list, 0].astype(int)]


            #NEW STATE:
            #1. phase
            phase2 = self.replay_D_s[row_list, 3, :]
            phase2=torch.tensor(phase2)

            #2. waiting
            waiting2 = self.replay_D_s[row_list, 2,:]
            wait2=torch.tensor(waiting2)

            #3. wtime (mu, sigma, max):
            wtime_mu2=self.replay_D_s[row_list, 7,:]
            wtime_mu2=torch.tensor(wtime_mu2)
            wtime_sigma2=self.replay_D_s[row_list, 8,:]
            wtime_sigma2=torch.tensor(wtime_sigma2)
            wtime_max2=self.replay_D_s[row_list, 9,:]
            wtime_max2=torch.tensor(wtime_max2)

            #y
            get_a=self.frap.forward( wait2.to(device).float(), phase2.to(device).float(), wtime_mu2.to(device).float(), wtime_sigma2.to(device).float(), wtime_max2.to(device).float(), i_position.to(device).float(), j_position.to(device).float())
            action2=get_a.max(dim=1).indices

            out2=self.frap_target.forward( wait2.to(device).float(), phase2.to(device).float(), wtime_mu2.to(device).float(), wtime_sigma2.to(device).float(), wtime_max2.to(device).float(), i_position.to(device).float(), j_position.to(device).float())
            y=torch.tensor((self.replay_D_ar[row_list, 1]+120)/150)+self.gamma*torch.gather(out2,1,action2.view(-1,1))

            L+=loss(y.float(), y_hat.float())
        
        #Backpropagate on both networks (estimation network)
        optimisers.zero_grad()
        L.backward()
        optimisers.step()

        #BUT cancel the effects of backpropagation on target network if it is not the right time
        self.update_target_network(C_target)


    def reset_game(self):

        #In case one agent loses the game, reset the intersections:
        self.all_intersections={}

        for i in range(self.N):
            for j in range(self.M):
                self.all_intersections[(i,j)]=Intersection(self.mini_thru, self.maxi_thru)

        #Then need to harmonise to make each intersection aware of its neighbors:
        self.city.set_harmony(self.all_intersections)
        

    def update_target_network(self, C:int):
        #This function allows to update the target network every C iterations
        #If it's not the right time (not modulo C): target network comes back to the target remember network which cancels the effect of backpropagation
        #THUS NEED TO PUT THIS FUNCTION AFTER BACKPROPAGATION IN TRAINING LOOP!!!!

        if self.time%C==0:
            self.frap_target=copy.deepcopy(self.frap)


    def train(self, iterations:int, epsilon:float, batch_size:int, C_target:int):

        #Re-initialise:
        self.city=City(self.N, self.M, self.mini_in,  self.maxi_in)
        self.learners={}

        #Define the optimisers:
        optimisers=optim.Adam(self.frap.parameters(), lr=1e-3)


        #PREPARE PLOTTING: (make dictionaries of lists to keep track of several intersections)
        R_history={}
        WAIT_history={}
        EXIT_history={}
        WAIT_position={}
        RIGHT_position={}
        LENGTH_EPISODE_history={}
        dead={}

        age_history={}

        average=[]
        #new_average=torch.load("metrics_checkpoint_3.chkpt")
        #total_length=len(new_average)//100
        #for i in range(total_length):
            #average.append(np.sum(new_average[100*i:100*i+100])/100)

        #print(average)


        EPISODES_OVERALL_history=[0]
        deadd=0

        Q_value={}
        Q_value2={}
        Q_value3={}

        #FILL IN:
        for i in range(self.N):
            for j in range(self.M):
                self.learners[(i,j)]=Q_Learner(self.gamma)

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

            #Exploration decay:
            epsilon*=epsilon_decay

            print("iteration",iteration)


            self.next_round(epsilon)
            self.backprop(optimisers, batch_size, C_target)

            age_history={}
            
            #Need to measure the episodes
            for i in range(self.N):
                for j in range(self.M):

                    Q_value[(i,j)]=[float(self.learners[(i,j)].y_hat),float(self.learners[(i,j)].y_hat)]
                    Q_value2[(i,j)]=[float(self.learners[(i,j)].y),float(self.learners[(i,j)].y)]
                    Q_value3[(i,j)]=[10*(float(self.learners[(i,j)].reward)+120)/150,(float(self.learners[(i,j)].reward)+120)/150]

                    age_history[(i,j)]=[]
                    for lane in range(8):
                        if self.all_intersections[(i,j)].wtime[lane]==[]:
                            age_history[(i,j)].append(0)
                        else:
                            age_history[(i,j)].append(np.max(self.all_intersections[(i,j)].wtime[lane]))
                    
                    R_history[(i,j)].append(self.learners[(i,j)].reward)
                    
                    L=[]
                    for k in range(8):
                        L.append(self.all_intersections[(i,j)].waiting[k])
                    WAIT_history[(i,j)].append(max(L))
                    WAIT_position[(i,j)]=L

                    L=[]
                    for k in range(4):
                        L.append(self.all_intersections[(i,j)].exiting[k])
                    EXIT_history[(i,j)]=L

                    L=[]
                    for k in range(4):
                        L.append(self.all_intersections[(i,j)].right[k])
                    RIGHT_position[(i,j)]=L

                    if self.learners[(i,j)].reward<min_reward: 
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
                fig9, ax9 = plt.subplots()
                ax9.set_title("right position")
                

                COLOR_LIST=["black", "gray", "silver", "white", "maroon", "red", "purple", "fuchsia", "green", "lime", "olive", "yellow",
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
                        ax9.plot(RIGHT_position[(i,j)][:], color=COLOR_LIST[i*self.M+j], label="("+str(i)+","+str(j)+")")

                    
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
                fig9.savefig("right_position"+run_number+".png")

                plt.close(fig1)
                plt.close(fig2)
                plt.close(fig3)
                plt.close(fig4)
                plt.close(fig5)
                plt.close(fig6)
                plt.close(fig7)
                plt.close(fig8)
                plt.close(fig9)

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
coord.train(20000000, 1, batch_size, 100)