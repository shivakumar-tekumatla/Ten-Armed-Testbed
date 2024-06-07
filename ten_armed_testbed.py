""" Solution to generate Figure 2.2 of the Sutton's RL text book """

import numpy as np 
import matplotlib.pyplot as plt 
class TestBed10Arms:
    def __init__(self,k=10,n_runs=2000,n_steps=1000) -> None:
        self.k = k #Number of arms 
        self.n_runs = n_runs # Number of runs 
        self.n_steps = n_steps # Time steps 
        self.q_mean = np.random.normal(0,1,size=(self.n_runs,self.k)) # Mean values for each action for all the runs 
        self.best_arms =  np.argmax(self.q_mean,1) #On an average these picks at each run would give the best result . Basically picks with best mean value overall 
        pass

    def test_bed(self,eps=0):
        Q=np.zeros((self.n_runs,self.k)) # Initial reward estimate for all actions and runs 
        N=np.zeros((self.n_runs,self.k)) # number of times each arm was pulled 
        step_mean_rewards = [] # Mean rewards of all the runs at each step 
        optimal_action_percentage = []    # Percentage of optimal action out of all the runs  for eah step 

        for step in range(self.n_steps): # For each step
            total_reward = 0 #Keeps track of all the rewards for a given step 
            best_action_count =0 
            for run in range(self.n_runs):
                if np.random.random()<=eps: 
                    # Explore 
                    At=np.random.randint(self.k) 
                else:
                    #Exploit 
                    At=np.argmax(Q[run]) #Based on the estimate, select the best action 
                if At == self.best_arms[run]: #If the arm selected is the best optimal one based on the normal distribution 
                    best_action_count+=1 
                R = np.random.normal(self.q_mean[run][At],1)#Reward for this action is taken as the normal dist with mean q_a and one std
                total_reward += R  
                # Now we need to update the N and Q 
                N[run][At] +=1 # Because that specific action is picked 
                Q[run][At] =(1/N[run][At])*(Q[run][At]*(N[run][At]-1)+R ) #Moving average 
            mean_R = total_reward /self.n_runs 
            step_mean_rewards.append(mean_R) #Keep track of all the mean rewards for each run at each step 
            optimal_action_percentage.append(best_action_count*100/self.n_runs) # keeping the overall percentage of optimal pull 
        return step_mean_rewards,optimal_action_percentage

def main():
    fig,((ax1),(ax2)) = plt.subplots(2,1)
    ax1.set(xlabel ="Steps",ylabel ="Average Reward")
    ax2.set(xlabel ="Steps",ylabel ="% Optimal action")
    test_bed = TestBed10Arms()
    Steps = np.arange(test_bed.n_steps) # X axis 
    Epsilon = [0,0.01,0.1] 
    for eps in Epsilon:
        mean_rewards,optimal_action_percentage = test_bed.test_bed(eps=eps)
        ax1.plot(Steps,mean_rewards,label="e="+str(eps))
        ax2.plot(Steps,optimal_action_percentage,label="e="+str(eps))
    ax1.legend()
    ax2.legend()
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()


