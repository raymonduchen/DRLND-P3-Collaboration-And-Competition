[//]: # "Image References"

[image1]: ./Report.assets/agent_tested.gif "Trained Agent"
[image2]: ./Report.assets/reward.png



## DRLND-P3 : Collaboration and Competition Report

![Trained Agent][image1]



### Project introduction

The goal of this project is training a multple agent deep reinforcement learning model to control two rackets (agents) to bounce a ball over a net to keep the ball in play as many time steps as possible in a virtual environment [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) from Udacity.

In this project, I trained a multi-agent deep deterministic policy gradient (MADDPG) model for the agent, there's one critic neural network and one actor neural network for both agents and replay buffer is shared for both agents. It can get average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

The implementation of MADDPG algorithm is based on [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) provided from Udacity with some modifications such as hyperparameters tuning, multiple agents support, different neural network architecture, model update periodically and learning from experiences multiple times for model update.



### MADDPG Learning Algorithm

Multi-agent deep deterministic policy gradient (MADDPG) algorithm is an extension version of deep deterministic policy gradient ([DDPG](https://arxiv.org/abs/1509.02971)) algorithm for multiple agents. It's an actor and critic algorithm.  

Actor neural network approximates policy and critic neural network approximates action-value function. Actor policy determines action based on state and critic evaluates action-value based on state and action.

In my implementation, both agents share the same actor neural network, critic neural network and replay buffer. 

#### Dealing with unstable learning

One of the biggest problem in reinforcement learning is its unstable learning. There're several techniques  used in this project's :

- **Experience replay using memory buffer**：DDPG or MADDPG uses memory buffer to store several experience tuples with state, action, reward and next state `(s, a, r, s')`, and sample it randomly for learning to avoid sequential experience correlation. In this project, I used a shared memory buffer for both agents.
- **Fixed target**：Similar to DQN, it's unstable to train model with moving target. Therefore, both MADDPG actor and MADDPG critic have their own pairs of local network and target network. 
- **Soft update**：Unlike DQN updates local network and target network by copying them for periodic steps, MADDPG updates them gradually (softly) for every learning step that controlled by a soft update parameter.
- **Ornstein and Uhlenbeck Noise**：A major advantage of off policy algorithms such as MADDPG is it can explore independently from the learning algorithm. Instead of using original actor policy, MADDPG add it with noise sampled from Ornstein and Uhlenbeck noise. This could help a more explorative policy in training and find a better policy.
- **Gradient clipping**：To avoid gradient explosion, I added gradient clipping as suggested in Udacity course to clip the norm of the gradients at 1. It places a upper limit and avoids too large update to neural network parameters. 
- **Periodic and multiple learning**：Frequent learning with few experiences replay is not a good training strategy. In this project, Learning is made for every 10 timesteps and it learns 10 times for each learning by sampling from shared memory buffer randomly.



#### MADDPG Neural Network Architecture 

The actor consists of 3 fully connected layer with 256, 128 and 4 units, respectively. The first two fully connected layers are followed with a batch normalization layer and ReLU nonlinear activation function. Final fully connected layer is followed with tanh activation function. The actor takes state (size = 24) as input and output action (size=2). Note that the state size 24 is because Unity environment internally stacked 3 measurments (the size of one measurement = 8) together to return the observation.

| Actor             | Input size | Output size |
| ----------------- | ---------- | ----------- |
| Layer1 (fc1)      | 24         | 256         |
| BatchNorm1d (bn1) | 256        | 256         |
| ReLU              | 256        | 256         |
| Layer2 (fc2)      | 256        | 128         |
| BatchNorm1d (bn2) | 128        | 128         |
| ReLU              | 128        | 128         |
| Layer3 (fc3)      | 128        | 4           |
| Tanh              | 4          | 4           |

The critic consists of 3 fully connected layer with 256, 128 and 1 units, respectively. The first two fully connected layers are followed with a ReLU nonlinear activation function. The critic takes state (size = 24) and action (size = 2) as input and output action value (size=1). The first layer takes state as input (size = 24) and the second layer takes action and concatenate it with previous layer output as input (size = 256 + 2).

| Critic        | Input size | Output size |
| ------------- | ---------- | ----------- |
| Layer1 (fcs1) | 24         | 256         |
| ReLU          | 256        | 256         |
| Layer2 (fc2)  | 256 + 2    | 128         |
| ReLU          | 128        | 128         |
| Layer3 (fc3)  | 128        | 1           |



#### MADDPG Hyperparameters

Key hyperparameters in this MADDPG implementation are shown as follows : 

| Hyperparameter      | Value    | Meaning                                                 |
| ------------------- | -------- | ------------------------------------------------------- |
| n_episodes          | 1000     | total max number of training episodes                   |
| max_t               | 1000     | total max number of steps in one episode                |
| BUFFER_SIZE         | int(1e6) | replay buffer size                                      |
| BATCH_SIZE          | 1024     | minibatch size sampled from replay buffer               |
| GAMMA               | 0.99     | discount factor                                         |
| TAU                 | 1e-1     | for soft update of target parameters                    |
| LR_ACTOR            | 5e-4     | learning rate of the actor                              |
| LR_CRITIC           | 5e-4     | learning rate of the critic                             |
| WEIGHT_DECAY        | 0        | L2 weight decay                                         |
| UPDATE_TIMESTEPS    | 10       | timestep period for updating model                      |
| MEMORY_SAMPLE_TIMES | 10       | times to learn from replay buffer for each model update |
| mu                  | 0        | the long-running mean in Ornstein-Uhlenbeck Noise       |
| theta               | 0.15     | the speed of mean reversion in Ornstein-Uhlenbeck Noise |
| sigma               | 0.4      | the volatility parameter in Ornstein-Uhlenbeck Noise    |



### Plot of Rewards

In this MADDPG implementation, the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment is solved in **859** episodes with average score of +0.5 over 100 consecutive episodes, after taking the maximum over both agents.



The plot of reward in training stage is shown as follows :

![reward][image2] 

### Future Work

- **Try different configuration of actor-critic network** : In this implementation, both agents shared the same one shared actor network and one shared critic network. There're other implementations such as two actor networks and one shared critic network for two agents or two actor networks and two critic networks for two agents.
- **Optimization hyperparameters and neural network architecture**：There're still lots of space we can try to adjust hyperparameters and neural network architecture. Different learning rate, replay buffer minibatch, different layers with different units or insertion with batch normalization layer may be a good try.
- **Another algorithms** : Aside from DDPG, there're lots of potential algorithms can be tested. For examples, [PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization), [TRPO](https://arxiv.org/abs/1502.05477) (Trust Region Policy Optimization) or [D4PG](https://arxiv.org/abs/1804.08617) (Distributed Distributional Deterministic Policy Gradients) can be good candidates.
- **Priority replay buffer**：Replay buffer is equally and randomly sampled in this project. However, there're some experience tuples having high rewards that deserve high sampling probability. Implementing [priority experience replay](https://arxiv.org/abs/1511.05952) may help to improve total expected rewards.