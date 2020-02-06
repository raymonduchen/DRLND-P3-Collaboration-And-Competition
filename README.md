[//]: # "Image References"

[image1]: ./Report.assets/agent_tested.gif "TrainedAgent"

# DRLND-P3 : Collaboration and Competition

![Trained Agent][image1]

### Project Details

This is my work for the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) (DRLND) Project 3 : Collaboration and Competition. The goal of this project is training a multiple agent deep reinforcement learning model to control two rackets (agents) to bounce a ball over a net to keep the ball in play as many time steps as possible in a virtual environment [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) from Udacity.

In this project, I trained a multi-agent deep deterministic policy gradient (MADDPG) model for the agent, there's one critic neural network and one actor neural network for both agents and replay buffer is shared for both agents. It can get average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

**Environment**

The virtual environment used in this project is [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) from Udacity, which is similar but not identical to the Tennis environment on the  [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

**Reward**

A reward of **+0.1** is received if an agent hits the ball over the net. A reward of **-0.01** is received if an agent hit the ground or hits the ball out of bound.

**Observation space**

The observation space consists of **8** variables corresponding to position and velocity of the ball and racket. Each agent receives its own, local observation.

**Action space**

The actions consists of **2** continuous variables corresponding to movement toward (or away from) the net, and jumping.

**Goal**

The goal of each agent is to keep the ball in play as long as possible. The task is episodic, and to solve the environment, the agents must get an average score of **+0.5** (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.



### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `DRLND-P2-Collaboration-And-Competition/` folder, and unzip (or decompress) the file. 

### Instructions

#### Train agent

1. Activate the conda environment `drlnd` as established in [Udacity deep reinforcement learning repository](https://github.com/udacity/deep-reinforcement-learning)

2. Open jupyter notebook `Tennis.ipynb`

3. Change kernel to `drlnd` in `Tennis.ipynb`

4. Change environment path in the first code cell, e.g.

   ```
   env = UnityEnvironment('Tennis_Linux/Tennis.x86_64')   # Linux
   ```

5. Execute code cells in `Tennis.ipynb`, and trained model will be saved in `solved_checkpoint_actor.pth` and `solved_checkpoint_critic.pth` as the average score > +0.5.

#### Test agent

1. Activate the conda environment `drlnd` as established in [Udacity deep reinforcement learning repository](https://github.com/udacity/deep-reinforcement-learning)

2. Open jupyter notebook `Tennis_Test.ipynb`

3. Change kernel to `drlnd` in `Tennis_Test.ipynb`

4. Change environment path in the first code cell, e.g.

   ```
   env = UnityEnvironment('Tennis_Linux/Tennis.x86_64')   # Linux
   ```

5. Execute code cells in `Tennis_Test.ipynb`, and the trained model will be tested for 3 times.


