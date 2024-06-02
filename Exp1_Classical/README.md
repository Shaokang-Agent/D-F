# Exp1: Classical Scenarios

## Envirments
We first conduct the experiments on $3$ classical scenarios: Job Scheduling, Matthew Effect and Manufacturing Plant. All scenarios have limited resources, thus, the agents encounter the mixed cooperative and competitive relationship with others under the general-sum rewards.

Job Scheduling. The Job Scheduling involves $4$ agents and $1$ resource, situated in a $5 \times 5$ grid world. At the start of each episode, the positions of the resource and agents are randomly initialized in the grid. The objective of all agents is to move to the location of the resource to obtain a $+1$ reward. Otherwise, the reward is $0$. However, only one agent can occupy the resource at any given time. The basic $5$ actions for all agents are: move up, move down, move left, move right and stay in place.

Matthew Effect. The scenario of Matthew Effect involves $10$ agents (pac-men) and $3$ resources (ghosts) in a continuous space. At the start of each episode, the positions of the resources and agents are randomly initialized in the environment. The resources remain stationary while the agents move to consume them, obtaining a $+1$ reward for each successful consumption. Each time an agent successfully consumes the resource, its size and speed will increase by a fixed value until it reaches the preset upper bounds, at which point a new resource is initialized in the environment. Therefore, the agent who gets more resources with larger size and faster speed has the advantage to compete for the following resources, which is known as the Matthew effect. The basic $5$ actions for all agents are: move up, move down, move left, move right and stay in place.

Manufacturing Plant. The scenario of Manufacturing Plant involves $5$ agents and $8$ gems with the map of $8 \times 8$ grid world. There are three types of gem (A,B,C). At the start of each episode, the positions and types of the gems are randomly initialized. If the agent collects a gem, it will obtain a $+0.01$ reward while a new gem with a random type will be initialized in the environment. Additionally, each agent has a unique list of the three gem types required to manufacture a product and obtain a $+1$ reward. Once the product is manufactured, the corresponding gems are consumed. The basic $5$ actions for all agents are: move up, move down, move left, move right and stay in place.

## Installation
```
# Create a python virtual environment with python=3.7.9
conda create -n Exp1 python=3.7.9
# Activate the environment
conda activate Exp1
# Install the requirements
pip install --upgrade pip
pip install -r requirements.txt
```

## Training


```
# Chooose Experiments: edit config.ini
# For example: Training PPO-DF algorithm
python PPO-DF.py 

# Training PPO algorithm:
python PPO.py 

# Training SOTO-ALF algorithm:
python SOTO-ALF-CLDE.py 

...
```

## Results
The rewards of all agents will be stored in the respecitive files: /job, /matthew and /plant.