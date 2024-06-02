# Exp2: Sequential Social Dilemmas (SSD)

## Envirments
The environment is a 2D grid game with the partially observable state with the picture of $15 \times 15 \times 3$. The action space is a discrete space that includes $7$ basic motions: move up, move down, move left, move right, stay, rotate clockwise and rotate counterclockwise. Each agent intends to collect more apples in the map and each apple responds with a $+1$ reward. 

Cleanup Environment. Apart from the basic motional actions, it has two specific actions: one is to fire a beam to attack others and the other is to clean up the river. Each agent is trying to collect more apples in the map and each apple responds with a $+1$ reward. However, as rivers become more polluted over time, the rate of apple regeneration decreases. Therefore, each agent must sacrifice its own profit by cleaning the river to ensure more apples in the future, resulting in a $-1$ reward. Moreover, if an agent is attacked by the beam, it incurs a penalty of $-50$. It is crucial to balance competition and cooperation to achieve not only short-term individual rewards but also long-term social welfare, as an extreme case of all agents being destroyed implies no one can clean the river.

Harvest Environment. Unlike the Cleanup environment, the Harvest environment only has the attack action rather than clean action. The objective is also to collect apples, which yield a reward of $+1$. However, the respawn rate of apples solely depends on the number of apples nearby, and when a new episode begins, the number of apples resets to the initial state. The competitive and cooperative relationship involves balancing the exploitation of apple resources to ensure long-term development. If all agents act greedily and collect nearby apples as quickly as possible, the respawn rate will decrease significantly, and all agents will have fewer opportunities to collect apples in the future. This highlights the need to balance short-term individual rewards with long-term social welfare.

## Installation
```
# Create a python virtual environment with python=3.7.9
conda create -n Exp2 python=3.7.9
# Activate the environment
conda activate Exp2
# Install the requirements
pip install --upgrade pip
pip install -r requirements.txt
```

## Training


```
# Chooose Experiments: edit parser (--env) in main.py
# Chooose algorithm: edit parser (--algorithm) in main.py such as DQN, DQN-AVG, DQN-MIN, DQN-RMF, DQN-IA, SOCIAL, QMIX, DF, PPO, MAPPO, DDPG, MADDPG.

# Training
python run_scripts/main.py --algorithm=DF
```

## Results
The rewards of all agents will be stored in the respecitive files: /data.

The TensorBoard UI files is stored in /runs