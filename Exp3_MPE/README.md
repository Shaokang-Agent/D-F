# Exp3: Multi-Agent Particle Environments (MPE)

## Envirments
we conduct experiments in the following $4$ basic testing scenarios: 

1\) Physical deception with $1$ adversary agent, $3$ cooperative agents, and $2$ landmarks. One of the landmarks is the ``target landmark''. Cooperative agents receive rewards based on their distance from the target landmark but receive negative rewards if the adversary agent approaches it. The adversary agent gets a reward based on its distance from the target landmark but does not know which landmark is the target. 

2\) Covert communication with $2$ cooperative agents (Alice and Bob), $1$ adversary agent (Eve). Alice must send a private message to Bob over a public channel. Alice and Bob receive rewards based on how well Bob reconstructs the message, but they receive negative rewards if Eve can reconstruct the message. Alice and Bob have a private key that they must learn to use to encrypt the message. 

3\) Predator-prey with $1$ prey agent and $3$ cooperative predator agents. The cooperative predator agents must learn to hit the prey agent together because the prey agent is smaller and moves faster. 

4\) Keep-away with $3$ cooperative agents, $1$ adversary agent, $2$ landmarks. The cooperative agents are rewarded based on the distance to the landmark. The adversary can push the cooperative agents away from the landmark and will be rewarded if it is closer to the landmark. 

## Installation
```
# Create a python virtual environment with python=3.6.12
conda create -n Exp3 python=3.6.12
# Activate the environment
conda activate Exp3
# Install the requirements
pip install --upgrade pip
pip install -r requirements.txt
```

## Training
```
# Chooose Experiments: edit parser (--scenario_name) in /common/arguments.py

# Chooose algorithm: edit parser (--algorithm) in /common/arguments.py such as DDPG, MADDPG and DF
# Battle with random policy 
python main.py --algorithm=DDPG
python main.py --algorithm=MADDPG
python main.py --algorithm=DF

# Battle with pre-trained DDPG policy
python main.py --algorithm=DDPG_DDPG
python main.py --algorithm=MADDPG_DDPG
python main.py --algorithm=DF_DDPG
```

## Results
The rewards of all agents will be stored in the respecitive files: /data.

The TensorBoard UI files is stored in /runs