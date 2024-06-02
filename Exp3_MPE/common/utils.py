import numpy as np
from multiagent_particle_envs.multiagent.multi_discrete import MultiDiscrete

def make_env(args):
    from multiagent_particle_envs.multiagent.environment import MultiAgentEnv
    import multiagent_particle_envs.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # env = MultiAgentEnv(world)
    args.n_players = env.n
    if args.scenario_name == "simple_tag":
        args.num_adversaries = 1
        args.n_agents = env.n - args.num_adversaries
        args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
        args.obs_shape.append(env.observation_space[args.n_agents].shape[0])
        action_shape = []
        for content in env.action_space:
            action_shape.append(content.n)
        args.action_shape = action_shape
    if args.scenario_name == "simple_world_comm":
        args.num_adversaries = 2
        args.n_agents = env.n - args.num_adversaries
        print(args.n_agents, args.num_adversaries)
        args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
        args.obs_shape.append(env.observation_space[args.n_agents].shape[0])
        args.obs_shape.append(env.observation_space[args.n_agents+1].shape[0])
        action_shape = []
        for content in env.action_space:
            if isinstance(content, MultiDiscrete):
                action_shape.append(9)
            else:
                action_shape.append(content.n)
        args.action_shape = action_shape
    if args.scenario_name == "simple_spread":
        args.n_agents = env.n
        args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
        action_shape = []
        for content in env.action_space:
            action_shape.append(content.n)
        args.action_shape = action_shape
    if args.scenario_name == "simple_adversary":
        args.num_adversaries = 1
        args.n_agents = env.n - args.num_adversaries
        args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.num_adversaries,env.n)]
        args.obs_shape.append(env.observation_space[0].shape[0])
        action_shape = []
        for content in env.action_space:
            action_shape.append(content.n)
        args.action_shape = action_shape[args.num_adversaries:env.n]
        args.action_shape.append(action_shape[0])
    if args.scenario_name == "simple_crypto":
        args.num_adversaries = 1
        args.n_agents = env.n - args.num_adversaries
        args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.num_adversaries,env.n)]
        args.obs_shape.append(env.observation_space[0].shape[0])
        action_shape = []
        for content in env.action_space:
            action_shape.append(content.n)
        args.action_shape = action_shape[args.num_adversaries:env.n]
        args.action_shape.append(action_shape[0])
    if args.scenario_name == "simple_push":
        args.num_adversaries = 1
        args.n_agents = env.n - args.num_adversaries
        args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.num_adversaries,env.n)]
        args.obs_shape.append(env.observation_space[0].shape[0])
        action_shape = []
        for content in env.action_space:
            action_shape.append(content.n)
        args.action_shape = action_shape[args.num_adversaries:env.n]
        args.action_shape.append(action_shape[0])
    args.high_action = 1
    args.low_action = -1
    print(args.obs_shape, args.action_shape)
    return env, args
