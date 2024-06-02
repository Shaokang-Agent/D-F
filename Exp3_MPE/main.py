from common.arguments import get_args
from common.utils import make_env

if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    if args.algorithm == "DDPG":
        from Runner.runner_ddpg import Runner
    elif args.algorithm == "MADDPG":
        from Runner.runner_maddpg import Runner
    elif args.algorithm == "DF":
        from Runner.runner_df import Runner
    elif args.algorithm == "MADDPG_DDPG":
        from Runner_adv.runner_maddpg_vs_ddpg import Runner
    elif args.algorithm == "DDPG_DDPG":
        from Runner_adv.runner_ddpg_vs_ddpg import Runner
    elif args.algorithm == "DF_DDPG":
        from Runner_adv.runner_df_vs_ddpg import Runner
    else:
        print("Run None")
    runner = Runner(args, env)
    for i in range(args.round):
        runner.run(i)
