import argparse
import numpy as np
from rl_lib.swiss_round.agent import DQNAgent
from rl_lib.swiss_round.environment import SwissRoundEnv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args')
    
    # Environment arguments
    parser.add_argument('--n_teams', type=int, default=18)
    parser.add_argument('--n_rounds', type=int, default=6)
    parser.add_argument('--threshold_ranks', nargs='+', type=int, default= [4,12])
    parser.add_argument('--threshold_rewards', nargs='+', type=int, default= [30,20])
    parser.add_argument('--max_draw_probability', type=float, default=0.3)
    parser.add_argument('--max_strength', type=float, default=2)
    parser.add_argument('--strength_decay_factor', type=float, default=0.9)
    parser.add_argument('--strength_decay_method', type=str, choices = ['linear','exponential'], default='linear')
    
    # Agent arguments
    parser.add_argument('--agent_id', type=int, default=0)
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default= [256,128,64])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epsilon_start', type=float, default=1)
    parser.add_argument('--epsilon_end', type=float, default=0.02)
    parser.add_argument('--epsilon_decay', type=float, default=0.9995)
    parser.add_argument('--train_episodes', type=int, default=20000)
    parser.add_argument('--test_episodes', type=int, default=2000)
    
    # Log arguments
    parser.add_argument('--log_dir', type=str, default="/home/admin/code/arnaud-odet/2_projets/reinforcement_learning/logs")
    
    

    args = parser.parse_args()

    for k, value in args._get_kwargs():
        if value is not None:
            print(k, value)
    
    if args.strength_decay_method == 'linear':
        team_strengths = np.linspace(args.max_strength, 0, args.n_teams)
    else :
        team_strengths = [args.max_strength * args.strength_decay_factor ** i for i in range(args.n_teams)]
    
    env = SwissRoundEnv(
        n_teams = args.n_teams,
        n_rounds = args.n_rounds,
        team_strengths = team_strengths,
        threshold_ranks = args.threshold_ranks,
        bonus_points = args.threshold_rewards,
        agent_id = args.agent_id,
        max_draw_probability = args.max_draw_probability
    )
    
    agent = DQNAgent(env,
                 hidden_dims=args.hidden_sizes,
                 dropout= args.dropout,
                 batch_size= args.batch_size,
                 lr = args.lr,
                 buffer_size=args.buffer_size,
                 gamma = args.gamma,
                 epsilon_start=args.epsilon_start,
                 epsilon_end=args.epsilon_end,
                 epsilon_decay=args.epsilon_decay,
                 log_dir = args.log_dir)
    agent.train(n_episodes=args.train_episodes)
    agent.evaluate(n_episodes=args.test_episodes)