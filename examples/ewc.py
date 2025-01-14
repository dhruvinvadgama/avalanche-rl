"""
    Example of replicating a small-scale (single gpu) version of the experiments
    presented in EWC original paper https://arxiv.org/abs/1612.00796 using AvalancheRL.
    The experiment interleaves learning of multiple atari games with a single network 
    (using a 'cumulative' action space) with an added penalty given by the Fisher
    importance matrix, which discourages updates to the parameters which are most
    important for a previous task. 
"""
from avalanche_rl.training.strategies.buffers import ReplayMemory
import torch
from torch.optim import Adam
from avalanche_rl.training.plugins.evaluation import RLEvaluationPlugin
from avalanche_rl.training.strategies.dqn import DQNStrategy, default_dqn_logger
from avalanche_rl.training.strategies.env_wrappers import ReducedActionSpaceWrapper
from avalanche_rl.benchmarks.generators.rl_benchmark_generators import atari_benchmark_generator
from avalanche_rl.training.plugins.ewc import EWCRL
from avalanche_rl.logging import TensorboardLogger
from avalanche_rl.models.dqn import EWCConvDeepQN
from avalanche_rl.training.plugins.strategy_plugin import RLStrategyPlugin
from avalanche_rl.training.strategies.rl_base_strategy import Timestep
from avalanche_rl.evaluation.metrics.reward import GenericFloatMetric
import json

import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    device = torch.device('cuda:0')

    # let's simplify things a bit for the agent: both games can be played with 3 (or 2 without considering NOOP) 
    # actions but the action space is unecessarily big due to action keys ordering; we then reduce the action space
    # to 3 actions and re-map LEFT-RIGHT actions so that we skip FIRE. Actions are re-mapped before the step() method is called.  
    action_space = 3

    def action_wrapper_class(env): return ReducedActionSpaceWrapper(
        env, action_space_dim=action_space, action_mapping={1: 2, 2: 3})

    n_envs = 1
    # frameskipping is done in wrapper
    scenario = atari_benchmark_generator(
        ['BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4'],
        n_parallel_envs=n_envs, frame_stacking=True,
        normalize_observations=True, terminal_on_life_loss=True,
        n_experiences=2, extra_wrappers=[action_wrapper_class],
        eval_envs=['BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4'])

    # let's instatiate an external replay memory
    memory_size = 10000
    memory = ReplayMemory(size=memory_size, n_envs=n_envs)
    ewc_plugin = EWCRL(400., memory, mode='separate',
                       start_ewc_after_experience=2)

    # log to tensorboard
    tb_logger = TensorboardLogger("/tmp/tb_data")

    # keep track of the loss
    ewc_penalty_metric = GenericFloatMetric(
        'loss', 'Ewc Loss', reset_value=0., emit_on=['after_backward'],
        update_on=['before_backward'])

    evaluator = RLEvaluationPlugin(
        *default_dqn_logger.metrics, ewc_penalty_metric,
        loggers=default_dqn_logger.loggers + [tb_logger])

    # here we'll have task-specific biases & gains per layer (2 since we're learning 2 games)
    model = EWCConvDeepQN(4, (84, 84), action_space, n_tasks=2, bias=True)
    print('Model', model)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # a custom plugin to show some functionalities: halve initial epsilon (for eps-greedy action-selection)
    # every two training experiences, so that more exploit is done
    class HalveEps(RLStrategyPlugin):

        def __init__(self):
            super().__init__()

        def after_training_exp(self, strategy: DQNStrategy, **kwargs):
            if strategy.training_exp_counter % 2 == 1:
                strategy._init_eps = strategy._init_eps / 2

    # adapted hyperparams from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
    # we run 6 experiences, alternating between the 2 games: the first two are longer (1e5 steps) the rest
    # are shorter (3e4 steps)
    strategy = DQNStrategy(
        model, optimizer,
        per_experience_steps=[Timestep(int(0.5e3)),
                              Timestep(int(0.5e3))] +
                             [Timestep(int(0.5e3)),
                              Timestep(int(0.5e3))] * 2, batch_size=32, exploration_fraction=.15,
        final_epsilon=.01, max_steps_per_rollout=4,
        plugins=[ewc_plugin, HalveEps()],
        # external replay memory is automatically filled with initial size and reset on new experience
        initial_replay_memory=memory, replay_memory_init_size=4000, double_dqn=True,
        target_net_update_interval=1000, eval_every=int(0.25e3),
        eval_episodes=4, evaluator=evaluator, device=device)


    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)

        results.append(strategy.train(experience, [scenario.test_stream]))
        # print('Training completed')
        # save model and optimizer
        #torch.save(
        #    {'model': model.state_dict(),
        #     'optim': optimizer.state_dict()},
        #    'pong-breakout.pt')

    # store metrics
    metrics = strategy.evaluator.get_all_metrics()
    #fname = f'metrics_ewc_example.json'
    #with open(fname, 'w') as f:
    #    json.dump(metrics, f, indent=4)

print('Training completed')

print(f'Length of results: {len(results)}')
print(f'Result type: {type(results)}')
# print(f'Results: \n {metrics}')

#for key, value in metrics.items():
#    print(f"{key}: {value}")

print(f'Results: \n {results}')


# use for saving it in TXT file
#with open("result.txt", 'w') as f:
#   for key, value in metrics.items():
#       f.write('%s:%s\n' % (key, value))

"""
x_data = metrics['[Train] Mean Reward (last 10 steps)'][0]
y_data = metrics['[Train] Mean Reward (last 10 steps)'][1]

# plot the data
plt.plot(x_data, y_data)

# add axis labels and title
plt.xlabel('Steps')
plt.ylabel('Mean Reward')
plt.title('Mean Reward (last 10 steps)')

# show the plot
plt.show() """

# np.save('results.npy', metrics)


train_mean_reward_steps = metrics['[Train] Mean Reward (last 10 steps)'][0]
train_mean_reward_vals = metrics['[Train] Mean Reward (last 10 steps)'][1]

eval_mean_reward_steps = metrics['[Eval] Mean Reward (last 4 steps)'][0]
eval_mean_reward_vals = metrics['[Eval] Mean Reward (last 4 steps)'][1]


plt.plot(train_mean_reward_steps, train_mean_reward_vals, label='train')
plt.plot(eval_mean_reward_steps, eval_mean_reward_vals, label='eval')
plt.xlabel('Steps')
plt.ylabel('Mean Reward')
plt.title('Mean Reward vs Steps')
plt.legend()
plt.show()


train_max_reward_steps = metrics['[Train] Max Reward (last 10 steps)'][0]
train_max_reward_vals = metrics['[Train] Max Reward (last 10 steps)'][1]

eval_max_reward_steps = metrics['[Eval] Std Reward (last 4 steps)'][0]
eval_max_reward_vals = metrics['[Eval] Std Reward (last 4 steps)'][1]

# plot max rewards for train and eval
plt.plot(train_max_reward_steps, train_max_reward_vals, label='train')
plt.plot(eval_max_reward_steps, eval_max_reward_vals, label='eval')
plt.xlabel('Steps')
plt.ylabel('Max Reward')
plt.title('Max Reward vs Steps')
plt.legend()
plt.show()