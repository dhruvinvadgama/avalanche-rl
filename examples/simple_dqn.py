from avalanche_rl.training.strategies import DQNStrategy
from avalanche_rl.models.dqn import MLPDeepQN, DQNModel
from torch.optim import Adam
from avalanche_rl.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator
import torch
from avalanche_rl.training.strategies.buffers import ReplayMemory
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
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
from avalanche_rl.models.dqn import EWCConvDeepQN
from avalanche_rl.models.dqn import ConvDeepQN
from avalanche_rl.training.strategies.env_wrappers import ReducedActionSpaceWrapper
import matplotlib.pyplot as plt

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
action_space = 3

def action_wrapper_class(env): return ReducedActionSpaceWrapper(
    env, action_space_dim=action_space, action_mapping={1: 2, 2: 3})

n_envs = 1
# frameskipping is done in wrapper
"""scenario = atari_benchmark_generator(
        ['BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4'],
        n_parallel_envs=n_envs, frame_stacking=True,
        normalize_observations=True, terminal_on_life_loss=True,
        n_experiences=6, extra_wrappers=[action_wrapper_class],
        eval_envs=['BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4']) """


tb_logger = TensorboardLogger("/tmp/tb_data")
evaluator = RLEvaluationPlugin(
        *default_dqn_logger.metrics,
        loggers=default_dqn_logger.loggers + [tb_logger])



scenario = gym_benchmark_generator(
        ['CartPole-v1'],
        n_parallel_envs=1, eval_envs=['CartPole-v1'], n_experiences=1)

    # CartPole setting
model = MLPDeepQN(input_size=4, hidden_size=1024,
                      n_actions=2, hidden_layers=2)

# my model new
# model = MLPDeepQN(input_size=4* 84* 84, hidden_size=1024, n_actions=action_space, hidden_layers=2)

print("Model", model)

# DQN learning rate
optimizer = Adam(model.parameters(), lr=1e-3)

strategy = DQNStrategy(model, optimizer, 100, batch_size=32, exploration_fraction=.2, rollouts_per_step=10,
                           replay_memory_size=1000, updates_per_step=10, replay_memory_init_size=1000, double_dqn=False,
                           target_net_update_interval=10, eval_every=100, eval_episodes=10, evaluator=evaluator,
                           device=device, max_grad_norm=None)

    # TRAINING LOOP
print('Starting experiment...')
results = []

for experience in scenario.train_stream:
    print("Start of experience ", experience.current_experience)
    print("Current Env ", experience.env)
    print("Current Task", experience.task_label, type(experience.task_label))
    strategy.train(experience, scenario.test_stream)

print('Training completed')
eval_episodes = 100
print(f"\nEvaluating on {eval_episodes} episodes!")
print(strategy.eval(scenario.test_stream))


metrics = strategy.evaluator.get_all_metrics()


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