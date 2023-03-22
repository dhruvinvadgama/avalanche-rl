from avalanche.benchmarks import AtariBenchmark
from avalanche.models import dqn
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import EWCPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche_rl.agents import DQNAgent
from avalanche_rl.core import Environment
from avalanche_rl.utils import run_experiment

# create Atari benchmark with Pong, Breakout, and SpaceInvaders games
benchmark = AtariBenchmark(['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4'])

# create DQN agent
agent = DQNAgent(input_shape=(4, 84, 84), num_actions=benchmark.n_classes)

# create EWC plugin for continual learning
ewc_plugin = EWCPlugin(agent)

# create evaluation plugin to log and print metrics
eval_plugin = EvaluationPlugin(
    accuracy_metrics(benchmark.stream_definitions),
    loggers=[],
    collect_all=True)

# create strategy plugin with EWC and evaluation plugins
strategy_plugin = StrategyPlugin([ewc_plugin, eval_plugin])

# create the environment and training strategy
env = Environment(benchmark)
training_strategy = dqn.DQN(
    optimizer='adam',
    criterion='mse',
    replay_buffer_size=100000,
    minibatch_size=32,
    target_update=1000,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.1,
    gamma=0.99,
    learning_rate=0.00025,
    input_shape=(4, 84, 84),
    num_actions=benchmark.n_classes,
    train_freq=4,
    gradient_steps=1,
    double_q=True,
    n_episodes=100,
    max_episode_steps=None)

# train the agent on the benchmark using the environment and training strategy
results = run_experiment(env, agent, training_strategy, strategy_plugin)

print(results)