import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import argparse
import pickle
import math
import random
import copy
import yaml
import time
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from sam_ppo_continuous_efficient import PPO_continuous
from torch.distributions import Uniform
from pprint import pprint

# ── Rimosso: import mujoco as mujoco_py
# Il nuovo binding Python si importa con `import mujoco` ma non serve
# usarlo direttamente; gymnasium/mujoco lo usa internamente.


class NoisyRewardEnv(gym.Wrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyRewardEnv, self).__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        # Gymnasium >= 0.26 restituisce (obs, reward, terminated, truncated, info)
        state, reward, terminated, truncated, info = self.env.step(action)
        noise = np.random.normal(0, self.noise_std)
        reward += noise
        return state, reward, terminated, truncated, info


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        # env.reset() restituisce (obs, info) in Gymnasium >= 0.26
        s, _ = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a
            s_, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
    return evaluate_reward / times


def save_agent(agent, save_path, state_norm, reward_scaling):
    agent.actor.save(f'{save_path}_actor')
    agent.critic.save(f'{save_path}_critic')
    with open(f'{save_path}_state_norm', 'wb') as file1:
        pickle.dump(state_norm, file1)
    with open(f'{save_path}_reward_scaling', 'wb') as file2:
        pickle.dump(reward_scaling, file2)
    print(f'model saved : {save_path}')


def main(args, number):

    total_steps = 0
    iteration = 0
    max_value = -np.inf
    convergence_threshold = args.convergence_threshold
    convergence_iteration = None

    seed, GAMMA = args.seed, args.GAMMA

    # ── Seed: in Gymnasium il seed si passa a env.reset(), non a env.seed()
    env         = gym.make(args.env)
    env_evaluate = gym.make(args.env)
    env_reset    = gym.make(args.env)   # usato per reset con seed iniziale

    # Seed action spaces (ancora supportato)
    env.action_space.seed(seed if seed is not None else 0)
    env_evaluate.action_space.seed(seed if seed is not None else 0)
    env_reset.action_space.seed(seed if seed is not None else 0)

    np.random.seed(seed)
    torch.manual_seed(seed if seed is not None else 0)
    torch.cuda.manual_seed_all(seed if seed is not None else 0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env.spec.max_episode_steps  # _max_episode_steps deprecato
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    if args.reward_noise:
        noise_std_train = args.reward_noise_std
        env = NoisyRewardEnv(env, noise_std=noise_std_train)
        print(f"Training with reward noise: std={noise_std_train}")

    evaluate_num = 0
    evaluate_rewards = []
    update_time = []
    total_steps = 0
    max_value = -np.inf

    if args.use_sam:
        save_path = (
            f"./models/SAM_PPO_reward_{args.reward_noise_std}_{args.env}_rho_{args.rho}_{GAMMA}"
            if args.reward_noise else
            f"./models/SAM_PPO_{args.env}_rho_{args.rho}_{GAMMA}"
        )
    else:
        save_path = (
            f"./models/PPO_reward_{args.reward_noise_std}_{args.env}_{GAMMA}"
            if args.reward_noise else
            f"./models/PPO_{args.env}_{GAMMA}"
        )

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    if args.use_sam:
        writer = SummaryWriter(log_dir=(
            f'runs/SAM_PPO_continuous/reward_env_{args.env}_{args.policy_dist}_number_{number}_seed_{seed}_GAMMA_{GAMMA}'
            if args.reward_noise else
            f'runs/SAM_PPO_continuous/env_{args.env}_{args.policy_dist}_number_{number}_seed_{seed}_GAMMA_{GAMMA}'
        ))
    else:
        writer = SummaryWriter(log_dir=(
            f'runs/PPO_continuous/reward_env_{args.env}_{args.policy_dist}_number_{number}_seed_{seed}_GAMMA_{GAMMA}'
            if args.reward_noise else
            f'runs/PPO_continuous/env_{args.env}_{args.policy_dist}_number_{number}_seed_{seed}_GAMMA_{GAMMA}'
        ))

    state_norm = Normalization(shape=args.state_dim)
    if args.use_reward_norm:
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    start_time = time.time()
    converge = False

    while total_steps < args.max_train_steps:
        iteration += 1

        # ── Reset: Gymnasium >= 0.26 restituisce (obs, info)
        #    Il seed viene passato qui, non con env.seed()
        s, _ = env.reset(seed=seed if iteration == 1 else None)

        # ── Posizione x: nuovo mujoco binding usa env.unwrapped.data.qpos
        #    (non più env.sim.data.qpos come in mujoco_py)
        try:
            x_pos = np.array([env.unwrapped.data.qpos[0]])
        except AttributeError:
            x_pos = np.array([0.0])  # fallback se l'env non espone qpos

        s_org = copy.deepcopy(s)

        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False

        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            # ── Step: Gymnasium >= 0.26 restituisce 5 valori
            s_, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # x_position è ancora disponibile in info per gli env MuJoCo
            if 'x_position' in info:
                x_pos = np.array([info['x_position']])

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # dw = done per cause proprie (caduta), non per timeout
            dw = terminated and (episode_steps != args.max_episode_steps)

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = copy.deepcopy(s_)
            if args.use_state_norm:
                s_org = copy.deepcopy(state_norm.denormal(s_, update=False))
            total_steps += 1

            if replay_buffer.count == args.batch_size:
                update_time_start = time.time()
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0
                update_time_finish = time.time()
                update_time_per_1 = update_time_finish - update_time_start
                update_time.append(update_time_per_1)
                print(f'time per 1 update : {update_time_per_1}')

            if args.evaluate_freq and total_steps % args.evaluate_freq == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print(
                    f"evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward} \t "
                    f"elapsed_time:{elapsed_time:.1f}s \t episode_step:{episode_steps} \t "
                    f"total_steps:{total_steps}"
                )
                writer.add_scalar('step_rewards_{}'.format(args.env), evaluate_rewards[-1], global_step=total_steps)

                if args.save_freq and evaluate_num % args.save_freq == 0:
                    if args.use_sam:
                        np.save(
                            f'./data_train/SAM_PPO_continuous_{args.policy_dist}_env_{args.env}'
                            f'_number_{number}_seed_{seed}_GAMMA_{GAMMA}.npy',
                            np.array(evaluate_rewards)
                        )
                    else:
                        np.save(
                            f'./data_train/PPO_continuous_{args.policy_dist}_env_{args.env}'
                            f'_number_{number}_seed_{seed}_GAMMA_{GAMMA}.npy',
                            np.array(evaluate_rewards)
                        )

                if evaluate_reward > max_value:
                    save_agent(agent, save_path, state_norm,
                               reward_scaling if args.use_reward_scaling else None)
                    max_value = evaluate_reward

                if evaluate_reward >= convergence_threshold and convergence_iteration is None:
                    convergence_iteration = iteration
                    convergence_steps = total_steps
                    print(f"Convergence reached at iteration {convergence_iteration}, "
                          f"total steps: {convergence_steps}")
                    converge = True

        if converge:
            break

    end_time = time.time()
    total_training_time = end_time - start_time
    average_time_per_iteration = total_training_time / iteration if iteration > 0 else 0

    print(f"Total training time: {total_training_time / 3600:.2f} hours")
    print(f"Iterations to convergence: {convergence_iteration}")
    print(f"Total iterations: {iteration}")
    print(f"Average time per iteration: {average_time_per_iteration:.2f} seconds")
    if update_time:
        print(f"Average update time: {sum(update_time) / len(update_time):.4f} seconds")


if __name__ == '__main__':
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config", type=str, default="configs/Hopper.yaml",
                                help="Path to the config file")
    initial_args, remaining_argv = initial_parser.parse_known_args()

    with open(initial_args.config, 'r') as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(
        parents=[initial_parser],
        description="Evaluation for Action Robustness in SAM-PPO-continuous"
    )

    parser.add_argument("--use_adv_norm",        action='store_true', help="Trick 1: advantage normalization")
    parser.add_argument("--use_state_norm",       action='store_true', help="Trick 2: state normalization")
    parser.add_argument("--use_reward_norm",      action='store_true', help="Trick 3: reward normalization")
    parser.add_argument("--use_reward_scaling",   action='store_true', help="Trick 4: reward scaling")
    parser.add_argument("--use_lr_decay",         action='store_true', help="Trick 6: learning rate decay")
    parser.add_argument("--use_grad_clip",        action='store_true', help="Trick 7: gradient clip")
    parser.add_argument("--use_orthogonal_init",  action='store_true', help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps",         action='store_true', help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh",             action='store_true', help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha",       action='store_true', help="Trick 11: adaptive entropy regularization")
    parser.add_argument("--use_sam",              action='store_true', help="Use SAM optimizer")
    parser.add_argument("--reward_noise",         action='store_true', help="Train with reward noise")

    parser.add_argument("--env",              type=str,   help="HalfCheetah-v5/Hopper-v5/Walker2d-v5")
    parser.add_argument("--next_steps",       type=int,   help="Number of next states")
    parser.add_argument("--random_steps",     type=int,   help="Uniformly sample action within random steps")
    parser.add_argument("--max_train_steps",  type=int,   help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq",    type=float, help="Evaluate the policy every N steps")
    parser.add_argument("--save_freq",        type=int,   help="Save frequency")
    parser.add_argument("--policy_dist",      type=str,   help="Beta or Gaussian")
    parser.add_argument("--batch_size",       type=int,   help="Batch size")
    parser.add_argument("--mini_batch_size",  type=int,   help="Minibatch size")
    parser.add_argument("--hidden_width",     type=int,   help="Number of neurons in hidden layers")
    parser.add_argument("--lr_a",            type=float, help="Learning rate of actor")
    parser.add_argument("--lr_c",            type=float, help="Learning rate of critic")
    parser.add_argument("--gamma",           type=float, help="Discount factor")
    parser.add_argument("--lamda",           type=float, help="GAE parameter")
    parser.add_argument("--epsilon",         type=float, help="PPO clip parameter")
    parser.add_argument("--K_epochs",        type=int,   help="PPO parameter")
    parser.add_argument("--entropy_coef",    type=float, help="Trick 5: policy entropy")
    parser.add_argument("--weight_reg",      type=float, help="Regularization for weight of critic")
    parser.add_argument("--seed",            type=int,   help="Seed")
    parser.add_argument("--GAMMA",           type=str,   help="File name")
    parser.add_argument("--rho",             type=float, help="Rho value for SAM optimizer")
    parser.add_argument("--convergence_threshold", type=float, default=8000,
                        help="Desired average reward for convergence")
    parser.add_argument("--reward_noise_std", type=float, default=0.1,
                        help="Standard deviation of reward noise during training")

    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)

    print("\n===== Evaluation Configuration =====")
    pprint(vars(args))
    print("====================================\n")

    os.makedirs("./models",     exist_ok=True)
    os.makedirs("./data_train", exist_ok=True)

    main(args, number=1)