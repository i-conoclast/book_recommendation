import time
import numpy as np
from src.algorithms.linear_full_posterior_samplling import LinearFullPosteriorSampling
from src.bandits.contextual_bandit import run_contextual_bandit

class Hparams(object):
    def __init__(self, dicts):
        for k, v in dicts.items():
            self.__setattr__(k, v)

def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
    """Displays summary statistics of the performance of each algorithm."""

    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('{} bandit completed after {} seconds.'.format(
        name, time.time() - t_init))
    print('---------------------------------------------------')

    performance_pairs = []
    for j, a in enumerate(algos):
        performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
    performance_pairs = sorted(performance_pairs,
                             key=lambda elt: elt[1],
                             reverse=True)
    for i, (name, reward) in enumerate(performance_pairs):
        print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

    print('---------------------------------------------------')
    print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
    print('Frequency of optimal actions (action, frequency):')
    print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
    print('---------------------------------------------------')
    print('---------------------------------------------------')

def main(contexts, 
         rewards,
         opt_rewards,
         opt_actions,
         a0,
         b0,
         lambda_prior,
         initial_pulls):
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    num_actions = np.max(rewards) + 1

    param_dicts = {'num_actions':num_actions,
                    'context_dim': context_dim,
                    'a0': a0,
                    'b0': b0,
                    'lambda_prior': lambda_prior,
                    'initial_pulls': initial_pulls}
    
    hparam = Hparams(param_dicts)
    algos = [LinearFullPosteriorSampling('LinFullPost', hparam)]

    t_init = time.time()
    h_actions, h_rewards = run_contextual_bandit(context_dim, num_actions, dataset, algos)

    display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, 'context')

if __name__ == "__main__":
    pass