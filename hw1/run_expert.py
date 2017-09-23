#!/usr/bin/env python3

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
from gym import wrappers
import load_policy
import matplotlib.pyplot as plt

  
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--hyperparameter', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')


    np.random.seed(0)

    expert_data = {}

    sess = tf.Session()
    with sess:
        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        
        # Model to train on expert data
        # A fully connected architecture worked well
        x = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
        
        W_1_1 = tf.Variable(tf.truncated_normal([env.observation_space.shape[0], 200], stddev=0.1))
        b_1_1 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
        h_1_1 = tf.nn.sigmoid(tf.matmul(x, W_1_1) + b_1_1)
        
        W_1 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
        b_1 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
        h_1 = tf.nn.sigmoid(tf.matmul(h_1_1, W_1) + b_1)
        
        W_2 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
        b_2 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
        h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)
        
        W = tf.Variable(tf.truncated_normal([200, env.action_space.shape[0]], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[env.action_space.shape[0]]))
        y = tf.matmul(h_2, W) + b
        
        y_ = tf.placeholder(tf.float32, [None, env.action_space.shape[0]])
        l2 = tf.reduce_mean(tf.nn.l2_loss(y-y_))


        train_step = tf.train.AdamOptimizer(1e-3).minimize(l2)

    
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        
        expert_returns = np.mean(returns)
        expert_std = np.std(returns)

        # Iterate over training and testing with differing hyperparameter values
        # Our analyzed hyperparameter is number of demonstrations
        hyp_returns = []
        hyp_stddev = []
        for k in range(20):
            # If hyperparam arg is not set, skip all but the last iteration
            if not args.hyperparameter:
                if k != 19:
                    continue
            
            # DAgger iterations loop
            da_returns = []
            da_stddev = []
            for d in range(6):
                # If we don't want DAgger, don't go through for loop
                if not args.dagger:
                    if d != 0:
                        break
                tf.global_variables_initializer().run()
                num_data = int(expert_data['observations'].size/env.observation_space.shape[0])
                # By changing num_data, we change the amount of data used to train
                # If 40 rollouts are used, we should get k+1 rollouts worth of data
                if args.hyperparameter:
                    num_data = int((num_data*(k+1))/20)
                    print('hyperparameter step %d' % (k))
                batch_size = 200
                epochs = 2000
                
                # Lessen the number of epochs for DAgger and hyperparams because it's too slow
                if args.dagger or args.hyperparameter:
                    epochs = 1000
                for j in range(epochs):
                    ind = np.arange(num_data)
                    np.random.shuffle(ind)
                    for i in range(int(num_data/batch_size)):
                        batch_xs = expert_data['observations'][ind[i*batch_size:(i+1)*batch_size]].reshape(batch_size,env.observation_space.shape[0])
                        batch_ys = expert_data['actions'][ind[i*batch_size:(i+1)*batch_size]].reshape(batch_size,env.action_space.shape[0])
            
                        t, loss = sess.run([train_step, l2], feed_dict={x: batch_xs, y_: batch_ys})
                        if j % 10 == 0 and i % 100 == 0:
                            print('step %d, batch %d, loss %g' % (j, i, loss))

        
        
                # Now run the newly trained model              
                returns = []
                observations = []
                actions = []
                for i in range(args.num_rollouts):
                    print('iter', i)
                    obs = env.reset()
                    done = False
                    totalr = 0.
                    steps = 0
                    while not done:
                        action = sess.run(y, feed_dict={x: obs.reshape(1,env.observation_space.shape[0])})
                        observations.append(obs)
                        actions.append(action)
                        obs, r, done, _ = env.step(action)
                        totalr += r
                        steps += 1
                        if args.render:
                            env.render()
                        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                        if steps >= max_steps:
                            break
                    returns.append(totalr)
        
                print('The following data is for the cloned behavior')
                print('returns', returns)
                print('mean return', np.mean(returns))
                print('std of return', np.std(returns))
                if args.hyperparameter:
                    hyp_returns.append(np.mean(returns))
                    hyp_stddev.append(np.std(returns))
                    
                # For dagger, we want to get expert actions for new observations
                if args.dagger:
                    da_returns.append(np.mean(returns))
                    da_stddev.append(np.std(returns))
                    dagger_obs = np.array(observations).reshape(-1,env.observation_space.shape[0])
                    dagger_action = policy_fn(dagger_obs)
                    dagger_action = dagger_action.reshape(-1,1,env.action_space.shape[0])
                    expert_data['observations'] = np.concatenate((expert_data['observations'], dagger_obs), axis=0)
                    expert_data['actions'] = np.concatenate((expert_data['actions'], dagger_action), axis=0)
                    print('DAgger step %d complete' % (d))
                    
            # Plot results vs DAgger iteration
            if args.dagger:
                plt.figure()
                plt.errorbar(np.arange(len(da_returns))+1, da_returns, yerr=da_stddev, label="DAgger")
                plt.errorbar(np.arange(len(da_returns))+1, np.ones(len(da_returns))*expert_returns, yerr=expert_std, label="Expert")
                plt.errorbar(np.arange(len(da_returns))+1, np.ones(len(da_returns))*da_returns[0], yerr=da_stddev[0], label="Cloning")
                plt.legend(loc='best')
                plt.title('Returns vs DAgger Iteration')
                plt.ylabel('Returns')
                plt.xlabel('DAgger Iteration Number')
                plt.show()

            
        # Plot results vs hyperparameter
        if args.hyperparameter:
            plt.figure()
            plt.errorbar(np.arange(len(hyp_returns))+1, hyp_returns, yerr=hyp_stddev)
            plt.title('Behaviorial Cloning Performance vs Demonstations')
            plt.ylabel('Returns')
            plt.xlabel('Number of Rollouts of Data')
            plt.show()


if __name__ == '__main__':
    main()
