import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()



class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
                 env, 
                 dyn_model, 
                 horizon=5, 
                 cost_fn=None, 
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        
        paths = []
        for i in range(self.num_simulated_paths):
            paths.append({})
            paths[i]['observations'] = []
            paths[i]['actions'] = []
            paths[i]['next_observations'] = []
            
        states = np.tile(state, [self.num_simulated_paths, 1])
        actions = np.zeros([self.num_simulated_paths, self.env.action_space.shape[0]])
        
        for i in range(self.horizon):
            
            for j in range(self.num_simulated_paths):
                paths[j]['observations'].append(states[j,:])
                actions[j,:] = self.env.action_space.sample()
                paths[j]['actions'].append(actions[j,:])

            states = self.dyn_model.predict(states, actions)[0]
            
            for j in range(self.num_simulated_paths):
                paths[j]['next_observations'].append(states[j,:])
        
        
        min_cost = np.finfo(float).max
        min_ind = 0
        for i in range(self.num_simulated_paths):
            path = paths[i]
            cost = trajectory_cost_fn(self.cost_fn, path['observations'], path['actions'], path['next_observations'])
            if cost < min_cost:
                min_cost = cost
                min_ind = i
                
        return paths[min_ind]['actions'][0]

