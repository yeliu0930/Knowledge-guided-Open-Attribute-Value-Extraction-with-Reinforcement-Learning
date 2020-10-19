import os
# os.chdir("/content/drive/My Drive/Knowledge Extraction/szhang37_code/KG_RL")
import tensorflow as tf
import pickle
import random
import pandas as pd 
import numpy as np
import json
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import argparse
from functools import reduce
#### Import ray related package
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.models.tf.visionnet_v2 import VisionNetwork as MyVisionNetwork
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.tune import Trainable
from ray.tune.logger import pretty_print
from ray.tune import run as run_tune
from ray.tune.registry import register_env
import gym
from gym import spaces
from gym.spaces import Discrete, Box
from ray import tune
from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG
### import self-defined function
import similarity_metrics
from utility_function import *
### import Environment
from Environment import KGRLEnv
### Import Customized DQN
from PolicyDQN import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='construtor')
  parser.add_argument('--method', type = str, default='bert',   help='method: bert, bidaf, qanet')
  parser.add_argument("--total_iteration", type=int, default=20, help="number of iterations in KG-RL")
  parser.add_argument("--seed", type=int, default=20201005, help="seed")
  args = parser.parse_args()
  method = args.method ## possible methods: "bert", "bidaf", "qanet"
  total_iteration = args.total_iteration
  seed = args.seed
  ########################################
  ## read preprocessed data(train/test) ##
  ########################################
  train_data = pickle.load(open("./preprocessed_data/train_data.pkl", "rb" ))
  test_data = pickle.load(open("./preprocessed_data/test_data.pkl", "rb" ))
  test_data_types = [test_data[i]['type'] for i in range(len(test_data))]

  do_bert = False
  if method == "bert":
    do_bert = True
  #######################################################
  ### Obtain the pickled predition for train and test ###
  #######################################################
  saving_path ="./preprocessed_data/"
  train_file = "pred_train_%s.pkl" %method ; test_file = "pred_test_%s.pkl" %method
  pred_train = pickle.load(open(saving_path + train_file, "rb" ))
  pred_test = pickle.load(open(saving_path + test_file, "rb" ))



  #### check the length of example_similarity_features should be 14 as mentioned in paper
  ans = '65nm'
  reference_values = ['16nm FinFET', '16nm FinFET', '14nm FinFET', '28纳米', '65nm',
          '28纳米', '16nm FinFET']
  example_similarity_features = get_sim_features(ans, reference_values, do_bert = do_bert)
  len_similarity_features = len(example_similarity_features)

  print("+" * 30 + " For method %s, the scores are " %method + "+" * 30)

  avg_score = []
  oracle_score = []
  max_conf_score = []
  majority_vote_score = []
  first_score = []

  flatten = lambda l: [item for sublist in l for item in sublist]
  for i in range(len(test_data)):
    cur = pred_test['Test_'+str(i)]
    candidate_answer = [cur[j][0][0] for j in range(len(cur))]
    confs = [cur[j][1][0] for j in range(len(cur))]
    if do_bert: ## if do bert the object should be preprocessed as original bert paper
      sims = [similarity_metrics.LevenSim(c, token_word(test_data[i]['o']) ) for c in candidate_answer]
    else:
      sims = [similarity_metrics.LevenSim(c, test_data[i]['o']) for c in candidate_answer]
    # majority vote
    c = Counter(candidate_answer)
    ans_majority_vote, _ = c.most_common()[0]
    first_score.append(sims[0])
    avg_score.append(np.mean(sims))
    max_conf_score.append(sims[np.argmax(confs)])
    majority_vote_score.append(similarity_metrics.LevenSim(ans_majority_vote, 
                                                           test_data[i]['o']))
    oracle_score.append(np.max(sims))

  test_data_types = [test_data[i]['type'] for i in range(len(test_data))]
  GPU_test_index = [i for i in range(len(test_data)) if test_data_types[i] == 'GPUs']
  Game_test_index = [i for i in range(len(test_data)) if test_data_types[i] == 'Games']
  Movie_test_index = [i for i in range(len(test_data)) if test_data_types[i] == 'Movies']
  Phone_test_index = [i for i in range(len(test_data)) if test_data_types[i] == 'phones']

  ## print the scores 

  print("first score is %0.3f (Overall)"%np.mean(np.array(first_score)))
  print("avg score is %0.3f (Overall)"%np.mean(np.array(avg_score)))
  print("mac conf score is %0.3f (Overall)"%np.mean(np.array(max_conf_score)))
  print("majority vote score is %0.3f (Overall)"%np.mean(np.array(majority_vote_score)))
  print("oracle score is %0.3f (Overall)"%np.mean(np.array(oracle_score)))

  index_to_use = GPU_test_index
  print("first score is %0.3f for GPU"%np.mean(np.array(first_score)[index_to_use]))
  print("avg score is %0.3f for GPU"%np.mean(np.array(avg_score)[index_to_use]))
  print("mac conf score is %0.3f for GPU"%np.mean(np.array(max_conf_score)[index_to_use]))
  print("majority vote score is %0.3f for GPU"%np.mean(np.array(majority_vote_score)[index_to_use]))
  print("oracle score is %0.3f for GPU"%np.mean(np.array(oracle_score)[index_to_use]))

  index_to_use = Game_test_index
  print("first score is %0.3f for GAME"%np.mean(np.array(first_score)[index_to_use]))
  print("avg score is %0.3f for GAME"%np.mean(np.array(avg_score)[index_to_use]))
  print("mac conf score is %0.3f for GAME"%np.mean(np.array(max_conf_score)[index_to_use]))
  print("majority vote score is %0.3f for GAME"%np.mean(np.array(majority_vote_score)[index_to_use]))
  print("oracle score is %0.3f for GAME"%np.mean(np.array(oracle_score)[index_to_use]))

  index_to_use = Movie_test_index
  print("first score is %0.3f for MOVIE"%np.mean(np.array(first_score)[index_to_use]))
  print("avg score is %0.3f for MOVIE"%np.mean(np.array(avg_score)[index_to_use]))
  print("mac conf score is %0.3f for MOVIE"%np.mean(np.array(max_conf_score)[index_to_use]))
  print("majority vote score is %0.3f for MOVIE"%np.mean(np.array(majority_vote_score)[index_to_use]))
  print("oracle score is %0.3f for MOVIE"%np.mean(np.array(oracle_score)[index_to_use]))

  index_to_use = Phone_test_index
  print("first score is %0.3f for PHONE"%np.mean(np.array(first_score)[index_to_use]))
  print("avg score is %0.3f for PHONE"%np.mean(np.array(avg_score)[index_to_use]))
  print("mac conf score is %0.3f for PHONE"%np.mean(np.array(max_conf_score)[index_to_use]))
  print("majority vote score is %0.3f for PHONE"%np.mean(np.array(majority_vote_score)[index_to_use]))
  print("oracle score is %0.3f for PHONE"%np.mean(np.array(oracle_score)[index_to_use]))


  """
  Training
  """
  np.random.seed(seed)
  tf.random.set_random_seed(seed)
  ray.init(num_gpus=1, log_to_driver=False, local_mode=True, ignore_reinit_error=True)
  ModelCatalog.register_custom_model("keras_q_model", DQNModel)

  qTrainer = DQNTrainer(env=KGRLEnv, config={# config to pass to env class
      "model": {
          "custom_model": "keras_q_model"
      },
      "seed" : seed,
      "env_config": {"training": True, "idx_to_test":None, "train_data" : train_data,"test_data": test_data,"pred_train":  pred_train, "pred_test" : pred_test, "do_bert" : do_bert},
      "buffer_size":100,
      "lr_schedule": [[0, 0.05], [20, 0.01], [30, 0.005], [50, 0.001]],
      "train_batch_size":100
    })


  
  prev_time = time.time()
  for i in range(total_iteration):
      print("iteration {};".format(i), \
            "%d sec/iteration;" % (time.time()- prev_time), \
            "%d min remaining" % ((total_iteration - i)*(time.time()- prev_time)/60))
      prev_time = time.time()
      qTrainer.train()
      
  print("Training Done!")


  """
  Evaluation
  """

  def evaluation_q(test_data, pred_test, qTrainer):
      rewards = []
      steps = []
      for i in range(len(test_data)):
        env = KGRLEnv({"training": False, "idx_to_test":i, "train_data" : None, "test_data": test_data,"pred_train":  None, "pred_test" : pred_test, "do_bert" : do_bert})
        state = env.state
        done = False
        while not done:
            action = qTrainer.compute_action(state)
            state, reward, done, results = env.step(action)
        rewards.append(reward)
        steps.append(results['steps'])
      return rewards, steps

  reward_list, step_list = evaluation_q(test_data, pred_test, qTrainer)
  avg_reward = np.mean(reward_list)
  avg_steps = np.mean(step_list)

  GPU_reward = np.mean(np.array(reward_list)[GPU_test_index])
  GPU_steps = np.mean(np.array(step_list)[GPU_test_index])

  Movie_reward = np.mean(np.array(reward_list)[Movie_test_index])
  Movie_steps = np.mean(np.array(step_list)[Movie_test_index])

  Game_reward = np.mean(np.array(reward_list)[Game_test_index])
  Game_steps = np.mean(np.array(step_list)[Game_test_index])

  Phone_reward = np.mean(np.array(reward_list)[Phone_test_index])
  Phone_steps = np.mean(np.array(step_list)[Phone_test_index])
  print("Training iteration {}..., \n average reward is {:0.3f},\
  average # of steps is {:0.3f}".format(i, avg_reward, avg_steps))

  print("Average rewards for GPU/Movie/Game/Phone are {:0.3f}/{:0.3f}/{:0.3f}/{:0.3f}"\
        .format(GPU_reward, Movie_reward, Game_reward, Phone_reward))

  print("Average # of steps for GPU/Movie/Game/Phone are {:0.3f}/{:0.3f}/{:0.3f}/{:0.3f}"\
        .format(GPU_steps, Movie_steps, Game_steps, Phone_steps))


