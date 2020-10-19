import pickle
import random
import pandas as pd 
import numpy as np
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
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
#from databunch import *
import tensorflow as tf
from utility_function import *



class KGRLEnv(gym.Env):
    def _build_init(self, kg_dir = "./related_triples_by_relation/"):
        """
        when testing, use idx_to_test, otherwise randomly sample a training data
        """
        if self.training:
          idx = np.random.choice(range(len(self.train_data))) 
          self.entry = self.train_data[idx]
        else:
          idx = self.idx_to_test
          self.entry = self.test_data[idx]
        self.query = self.entry['s'] + ' ' + self.entry['p']
        self.text_list = self.entry['corpus']
        ######################################################
        ## obtain the answer from extraction system output ###
        ######################################################
        if self.training:
          self.answer_list = self.pred_train[self.entry['id']] 
        else:
          self.answer_list = self.pred_test[self.entry['id']]

        assert len(self.text_list) == len(self.answer_list), "Wrong, text length %d, answer length %d" %(len(self.text_list), len(self.answer_list))

        self.text_answer = [[self.text_list[i], self.answer_list[i]] for i in range(len(self.text_list))]
        
        self.max_index = len(self.text_list)
        ### #####################################################################
        ## initialize the index of current/new candidate as 0/1 respectively. ###
        #########################################################################
        self.cur_index = 0
        self.new_index = 1
        self.cur = self.text_answer[self.cur_index]
        try:
          self.new = self.text_answer[self.new_index]
        except:
          ####################################################################
          ## exception would happen when size of raw text is less than 2. ####
          ## which cannot happen in preprocessed data ########################
          ####################################################################
          self.new =  self.cur
        self.curans = self.cur[1][0]
        self.newans = self.new[1][0]
        self.answer_seen = self.cur[1][0]
        self.truth = "".join(self.entry['o'])

        #################################################################
        ## if do bert, we need to squeeze the space #####################
        #################################################################
        if self.do_bert:
          self.truth = token_word(self.truth)
        # get reference values
        #os.chdir('/content/drive/My Drive/Knowledge Extraction/related_triples_by_relation')
        filename = "%s.csv" % self.entry['p']
        related_triples_to_use = pd.read_csv(kg_dir + filename, sep='\t', header = None)
        self.reference_values = related_triples_to_use[2].values


    def __init__(self, env_config, T= 20, len_similarity_features = 14):
      
        """
        initialize the environment
        """
        self.idx_to_test = env_config["idx_to_test"]
        self.len_similarity_features = len_similarity_features
        self.training = env_config["training"]
        self.train_data = env_config["train_data"]
        self.test_data = env_config["test_data"]
        self.pred_train = env_config["pred_train"]
        self.pred_test = env_config["pred_test"]
        self.do_bert =env_config["do_bert"]
        self._counter = 0 # For terminating the episode
        self._build_init()
        self.state = self.getState(self.cur, self.new)
        self._horizon = env_config.get("T", T)
        self._setup_spaces()

    def _setup_spaces(self):
        ##############
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, 
                                            [2 + 1 + 2 * self.len_similarity_features, ])
        ##############

    def step(self, action):
        self.new_index += 1
        if self.new_index >= self.max_index: # exceed the given size will stop (10 in paper)
            reward = similarity_metrics.LevenSim(self.curans[0], self.truth)
            done = True
            return self.state, reward, done, {"final_answer":self.curans,
                                              "steps": self.new_index}

        else:
          if action == 0: # still use old current as current
              self.new = self.text_answer[self.new_index]
              self.newans = self.new[1][0]
              self.state = self.getState(self.cur, self.new)
              reward = 0

          elif action == 1: # accept new as current
              self.cur_index = self.new_index - 1
              self.cur = self.text_answer[self.cur_index]
              self.curans = self.cur[1][0]
              self.new = self.text_answer[self.new_index]
              self.newans = self.new[1][0]
              self.state = self.getState(self.cur, self.new)
              reward = 0
          else:              
              reward = similarity_metrics.LevenSim(self.curans[0], self.truth)
              done = True
              return self.state, reward, done, {"final_answer":self.curans,
                                              "steps": self.new_index}
          self._counter += 1
          done = self._counter >= self._horizon
          return self.state, reward, done,  {"final_answer":self.curans,
                                              "steps": self.new_index}

    def reset(self):
        self._build_init()
        self.state = self.getState(self.cur, self.new)
        self._counter = 0
        return self.state

    def getState(self, cur, new):
        # input: current best text and answer, new text and answer that seen before
        # output: state
        curans = cur[1][0]
        newans = new[1][0]
        # state (1) confidence scores (dim: K*2)
        curconf = cur[1][1]  
        newconf = new[1][1]
        # state (2) similarity between texts (dim: 1)
        textsim = [self.textSimilarity(cur[1], new[1])]
        # state (3) 
        flatten = lambda l: [item for sublist in l for item in sublist]
        ref_score_cur = flatten([get_sim_features(i, self.reference_values, do_bert = self.do_bert) for i in curans])
        ref_score_new = flatten([get_sim_features(i, self.reference_values, do_bert = self.do_bert) for i in newans])
        # combine all states
        state = curconf + newconf + ref_score_cur + ref_score_new + textsim
        return state

    # function to compute cosine similarity bewteen two texts
    def textSimilarity(self, text1, text2):
        corpus = [text1, text2]
        vectorizer = TfidfVectorizer()
        try:
          tfidf = vectorizer.fit_transform(corpus)
          words = vectorizer.get_feature_names()
          similarity_matrix = cosine_similarity(tfidf)
          similarity = similarity_matrix[0][1]
        except:
          similarity = 0
        return similarity
