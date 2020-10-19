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


class DQNModel(DistributionalQModel):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kw):
        dropout_rate = 0.2
        num_outputs = 5
        hidden_dim = 10
        tf = try_import_tf()
        super(DQNModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        # Define the core model layers which will be used by the other
        # output heads of DistributionalQModel
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_0 = tf.keras.layers.Dropout(rate=dropout_rate,name="my_layer0")(self.inputs)
        layer_1 = tf.keras.layers.Dense(
            hidden_dim,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(layer_0)
        layer_2 = tf.keras.layers.Dropout(rate=dropout_rate,name="my_layer2")(layer_1)

        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(layer_2)

        self.base_model = tf.keras.Model(inputs = self.inputs, 
                                         outputs = layer_out)
        self.register_variables(self.base_model.variables)

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state


        