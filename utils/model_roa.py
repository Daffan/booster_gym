# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

from .lipsnet import LipsNet
from .lipmlp import LipMLP

# History Encoder
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size), self.activation_fn,
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), self.activation_fn
        )

    def forward(self, obs):
        # (batch_size, T, num_prop)
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
    def __init__(
            self,
            mlp_input_dim_a,
            actor_hidden_dims,
            activation,
            num_actions,
            num_priv, num_hist,
            num_prop, priv_encoder_dims, network="mlp"):
        super().__init__()
        self.num_actions = num_actions

        # Policy
        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_encoder_layers.append(activation)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv

        self.num_priv = num_priv
        self.num_hist = num_hist
        self.num_prop = num_prop

        # Priv Encoder
        # encoder_dim = 8
        # self.priv_encoder =  nn.Sequential(*[
        #                         nn.Linear(num_priv, 256), activation,
        #                         nn.Linear(256, 128), activation,
        #                         nn.Linear(128, encoder_dim), 
        #                         # nn.Tanh()
        #                         nn.LeakyReLU()
        #                     ])
        
        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)

        # Policy
        if network == "lipsnet":
            self.actor = LipsNet(
                f_sizes=[num_prop + priv_encoder_output_dim, *actor_hidden_dims, num_actions],
                f_hid_nonliear=nn.ReLU,
                f_out_nonliear=nn.Identity,
                global_lips=False,
                k_init=1,
                k_sizes=[num_prop + priv_encoder_output_dim, 32, 1],
                k_hid_act=activation,
                k_out_act=activation,
                loss_lambda=0.1,
                eps=1e-4,
                squash_action=False
            )
        elif network == "mlp":
            actor_layers = []
            actor_layers.append(nn.Linear(num_prop + priv_encoder_output_dim, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for layer_index in range(len(actor_hidden_dims)):
                if layer_index == len(actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                    actor_layers.append(activation)
            self.actor = nn.Sequential(*actor_layers)
        elif network == "lipmlp":
            self.actor = LipMLP(
                [num_prop + priv_encoder_output_dim, *actor_hidden_dims, num_actions],
                activation=activation
            )
    
    def forward(self, obs, hist_encoding=False):
        obs_prop = obs[:, :self.num_prop]
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        actor_input = torch.cat([obs_prop, latent], dim=1)
        return self.actor(actor_input)
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop: self.num_prop + self.num_priv]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))

    def act(self, obs):
        # (batch_size, history_length, num_prop)
        latent = self.infer_hist_latent(obs)
        actor_input = torch.cat([obs[:, -1], latent], dim=1)
        return self.actor(actor_input)

class ActorCriticHistory(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        priv_encoder_dims=[64, 20],
                        activation='elu',
                        init_std=-2.0,
                        network="mlp",
                        **kwargs):
        # if kwargs:
        #     print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticHistory, self).__init__()
        self.num_actions = num_actions

        num_priv = kwargs['num_priv']
        num_hist = kwargs['num_hist']
        num_prop = kwargs['num_prop']

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
            
        self.actor = Actor(
            mlp_input_dim_a,
            actor_hidden_dims,
            activation,
            num_actions,
            num_priv, num_hist,
            num_prop, priv_encoder_dims, network)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_actions), fill_value=init_std), requires_grad=True)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding=False):
        if len(observations.shape) == 3:
            batch_size, seq_len, obs_dim = observations.shape
            observations = observations.reshape(-1, observations.shape[-1])
            mean = self.actor(observations, hist_encoding)
            # reshape back
            mean = mean.reshape(batch_size, seq_len, -1)
        else:
            mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + torch.exp(self.logstd).expand_as(mean))

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False):
        if len(observations.shape) == 3:
            batch_size, seq_len, obs_dim = observations.shape
            observations = observations.reshape(-1, observations.shape[-1])
            actions_mean = self.actor(observations, hist_encoding)
            # reshape back
            actions_mean = actions_mean.reshape(batch_size, seq_len, -1)
        else:
            actions_mean = self.actor(observations, hist_encoding)
        return actions_mean

    def est_value(self, critic_observations, **kwargs):
        if len(critic_observations.shape) == 3:
            batch_size, seq_len, obs_dim = critic_observations.shape
            critic_observations = critic_observations.reshape(-1, critic_observations.shape[-1])
            value = self.critic(critic_observations)
            # reshape back
            value = value.reshape(batch_size, seq_len, -1)
        else:
            value = self.critic(critic_observations)
        return value.squeeze(-1)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None