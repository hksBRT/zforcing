import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp
import keras
import os
import pdb
import math

tf.logging.set_verbosity(3)

class LReLU(tf.keras.layers.Layer):
    def __init__(self, c=1.0 / 3):
        super(LReLU, self).__init__()
        self.c = c

    def forward(self, x):
        x = tf.keras.layers.LeakyReLU(x, alpha=self.c)
        return tf.clip_by_value(x, -3.0, 3.0)

def log_prob_gaussian(x, mu, log_vars, mean=False):
    lp = - 0.5 * math.log(2 * math.pi) \
        - log_vars / 2 - (x - mu) ** 2 / (2 * tf.exp(log_vars))
    if mean:
        return tf.reduce_mean(lp, -1)
    return tf.reduce_sum(lp, -1)

def gaussian_kld(mu_left, logvar_left, mu_right, logvar_right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (tf.exp(logvar_left) / tf.exp(logvar_right)) +
                        ((mu_left - mu_right)**2.0 / tf.exp(logvar_right)) - 1.0)
    assert len(gauss_klds.shape) == 2
    return tf.reduce_sum(gauss_klds, axis=1)

class ZForcing(tf.keras.Model):
    def __init__(
        self,
        inp_dim,
        emb_dim,
        rnn_dim,
        z_dim,
        mlp_dim,
        out_dim,
        out_type="gaussian",
        cond_ln=False,
        nlayers=1,
        z_force=False,
        dropout=0.0,
        use_l2=False,
        drop_grad=False,
    ):
        super(ZForcing, self).__init__()
        assert not drop_grad, "drop_grad is not supported!"

        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.rnn_dim = rnn_dim
        self.nlayers = nlayers
        self.z_dim = z_dim
        self.dropout = dropout
        self.out_type = out_type
        self.mlp_dim = mlp_dim
        self.cond_ln = cond_ln
        self.z_force = z_force
        self.use_l2 = use_l2
        self.drop_grad = drop_grad

        self.initial_hidden_state = tfe.Variable(tf.zeros(shape=[rnn_dim]), trainable=True)
        self.initial_cell_state = tfe.Variable(tf.zeros(shape=[rnn_dim]), trainable=True)
        # self.hidden_state = None
        # self.cell_state = None

        # EMBEDDING LAYERS
        self.embedding_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=emb_dim, input_dim=inp_dim), 
            tf.keras.layers.Dropout(rate=dropout)
        ])

        # LSTM LAYERS
        # fwd - generation, bwd - inference
        self.rnn_fwd_layer = tf.keras.layers.LSTMCell(
            units=rnn_dim,
            use_bias=True,
            kernel_initializer="glorot_uniform",  # different from orig code
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
        )

        self.rnn_bwd_layer = tf.keras.layers.LSTM(
            units=rnn_dim, 
            recurrent_initializer="orthogonal",
            return_state=False,
            return_sequences=True,
        )

        #  PRIOR LAYERS - gives mean and logvar
        self.latent_prior_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=mlp_dim, input_dim=rnn_dim), 
            LReLU(), 
            tf.keras.layers.Dense(units=z_dim * 2)
        ])

        # POSTERIOR LAYER - gives mean and logvar
        self.latent_post_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=mlp_dim, input_dim=rnn_dim * 2), 
            LReLU(), 
            tf.keras.layers.Dense(units=z_dim * 2)
        ])

        # FINAL LAYERS - gives next state
        self.final_fwd_layer = tf.keras.layers.Dense(units=out_dim, input_dim=rnn_dim)
        self.final_bwd_layer = tf.keras.layers.Dense(units=out_dim, input_dim=rnn_dim)

        # EXTRA GENERATION LAYER IN PAPER
        if cond_ln:
            self.generation_layer = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units=mlp_dim, input_dim=z_dim), 
                LReLU(), 
                tf.keras.layers.Dense(units=8 * rnn_dim)
            ])
        else:
            self.generation_layer = tf.keras.layers.Dense(units=mlp_dim, input_dim=z_dim)

        # RECONSTRUCTION LAYER
        self.aux_reconstruction_layer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=mlp_dim, input_dim=z_dim + rnn_dim),
            LReLU(),
            tf.keras.layers.Dense(units=2 * rnn_dim),
        ])

    def reparametrize(self, mu, logvar, eps=None):
        std = tf.exp(tf.multiply(logvar,0.5))
        if eps is None:
            eps = tf.random.normal(std.shape)
        
        eps = tf.add(tf.multiply(eps,std),mu)
        return eps

    def init_hidden_state(self, batch_size):
        """
        Return a properly shaped initial state for the RNN.
        Args:
            batch_size: The size of the first dimension of the inputs
                        (excluding time steps).
        """
        hidden_state = tf.tile(self.initial_hidden_state[None, ...], [batch_size, 1])
        cell_state = tf.tile(self.initial_cell_state[None, ...], [batch_size, 1])
        return hidden_state, cell_state

    def generative_model(self, inputs, targets, hidden_states, cell_states, bwd_lstm_output=None, z_step=None):
        # pdb.set_trace()
        num_steps = inputs.shape[0]         
        inputs = self.embedding_layer(inputs)
        klds, zs, log_pz, log_qz, aux_cs = [], [], [], [], []

        rnn_states = [(hidden_states, cell_states)]
        eps = tf.random.normal([num_steps, inputs.shape[1], self.z_dim], dtype=self.trainable_variables[0].dtype)
        big = tf.zeros(inputs.shape[1], dtype=self.trainable_variables[0].dtype) + 0.5
        big = tfp.distributions.Bernoulli(big).sample()
        big = tf.expand_dims(big, 1)
        assert (z_step is None) or (nsteps == 1)

        for step in range(num_steps):
            print("step ",step)
            hidden_step, cell_step = rnn_states[step] #check if same for every step
            # print("hidden ",hidden_step)
            input_step = inputs[step]
            r_step = eps[step]

            # prior func of hidden state
            z_pri_params = self.latent_prior_layer(hidden_step)
            z_pri_params = tf.clip_by_value(z_pri_params, -8., 8.) #8 in orig code
            z_pri_mu, z_pri_logvar = tf.split(z_pri_params, 2, axis=1)

            # incorp posterior
            if bwd_lstm_output is not None:
                b_step = bwd_lstm_output[step]
                z_post_params = self.latent_post_layer(tf.concat((hidden_step, b_step), axis=1))
                z_post_params = tf.clip_by_value(z_post_params, -8., 8.)
                z_post_mu, z_post_logvar = tf.split(z_post_params, 2, axis=1)                

                # kl div between prior and prior-bwd
                kld = gaussian_kld(z_post_mu, z_post_logvar, z_pri_mu, z_pri_logvar)

                z_step = self.reparametrize(z_post_mu, z_post_logvar, eps=r_step)

                # use only latent variables
                if self.z_force:
                    hidden_step = hidden_step * 0.

                # looks like aux loss, reconstruct bwd step pred using h and z
                aux_params = self.aux_reconstruction_layer(tf.concat((hidden_step, z_step), axis=1))
                aux_params = tf.clip_by_value(aux_params, -8., 8.)
                aux_mu, aux_logvar = tf.split(aux_params, 2, axis=1)

                # disconnect gradient here,paper talks about this
                b_step_ = tf.stop_gradient(b_step)

                # l2 loss or log prob
                if self.use_l2:
                    aux_step = tf.reduce_sum((b_step_ - tf.tanh(aux_mu)) ** 2.0, 1)
                else:
                    aux_step = -log_prob_gaussian(
                            b_step_, tf.tanh(aux_mu), aux_logvar, mean=False)
            
            # generation phase
            else:
                if z_step is None:
                    z_step = self.reparametrize(z_pri_mu, z_pri_logvar, eps=r_step)

                aux_step = tf.reduce_sum(z_pri_mu * 0., -1)
                z_post_mu, z_post_logvar = z_pri_mu, z_pri_logvar
                kld = aux_step
            
            z_gen_step = self.generation_layer(z_step)
            
            # the generation part with LSTM
            # TO DO: ADD LAYER NORM LSTM CELL
            # if self.cond_ln:
            #     z_gen_step = tf.clip_by_value(z_gen_step, -3, 3)
            #     gain_hh, bias_hh = tf.split(z_gen_step, 2, axis=1)
            #     gain_hh = 1. + gain_hh
            #     hidden_new, cell_new = self.rnn_fwd_layer(input_step, (hidden_step, cell_step),
            #                                 gain_hh=gain_hh, bias_hh=bias_hh) #1lstmcell

            # else:
            rnn_fwd_output, rnn_states_new = self.rnn_fwd_layer(tf.concat((z_gen_step, input_step), 1),
                                        (hidden_step, cell_step)) #1lstmcell


            rnn_states.append(rnn_states_new)
            klds.append(kld)
            zs.append(z_step)
            aux_cs.append(aux_step)

            # prior loss
            log_pz.append(log_prob_gaussian(z_step, z_pri_mu, z_pri_logvar))

            #inf loss
            log_qz.append(log_prob_gaussian(z_step, z_post_mu, z_post_logvar))

        klds = tf.stack(klds, 0)
        aux_cs = tf.stack(aux_cs, 0)
        log_pz = tf.stack(log_pz, 0)
        log_qz = tf.stack(log_qz, 0)
        zs = tf.stack(zs, 0)

        # get only hidden states for all steps
        #first one is initialized, so all 0
        fwd_outputs = [s[0] for s in rnn_states[1:]]
        fwd_outputs = tf.stack(fwd_outputs, 0)
        
        
        fwd_final_outputs = self.final_fwd_layer(fwd_outputs) #final network to predict next states
        
        return fwd_final_outputs, rnn_states[1:], klds, aux_cs, zs, log_pz, log_qz            



    def inference_model(self,inputs, targets, hidden_states, cell_states):
        seq_lengths = tf.constant(targets.shape[0], shape=[targets.shape[1]])
        rev_targets = tf.reverse_sequence(targets, seq_lengths, seq_axis=0, batch_axis=1)
        rev_targets_and_input = tf.concat([rev_targets, inputs[:1]], axis=0)

        rev_targets_and_input = self.embedding_layer(rev_targets_and_input)

        #as per paper, share params btn bwd and fwd rnn by passing hidden
        rev_targets_and_input = tf.transpose(rev_targets_and_input,perm=[1,0,2])
        #this is bt
        rev_lstm_output = self.rnn_bwd_layer(rev_targets_and_input, initial_state=[hidden_states,cell_states]) 
        

        rev_lstm_output = tf.transpose(rev_lstm_output,perm=[1,0,2])
        rev_final_output = self.final_bwd_layer(rev_lstm_output[:-1])

        rev_lstm_output = tf.reverse_sequence(rev_lstm_output, seq_lengths, seq_axis=0, batch_axis=1)
        rev_final_output = tf.reverse_sequence(rev_final_output, seq_lengths, seq_axis=0, batch_axis=1)
        return rev_lstm_output, rev_final_output

    def call(self,inputs, targets, hidden, cell, return_stats=False):
        bwd_lstm_output, bwd_final_output = self.inference_model(inputs, targets, hidden, cell)
        fwd_final_output, fwd_lstm_output, klds, aux_nll, zs, log_pz, log_qz = self.generative_model(inputs, targets, hidden, cell, bwd_lstm_output)
        
        kld = tf.reduce_sum(klds, 0)
        log_pz = tf.reduce_sum(log_pz, 0)
        log_qz = tf.reduce_sum(log_qz, 0)
        aux_nll = tf.reduce_sum(aux_nll, 0)

        if self.out_type == 'gaussian':
            out_mu, out_logvar = tf.split(fwd_final_output, 2, -1)
            
            # for forward, true is y
            fwd_nll = -log_prob_gaussian(targets, out_mu, out_logvar)
            fwd_nll = tf.reduce_sum(fwd_nll,0)

            # for bwd, true is x
            out_mu, out_logvar = tf.split(bwd_final_output, 2, -1)
            bwd_nll = -log_prob_gaussian(inputs, out_mu, out_logvar)
            bwd_nll = tf.reduce_sum(bwd_nll,0)
        
        if return_stats:        
            return fwd_nll, bwd_nll, aux_nll, kld, log_pz, log_qz
        
        return tf.reduce_mean(fwd_nll), tf.reduce_mean(bwd_nll), tf.reduce_mean(aux_nll), tf.reduce_mean(kld)

#######################################
###### UNIT TEST PORTION ##############
def create_test_input():
    batch_size = 1
    seq_len = 5
    feature_size = 3
    x = tf.random_normal((batch_size,seq_len,feature_size))
    y = tf.random_normal((batch_size,seq_len,feature_size))
    x = tf.transpose(x, perm=[1,0,2])
    y = tf.transpose(y, perm=[1,0,2])

    return x,y
                
def unit_test_model():
    x,y = create_test_input()
    model = ZForcing(inp_dim=x.shape[2], emb_dim=10, rnn_dim=20, z_dim=15,
                     mlp_dim=10, out_dim=x.shape[2]*2, nlayers=1,
                     cond_ln=False)
    hidden_state, cell_state = model.init_hidden_state(x.shape[1])
    fwd_nll, bwd_nll, aux_nll, kld = model(x,y,hidden_state, cell_state)
    print(fwd_nll, bwd_nll, aux_nll, kld)

if __name__ == "__main__":
    tf.enable_eager_execution()
    unit_test_model()





