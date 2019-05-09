import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import keras
import os
import pdb

tf.logging.set_verbosity(3)

class LReLU(tf.keras.layers.Layer):
    def __init__(self, c=1.0 / 3):
        super(LReLU, self).__init__()
        self.c = c

    def forward(self, x):
        x = tf.keras.layers.LeakyReLU(x, alpha=self.c)
        return tf.clip_by_value(x, -3.0, 3.0)


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
            return_state=True,
            return_sequences=False
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

    def generative_model():
        pass        

    def inference_model(self,inputs, targets, hidden_states, cell_states):
        seq_lengths = tf.constant(targets.shape[0], shape=[targets.shape[1]])
        rev_targets = tf.reverse_sequence(targets, seq_lengths, seq_axis=0, batch_axis=1)
        rev_targets_and_input = tf.concat([rev_targets, inputs[:1]], axis=0)

        rev_targets_and_input = self.embedding_layer(rev_targets_and_input)

        #as per paper, share params btn bwd and fwd rnn by passing hidden
        rev_targets_and_input = tf.transpose(rev_targets_and_input,perm=[1,0,2])
        pdb.set_trace()
        rev_lstm_output, _ = self.rnn_bwd_layer(rev_targets_and_input, initial_state=[hidden_states,cell_states]) 
        

        rev_hidden_states = tf.transpose(rev_hidden_states,perm=[1,0,2])

        rev_outputs = self.final_bwd_layer(rev_hidden_states[:-1])

        hidden_states = tf.reverse_sequence(rev_hidden_states, seq_lengths, seq_axis=0, batch_axis=1)
        outputs = tf.reverse_sequence(rev_outputs, seq_lengths, seq_axis=0, batch_axis=1)
        return hidden_states, outputs

    def call(self,inputs, targets, hidden, cell):
        hidden,output = self.inference_model(inputs, targets, hidden, cell)
        return output

def create_test_input():
    batch_size = 4
    seq_len = 10
    feature_size = 5
    x = tf.random_normal((batch_size,seq_len,feature_size))
    y = tf.random_normal((batch_size,seq_len,feature_size))
    x = tf.transpose(x, perm=[1,0,2])
    y = tf.transpose(y, perm=[1,0,2])

    return x,y
                
def unit_test_model():
    model = ZForcing(inp_dim=5, emb_dim=10, rnn_dim=15, z_dim=20,
                     mlp_dim=25, out_dim=30, nlayers=1,
                     cond_ln=False)
    x,y = create_test_input()
    hidden, cell_state = model.init_hidden_state(x.shape[1])
    pred = model(x,y,hidden, cell_state)


    


if __name__ == "__main__":
    tf.enable_eager_execution()
    unit_test_model()







