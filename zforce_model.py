import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import keras


class LReLU(tf.keras.Model):
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

        # EMBEDDING LAYERS
        self.embedding_layer = tf.keras.models.Sequential(
            tf.keras.layers.Dense(units=emb_dim, input_dim=inp_dim), tf.keras.layers.Dropout(rate=dropout)
        )

        # LSTM LAYERS
        # fwd - generation, bwd - inference
        self.rnn_fwd_layer = tf.keras.layers.LSTMCell(
            units=rnn_dim,
            use_bias=True,
            kernel_initializer="glorot_uniform",  # different from orig code
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
        )

        self.rnn_bwd_layer = tf.keras.layers.CuDNNLSTM(units=rnn_dim, recurrent_initializer="orthogonal")

        #  PRIOR LAYERS - gives mean and logvar
        self.latent_prior_layer = tf.keras.models.Sequential(
            tf.keras.layers.Dense(units=mlp_dim, input_dim=rnn_dim), LReLU(), tf.keras.layers.Dense(units=z_dim * 2)
        )

        # POSTERIOR LAYER - gives mean and logvar
        self.latent_post_layer = tf.keras.models.Sequential(
            tf.keras.layers.Dense(units=mlp_dim, input_dim=rnn_dim * 2), LReLU(), tf.keras.layers.Dense(units=z_dim * 2)
        )

        # FINAL LAYERS - gives next state
        self.final_fwd_layer = tf.keras.layers.Dense(units=out_dim, input_dim=rnn_dim)
        self.final_bwd_layer = tf.keras.layers.Dense(units=out_dim, input_dim=rnn_dim)

        # EXTRA GENERATION LAYER IN PAPER
        if cond_ln:
            self.generation_layer = tf.keras.models.Sequential(
                tf.keras.layers.Dense(units=mlp_dim, input_dim=z_dim), LReLU(), tf.keras.layers.Dense(units=8 * rnn_dim)
            )
        else:
            self.generation_layer = tf.keras.layers.Dense(units=mlp_dim, input_dim=z_dim)

        # RECONSTRUCTION LAYER
        self.aux_reconstruction_layer = tf.keras.models.Sequential(
            tf.keras.layers.Dense(units=mlp_dim, input_dim=z_dim + rnn_dim),
            LReLU(),
            tf.keras.layers.Dense(units=2 * rnn_dim),
        )

    def reparametrize(self, mu, logvar, eps=None):
        std = logvar.mul(0.5).exp_()
        if eps is None:
            eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def generative_model():
        pass        

    def inference_model():
        pass

def reparametrize( mu, std, eps=None):
    
    if eps is None:
        eps = std.new(std.size()).normal_()
    return eps.mul(std).add_(mu)
    
def unit_test_model():
    pass


if __name__ == "__main__":
    unit_test_model()
    delta = tf.get_variable('delta', initializer=1.)
    x = tf.random_uniform((), -1., 1.)  # Input
    print(reparametrize(x,delta))