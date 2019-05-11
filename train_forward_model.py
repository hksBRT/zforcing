import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import keras
import os
import pdb
import math
import argparse
import random
from zforce_model import ZForcing
import time

def arg_parser():
    # Parse arguments given by user
    parser = argparse.ArgumentParser()
    # wandb_parser = parser.add_mutually_exclusive_group(required=False)
    # wandb_parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    # wandb_parser.add_argument(
    #     "--dont-use-wandb", dest="use_wandb", action="store_false"
    # )
    # parser.set_defaults(use_wandb=False)
    # parser.add_argument("--load-file-names", type=str, default="load_file_names.txt")
    # parser.add_argument("--save-dir", type= str)
    # parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--total-epochs", type=int, default=10)
    parser.add_argument("--rnn_dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--bwd", type=float, default=0)
    parser.add_argument("--aux_sta", type=float, default=0.0)
    parser.add_argument("--aux_end", type=float, default=0.0)
    parser.add_argument("--cond_ln", action='store_true')
    parser.add_argument("--z_force", action='store_true')
    # parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()
    return args

def train():
    args = arg_parser()
    tf.enable_eager_execution()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    log_interval = 1
    inp_dim = 200
    batch_size = args.batch_size
    num_train_batches = 1

    model = ZForcing(inp_dim=200, emb_dim=args.emb_dim, rnn_dim=args.rnn_dim, z_dim=args.z_dim,
                     mlp_dim=args.mlp_dim, out_dim=400, nlayers=1,
                     cond_ln=args.cond_ln, z_force=args.z_force)

    hidden_state, cell_state = model.init_hidden_state(batch_size)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)

    kld_step = 0.00005
    aux_step = abs(args.aux_end - args.aux_sta) / (2 * num_train_batches)  # Annealing over two epochs.
    print("aux_step: {}".format(aux_step))
    kld_weight = args.kla_sta
    aux_weight = args.aux_sta
    t = time.time()    

    for epoch in range(args.total_epochs):
        print('Epoch {}'.format(epoch))
        step = 0
        b_fwd_loss, b_bwd_loss, b_kld_loss, b_aux_loss, b_all_loss = \
            (0., 0., 0., 0., 0.)

        # for x,y in train_data:
        step+=1
        with tf.GradientTape() as tape:
            fwd_nll, bwd_nll, aux_nll, kld = model(x, y, hidden_state, cell_state)
            bwd_nll = (aux_weight > 0.) * (bwd * bwd_nll)
            aux_nll = aux_weight * aux_nll
            all_loss = fwd_nll + bwd_nll + aux_nll + kld_weight * kld
            # anneal kld cost
            kld_weight += kld_step
            kld_weight = min(kld_weight, 1.)
            # anneal auxiliary cost
            if args.aux_sta <= args.aux_end:
                aux_weight += aux_step
                aux_weight = min(aux_weight, args.aux_end)
            else:
                aux_weight -= aux_step
                aux_weight = max(aux_weight, args.aux_end)

            if kld.data[0] >= 10000:
                continue
            if np.isnan(all_loss.data[0]) or np.isinf(all_loss.data[0]):
                print("NaN", end="\r")  # Useful to see if training is stuck.
                continue        

        grads = tape.gradient(all_loss, model.trainable_variables)
        # grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=100)
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables),
            global_step=tf.train.get_or_create_global_step(),
        )

        b_all_loss += all_loss.data[0]
        b_fwd_loss += fwd_nll.data[0]
        b_bwd_loss += bwd_nll.data[0]
        b_kld_loss += kld.data[0]
        b_aux_loss += aux_nll.data[0]

        if step % log_interval == 0:
            s = time.time()
            log_line = 'epoch: [%d/%d], step: [%d/%d], loss: %f, fwd loss: %f, aux loss: %f, bwd loss: %f, kld: %f, kld weight: %f, aux weight: %.4f, %.2fit/s' % (
                epoch, num_epochs, step, nbatches,
                b_all_loss / log_interval,
                b_fwd_loss / log_interval,
                b_aux_loss / log_interval,
                b_bwd_loss / log_interval,
                b_kld_loss / log_interval,
                kld_weight,
                aux_weight,
                log_interval / (s - t))
            b_all_loss = 0.
            b_fwd_loss = 0.
            b_bwd_loss = 0.
            b_aux_loss = 0.
            b_kld_loss = 0.
            print(log_line)        

if __name__ == "__main__":
    train()