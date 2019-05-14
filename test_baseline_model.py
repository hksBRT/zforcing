import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import keras
import os
import math
import json
import pickle
import random
import argparse
import pandas as pd
import pdb
import time

from keras.models import Sequential
import matplotlib.pyplot as plt
import wandb
from wandb.tensorflow import WandbHook

import envs.constants as constants
from envs.keys import STATE_KEYS, ACTION_KEYS
from models.baselines import FeedForwardModel, LstmModel
from utils.data_loader import load_data, make_datasets
from utils.fwd_model_rollout import fwd_model_rollout
from utils.normalizer import Normalizer, MinMaxNormalizer
import utils.plot as plot

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class DataNormalizer(tfe.Checkpointable):
    def __init__(self, states_dim):
        self.states_normalizer = Normalizer(states_dim)
        self.deltas_normalizer = Normalizer(states_dim)
        self.actions_normalizer = MinMaxNormalizer(
            min_value=constants.action_low, max_value=constants.action_high
        )


def load_and_stack_data(load_file_names=None):
    with open(load_file_names) as f:
        file_name_list = f.readlines()
    file_name_list = [x.strip() for x in file_name_list]

    states_list = []
    deltas_list = []
    actions_list = []
    weights_list = []

    for file_name in file_name_list:
        file_dir = os.path.join(os.getcwd(), "testing_pipeline", file_name)
        states, deltas, actions, weights = load_data(file_dir)
        states_list.append(states)
        deltas_list.append(deltas)
        actions_list.append(actions)
        weights_list.append(weights)

    states_full = np.concatenate(states_list)
    deltas_full = np.concatenate(deltas_list)
    actions_full = np.concatenate(actions_list)
    weights_full = np.concatenate(weights_list)

    weights_full = np.expand_dims(weights_full, axis=-1)
    return states_full, deltas_full, actions_full, weights_full


def model_training(model, train_dataset, data_normalizer, optimizer, clip_norm_val):
    mean_train_loss_epoch = tfe.metrics.Mean()
    for (batch, (states, deltas, actions, weights)) in enumerate(train_dataset):
        states_normalized = data_normalizer.states_normalizer(
            states, weights=weights, training=True
        )
        deltas_normalized = data_normalizer.deltas_normalizer(
            deltas, weights=weights, training=True
        )
        actions_normalized = data_normalizer.actions_normalizer(
            actions, weights=weights
        )

        input = tf.concat([states_normalized, actions_normalized], axis=-1)
        with tf.GradientTape() as tape:
            deltas_normalized_pred = model(input, training=True, reset_state=True)
            loss = tf.losses.mean_squared_error(
                labels=deltas_normalized,
                predictions=deltas_normalized_pred,
                weights=weights,
            )

        # print("batch {} loss {}".format(batch+1,loss.numpy()))
        mean_train_loss_epoch(loss.numpy())
        grads = tape.gradient(loss, model.trainable_variables)
        grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=clip_norm_val)
        optimizer.apply_gradients(
            zip(grads_clipped, model.trainable_variables),
            global_step=tf.train.get_or_create_global_step(),
        )
    return mean_train_loss_epoch.result().numpy()


def model_evaluation(model, eval_dataset, data_normalizer):
    mean_eval_loss_epoch = tfe.metrics.Mean()
    for (batch, (states, deltas, actions, weights)) in enumerate(eval_dataset):
        states_normalized = data_normalizer.states_normalizer(
            states, weights=weights, training=False
        )
        deltas_normalized = data_normalizer.deltas_normalizer(
            deltas, weights=weights, training=False
        )
        actions_normalized = data_normalizer.actions_normalizer(
            actions, weights=weights
        )

        input = tf.concat([states_normalized, actions_normalized], axis=-1)
        deltas_normalized_pred = model(input, training=False, reset_state=True)
        loss = tf.losses.mean_squared_error(
            labels=deltas_normalized,
            predictions=deltas_normalized_pred,
            weights=weights,
        )

        mean_eval_loss_epoch(loss)

    return mean_eval_loss_epoch.result().numpy()


def model_testing(
    model,
    test_dataset,
    data_normalizer,
    episodes,
    trajectories_to_plot,
    save_dir,
    use_wandb=False,
):
    # plot rollouts vs true states
    test_error = tfe.metrics.Mean()
    if use_wandb:
        wandb_ob=wandb
    else:
        wandb_ob=None
    
    for batch_states, batch_deltas, batch_actions, batch_weights in test_dataset:
        initial_states = batch_states[:, 0, :]

        if episodes == -1:
            episodes = batch_states.shape[0]
        if trajectories_to_plot == -1:
            trajectories_to_plot = episodes

        rollouts = fwd_model_rollout(
            model,
            initial_states,
            batch_actions,
            data_normalizer,
            steps=batch_states.shape[1],
            episodes=episodes,
            clip_min_value=constants.state_low,
            clip_max_value=constants.state_high,
        )

        calc_feature_loss = True

        if calc_feature_loss is True:
            feature_wise_loss = np.zeros(5)
            for i in range(5):
                feature_wise_loss[i] = tf.losses.mean_squared_error(
                    labels=batch_states[
                        :episodes, :, i : i + 1
                    ],  # choose how many states to consider
                    predictions=rollouts[:episodes, :, i : i + 1],
                    weights=batch_weights[:episodes, :],
                )
                print(
                    "loss {} \t {:0.3f}".format(
                        STATE_KEYS[i], math.sqrt(feature_wise_loss[i])
                    )
                )

        # computes mse on height states only
        loss = tf.losses.mean_squared_error(
            labels=batch_states[:episodes, :, :5],  # choose how many states to consider
            predictions=rollouts[:episodes, :, :5],
            weights=batch_weights[:episodes, :],
        )

        rollouts = (
            rollouts * batch_weights[:episodes]
        )  # bcs we are not masking rollouts, during loss calc we pass weights to mask but not during plotting
        batch_states = (
            batch_states * batch_weights[:episodes]
        )  # bcs batch state seems to go to 151 steps, no idea why

        test_error(loss / 10000.0)
        plot.plot_trajectories(
            rollouts[:episodes, :, :5],
            batch_states[:episodes, :, :5],
            trajectories_to_plot=trajectories_to_plot,
            save_dir=save_dir,
            wandb_ob=wandb_ob,
        )

        # pdb.set_trace()
    return test_error.result().numpy()


def main():
    # Parse arguments given by user
    parser = argparse.ArgumentParser()
    wandb_parser = parser.add_mutually_exclusive_group(required=False)
    wandb_parser.add_argument("--use-wandb", dest="use_wandb", action="store_true")
    wandb_parser.add_argument(
        "--dont-use-wandb", dest="use_wandb", action="store_false"
    )
    parser.set_defaults(use_wandb=False)
    parser.add_argument("--load-file-names", type=str, default="load_file_names.txt")
    # parser.add_argument("--save-dir", type= str)
    parser.add_argument("--forward-model", type=str, default="lstm")
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--total-epochs", type=int, default=10)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delete-ground", type=int, default=1)
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args()

    if args.use_wandb is True:
        wandb.init(entity="blueriver", project="vector-boom-control")
        wandb.config.update(args)  # adds all of the arguments as config variables
        wandb.save(args.load_file_names)

    tf.enable_eager_execution()

    # set random seeds for reproducibility and
    # easier comparisons between experiments
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # load data from the file/s
    states_full, deltas_full, actions_full, weights_full = load_and_stack_data(
        args.load_file_names
    )

    if args.delete_ground == 1:
        print("Deleting ground sensor readings")
        states_full = np.delete(states_full, slice(5, 10), axis=2)
        deltas_full = np.delete(deltas_full, slice(5, 10), axis=2)

    # trajectories_to_plot = deltas_full.shape[0]
    # plot.plot_trajectories(
    #     deltas_full[:, :, :5],
    #     trajectories_to_plot=trajectories_to_plot,
    #     save_dir=os.getcwd() + "/plots_trajectories",
    #     wandb_ob=None,
    # )

    states_dim = states_full.shape[2]
    actions_dim = actions_full.shape[2]
    total_episodes = states_full.shape[0]

    print("Shape of states:", states_full.shape)
    print("Shape of actions:", actions_full.shape)

    # use tf dataset api to shuffle and batch
    train_dataset, eval_dataset, test_dataset = make_datasets(
        states_full,
        deltas_full,
        actions_full,
        weights_full,
        batch_size=args.train_batch_size,
        eval_proportion=0.1,
        test_proportion=0.05,
    )

    data_normalizer = DataNormalizer(states_dim)

    # define model, optimizer and batch sizes
    if args.forward_model == "ff":
        print("Using Feed Forward model")
        model = FeedForwardModel(states_dim, actions_dim, args.hidden_units)
    elif args.forward_model == "lstm":
        print("Using LSTM model")
        model = LstmModel(states_dim, args.hidden_units)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)

    train_loss_total = []  # store loss values for all epoch
    eval_loss_total = []
    best_eval_loss = np.inf
    best_eval_loss_epoch = None
    best_model = model

    cur_sys_time = time.strftime("%Y%m%d-%H%M")
    checkpoint_dir = os.getcwd()+"/checkpoints/"+cur_sys_time
    results_dir = os.getcwd()+"/results/"+cur_sys_time
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    checkpoint = tf.train.Checkpoint(
        data_normalizer=data_normalizer,
        optimizer=optimizer,
        model=model,
        optimizer_step=tf.train.get_or_create_global_step(),
    )

    with tf.device(args.device):
        # # start the training process
        for epoch in range(args.total_epochs):
            start = time.time()
            train_loss_epoch = model_training(
                model, train_dataset, data_normalizer, optimizer, args.grad_clip_norm
            )
            end = time.time()
            eval_loss_epoch = model_evaluation(model, eval_dataset, data_normalizer)

            if eval_loss_epoch < best_eval_loss:
                best_eval_loss = eval_loss_epoch
                best_eval_loss_epoch = epoch + 1
                best_model = model

            checkpoint.save(checkpoint_prefix)

            train_loss_total.append(train_loss_epoch)
            eval_loss_total.append(eval_loss_epoch)
            print(
                "epoch {} \t epoch-mean-train-loss {:0.2f} \t epoch-eval-loss {:0.2f} \t time elapsed {:0.2f}".format(
                    epoch + 1, train_loss_epoch, eval_loss_epoch, end - start
                )
            )
            start = end

        # load model with lowest eval loss
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        print("------------------------------------------------------")
        print("------------------------------------------------------")
        test_error = model_testing(
            best_model,
            test_dataset,
            data_normalizer,
            episodes=-1,
            trajectories_to_plot=-1,
            save_dir=results_dir,
            use_wandb=args.use_wandb,
        )

        print("------------------------------------------------------")
        print("Test Error on learnt model {:0.2f}".format(test_error))

        plt.figure(1)
        plt.subplot(211)
        plt.plot(train_loss_total)
        plt.xlabel("Batch #")
        plt.ylabel("Training Loss [MSE]")
        plt.grid(axis="both")

        plt.subplot(212)
        plt.plot(eval_loss_total, "r")
        plt.xlabel("Batch #")
        plt.ylabel("Eval Loss [MSE]")
        plt.grid(axis="both")

        if args.use_wandb:
            wandb.log(
                {
                    "test_error": test_error,
                    "best_eval_loss": best_eval_loss,
                    "epoch": best_eval_loss_epoch,
                    "Loss Plots": [wandb.Image(plt, caption="")],
                }
            )

        plt.savefig(os.path.join(results_dir, "loss_plots"))


if __name__ == "__main__":
    main()