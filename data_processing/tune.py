import random

import time

import os

import functools

import argparse
import numpy as np

import ray
import torch
from ray import tune


def run_preprocessing(config):
    from data_processing.combined import do_stage
    model_type = config["model_type"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    preprocessed_output_dir = config["preprocessed_output_dir"]
    refactored_output_dir = config["refactored_output_dir"]
    refactor_options = config["refactor_options"]  # Tuned by Ray

    do_stage_with_params = functools.partial(
        do_stage, model_type=model_type, input_dir=input_dir, output_dir=output_dir,
        refactored_output_dir=refactored_output_dir, preprocessed_output_dir=preprocessed_output_dir,
        refactor_options=refactor_options, vanilla=False)
    do_stage_with_params('refactor_reveal', test=config["test"])
    do_stage_with_params('add_refactored_code', test=config["test"])
    do_stage_with_params('preprocess', test=config["test"])
    do_stage_with_params('Devign-preprocess', test=config["test"])


def run_training(config):
    from Devign.main import main as devign
    preprocessed_output_dir = config["preprocessed_output_dir"]

    if config["model_type"] == 'devign':
        raw_args = ['--ray', '--n_folds', '1', '--model_type', 'devign', '--input_dir', preprocessed_output_dir + '/', '--seed', '0']
        if config["test"]:
            raw_args.append('--test')
        devign(raw_args)  # Train GGNN
    else:
        from Vuld_SySe.representation_learning.api_test import main as reveal

        raw_args = ['--n_folds', '1', '--model_type', 'ggnn', '--input_dir', preprocessed_output_dir + '/',
                    '--seed', '0', '--safe_after']
        if config["test"]:
            raw_args.append('--test')
        devign(raw_args)      # Train GGNN

        raw_args = ['--ray', '--n_folds', '1', '--dataset', preprocessed_output_dir + '/after_ggnn/',
                    '--model_dir', preprocessed_output_dir + '/models/', '--seed', '0', '--features', 'ggnn']
        if config["test"]:
            raw_args.append('--test')
        # run_model(['--dataset', preprocessed_output_dir + '/ggnn_input/processed.bin', '--model_dir', preprocessed_output_dir + '/models/', '--output_dir', preprocessed_output_dir + '/after_ggnn/'])   # Generate GGNN output
        # devign(raw_args)      # Traxin MLP
        reveal(raw_args)


def training_function(config, checkpoint_dir=None):
    print('config:', config)
    print('checkpoint_dir:', checkpoint_dir)

    os.chdir("/work/LAS/weile-lab/benjis/weile-lab/thesis/ReVeal/deployment/ReVeal/")

    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.random.manual_seed(config["seed"])

    if config["phony"]:
        print(f'Returning immediately...')
        for i in range(10):
            score = random.random()
            print(f'Iteration {i} score {score}...')
            tune.report(valid_f1=score)
            time.sleep(1)
        return

    config["preprocessed_output_dir"] = os.path.join(config["output_dir"], "preprocessed_output")
    config["refactored_output_dir"] = os.path.join(config["output_dir"], "refactored_pickle")
    refactor_options = ["style", config["style"]]
    if config["threshold"] is not None:
        refactor_options = ["threshold", f'{config["threshold"]}']
    if config["no-new-names"]:
        refactor_options.append("no-new-names")
    config["refactor_options"] = refactor_options
    print('config["refactor_options"]:', config["refactor_options"])
    run_preprocessing(config)
    run_training(config)

    if config["test"]:
        print(f'Test bogus...')
        for i in range(10):
            score = random.random()
            print(f'Test iteration {i} score {score}...')
            tune.report(valid_f1=score)
            time.sleep(1)


def run_tuning(args):
    if args.quit_after_init:
        return

    analysis = tune.run(
        training_function,
        config={
            "threshold": tune.sample_from(lambda spec: np.random.choice(np.linspace(0.0, 1.0, 100)) if spec.config.style == 'threshold' else None),
            "no-new-names": tune.choice([True, False]),
            "style": tune.choice(["one_of_each", "k_random", "threshold"]),
            "model_type": "devign",
            "input_dir": "data/devign",
            "output_dir": args.output_dir,
            "seed": args.seed,
            "phony": args.phony,
            "test": args.test,
        },
        resources_per_trial={"cpu": args.cpus_per_task, "gpu": args.gpus_per_task},
        local_dir='./ray_results/',
        metric="valid_f1",
        mode="max",
        name='devign',
        num_samples=10,
        # stop=ra
    )

    print("Best config: ", analysis.get_best_config(metric="valid_f1", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    print(df)
    df.to_pickle('analysis.results_df.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", type=str, default=None)
    # parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output_tune")
    parser.add_argument("--quit_after_init", action='store_true')
    parser.add_argument("--phony", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--num_cpus", type=int, default=9)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--cpus_per_task", type=int, default=4)
    parser.add_argument("--gpus_per_task", type=int, default=1)
    args = parser.parse_args()

    print("args:", args)

    # Connect to head node
    ray.init(address=args.address, num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    print(ray.cluster_resources())

    np.random.seed(args.seed)

    run_tuning(args)


if __name__ == '__main__':
    main()
