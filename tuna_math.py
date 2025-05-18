import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from train import main


def objective(trial):
    dropout = trial.suggest_float("dropout", 0.000001, 0.1, log=True)
    embed_size = trial.suggest_int("embed_size", 120, 780, step=12)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 1.2, log=True)
    steps2think = trial.suggest_int("steps2think", 1, 10)
    total_batch_size = trial.suggest_int("batch_size", 2048, 2048, step=1024)
    learning_rate = trial.suggest_float("learning_rate", 0.0003, 0.0006, log=True)
    mini_batch_size = 256
    grad_acc = total_batch_size // mini_batch_size
    context_len = 25 + steps2think
    rotary = trial.suggest_int("rotary", 0, 0)
    if rotary > 0:
        rotary_emb = embed_size // 12 // 2 // 2 * 2 // rotary
    else:
        rotary_emb = 0
    command = f'--config_name custom:gptv0 --tokenizer_name gpt2 --block_size {context_len} --config_overrides="dropout":{dropout},"vocab_size":21,"context_len":{context_len},"embed_size":{embed_size},"rotary_emb":{rotary_emb} --per_device_train_batch_size {mini_batch_size} --gradient_accumulation_steps {grad_acc} --name_exp tuna_math2 --weight_decay {weight_decay} --num_train_epochs 50 --dataset_name custom:math_dataset --per_device_eval_batch_size 128 --max_math_value 5000 --steps2think {steps2think} -lr {learning_rate}'
    args = command.split(" ")
    print(command)
    print(args)
    return main(args, trial)


if __name__ == "__main__":
    storage = "sqlite:///experiments/math2.sqlite3"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
    study = optuna.create_study(
        study_name="math2", pruner=pruner, direction="maximize", storage=storage, load_if_exists=True
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    if len(study.trials) > 0:
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        try:
            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
        except:
            pass
        
    study.optimize(objective, n_trials=100, timeout=None)
