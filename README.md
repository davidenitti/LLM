# LLM
training LLM models and exploring variations

work in progress...

I re-implemented gpt-2 model and extended the train script from the training in hugging faces

# Pre-training
We first consider the task of pre-training, that is training to predict the next token (sequence of characters, e.g. part or entire word)

## train GPT-2 on a small dataset
Example to train GPT-2 model (124M parameters) on a (relatively) small dataset like wikitext:
```
python train.py --config_name custom:gptv0 --tokenizer_name gpt2 '--config_overrides="dropout":0.06' --per_device_train_batch_size 16 --gradient_accumulation_steps 4 --name_exp dropout0_06_wd1_1_ep25 --weight_decay 1.1 --num_train_epochs 25 --min_learning_rate 0.000001 --lr_scheduler_type cosine_with_min_lr --dataset_name "Salesforce/wikitext" --dataset_config_name "wikitext-103-raw-v1"
```
you should get a perplexity of 16.29
In the example the total batch size is per_device_train_batch_size * gradient_accumulation_steps = 64, if you have issues with GPU memory reduce the per_device_train_batch_size and increase the gradient_accumulation_steps, so the product is the same.

if you don't want to log to wandb and other ways use --no_tracking
## Reproduce GPT-2 on a larger dataset
Example to reproduce GPT-2 using openwebtext dataset
(you need around 230 GB of disk space)

```
python train.py --config_name custom:gptv0 --tokenizer_name gpt2 '--config_overrides="dropout":0.00' --per_device_train_batch_size 10 --gradient_accumulation_steps 10 --name_exp baseline_wd0_1_ep3 --weight_decay 0.1 --num_train_epochs 3 --dataset_name Skylion007/openwebtext --trust_remote_code --no_wait --save_checkpoint --learning_rate 0.0005
```
you should get a perplexity of 18.84

There is room for improvement increasing the batch size, learning rate and the number of epochs e.g.:
```
python train.py --config_name custom:gptv0 --tokenizer_name gpt2 '--config_overrides="dropout":0.00' --per_device_train_batch_size 12 --gradient_accumulation_steps 12 --name_exp baseline2 --weight_decay 0.1 --num_train_epochs 4 --dataset_name Skylion007/openwebtext --trust_remote_code --save_checkpoint
```
you should get a perplexity of ???

If you have big GPUs you can go higher, e.g. the implementation of GPT-2 of Karpathy (nanogpt) uses a total batch size of 480 and around 33 epochs (600k iterations) and reaches a perplexity of 17.28 (val loss of 2.85)

## Use rotary positional encoding
I modified the gpt-2 model to use rotary positional encoding, adapting the implementation of deepseek-v3

```
python train.py --config_name custom:gptv0 --tokenizer_name gpt2 '--config_overrides="dropout":0.00,"rotary_emb":32' --per_device_train_batch_size 10 --gradient_accumulation_steps 10 --name_exp rotary_wd0_1_ep3 --weight_decay 0.1 --num_train_epochs 3 --dataset_name Skylion007/openwebtext --trust_remote_code --save_checkpoint --learning_rate 0.0005
```
you should get a perplexity of ???