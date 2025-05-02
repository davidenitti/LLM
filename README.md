# LLM
train LLM models and exploring variations

work in progress...

# Reproduce GPT-2
Example to reproduce GPT-2 using openwebtext dataset
(you need around 230 GB of disk space)
```
python train.py --config_name custom:gptv0 --tokenizer_name gpt2 '--config_overrides="dropout":0.00' --per_device_train_batch_size 10 --gradient_accumulation_steps 10 --name_exp baseline --weight_decay 0.1 --num_train_epochs 4 --dataset_name Skylion007/openwebtext --trust_remote_code --save_checkpoint
```
you can change the batch size which is per_device_train_batch_size*gradient_accumulation_steps
nanogpt of Karpathy uses 480 of total batch size