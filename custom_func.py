from model import CustomGPTConfig, CustomGPTmodel
from rnn import CustomRNNConfig, CustomRNNmodel

def custom_config(name_config):
    if name_config == "custom:gptv0":
        return CustomGPTConfig()
    elif name_config == "custom:rnnv0":
        return CustomRNNConfig()
    else:
        raise NotImplementedError()


def custom_model(config):
    if isinstance(config, CustomGPTConfig):
        return CustomGPTmodel(config)
    elif isinstance(config, CustomRNNConfig):
        return CustomRNNmodel(config)
    else:
        raise NotImplementedError()
