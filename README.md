# Model pruning using Microsoft NNI
### Authors
Jinam Shah </br>
Saumil Shah

## Folder structure

The python file **model_VAE** contains the main definition of the model i.e. VAE. This is inspired from the pytorch/examples github but is a little different in the actual definition of the model layers.

The **pruning_VAE** contains the methods used for pruning the model and the different configurations used for model pruning. It uses two pruning methods and two different configurations from the NNI pruning library.

The log_files folder contains the log files for the different configurations tried.
```
File naming format: <Pruner used>_<Config used>_<Device used>.txt
```
We have also consolidated the NNI configurations into `config_list.txt`

## Code execution
The pruning example can be run by simply triggering `python3 pruning_VAE.py`

Lines 10 and 11 of the **pruning_VAE** file provide the choice between the different pruners and the different configurations for pruning on the model defined in **model_VAE**.
```
pruner_used = "FPGMPruner" or "L1NormPruner"
config_choice = "config_list_1" or "config_list_2"

config list details:
config_list_1 = [
  {
    'sparsity_per_layer': 0.4,
    'op_types': ['Linear']
  }, 
  {
    'exclude': True,
    'op_names': ['fc1', 'fc4']
  }
]
config_list_2 = [
  {
    'sparsity_per_layer': 0.7,
    'op_types': ['Linear']
  }, 
  {
    'exclude': True,
    'op_names': ['fc4']
  }
]
```

We also provide the option of using CPU or GPU of the machine that you are running the code on. Although for this, a code change in the model definition (**model_VAE.py**) is required. As a simple way of doing this, we just directly change the argument defined for `no-cuda` (line 15).

For using `GPU`:
```
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
```
For using `CPU`
```
parser.add_argument('--no-cuda', action='store_true', default=True,help='disables CUDA training')
```
