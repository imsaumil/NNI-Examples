from model_VAE import VAE, device, trainer, evaluator
from torch.optim import Adam
from nni.compression.pytorch.pruning import L1NormPruner, FPGMPruner
from nni.compression.pytorch.speedup import ModelSpeedup
import torch
import time


epochs = 15
pruner_used = "FPGMPruner"          # -> (L1NormPruner, FPGMPruner)
config_choice = "config_list_1"     # -> or config_list_2

if __name__ == '__main__':

    print("\nDEVICE BEING USED: ", device, "\n")

    # Defined original unpruned model
    model = VAE().to(device)

    print("ORIGINAL UN-PRUNED MODEL: \n\n", model, "\n\n")

    # Starting time for unpruned model
    start_time = time.time()

    # Running the pre-training stage with original unpruned model
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        trainer(model, optimizer)
        evaluator(model)

    # Ending time for unpruned model
    end_time = time.time()

    # The total execution time of unpruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF UNPRUNED MODEL: ", exec_time, "\n\n")

    # Specifying two sets of configuration
    if config_choice == "config_list_1":
        # Defining the configuration list for pruning
        configuration_list = [{
            'sparsity_per_layer': 0.4,
            'op_types': ['Linear']
        }, {
            'exclude': True,
            'op_names': ['fc1', 'fc4']
        }]
    else:
        # Defining the alternate configuration list for pruning
        configuration_list = [{
            'sparsity_per_layer': 0.7,
            'op_types': ['Linear']
        }, {
            'exclude': True,
            'op_names': ['fc4']
        }]

    # Defining the pruner to be used
    if pruner_used == "L1NormPruner":
        # Wrapping the original model with pruning wrapper
        pruner = L1NormPruner(model, configuration_list)
    else:
        # Wrapping the original model with pruning wrapper
        pruner = FPGMPruner(model, configuration_list)

    print("PRUNER WRAPPED MODEL WITH {}: \n\n".format(pruner_used), model, "\n\n")

    # Next, compressing the model and generating masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()), "\n")

    # Need to unwrap the model before speeding-up.
    pruner._unwrap_model()

    ModelSpeedup(model, torch.rand(64, 1, 28, 28).to(device), masks).speedup_model()

    print("\nPRUNED MODEL WITH {}: \n\n".format(pruner_used), model, "\n\n")

    # Starting time for pruned model
    start_time = time.time()

    # Running the pre-training stage with pruned model
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        trainer(model, optimizer)
        evaluator(model)

    # Ending time for pruned model
    end_time = time.time()

    # The total execution time of pruned model
    exec_time = end_time - start_time
    print("\nTHE TOTAL EXECUTION TIME OF PRUNED MODEL: ", exec_time, "\n\n")