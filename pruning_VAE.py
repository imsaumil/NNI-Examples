import torch
import argparse
# from model_VAE import VAE, device, trainer, evaluator
from model_VAE import VAE, trainer, evaluator
from torch.optim import Adam
from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner
# from nni.compression.pytorch.pruning import L1NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup


# epochs = 15

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NNI VAE Example')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    epochs = args.epochs
    pruning_method = args.pruning_method

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = VAE().to(device)
    print("ORIGINAL UN-PRUNED MODEL: \n\n", model)

    optimizer = Adam(model.parameters(), lr=1e-3)

    # Running the pre-training stage with original unpruned model
    for epoch in range(epochs):
        trainer(model, optimizer)
        evaluator(model)

    # Defining the configuration list for pruning
    configuration_list = [{
        'sparsity_per_layer': 0.4,
        'op_types': ['Linear']
    }, {
        'exclude': True,
        'op_names': ['fc1', 'fc4']
    }]

    # Wrapping the original model with pruning wrapper
    pruner = L1NormPruner(model, configuration_list)
    print("\nPRUNER WRAPPED MODEL: \n\n", model, "\n")

    # Next, compressing the model and generating masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

    # Need to unwrap the model before speeding-up.
    pruner._unwrap_model()

    ModelSpeedup(model, torch.rand(64, 1, 28, 28).to(device), masks).speedup_model()

    print("\nPRUNED MODEL: \n\n", model)

    # Running the pre-training stage with original unpruned model
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        trainer(model, optimizer)
        evaluator(model)