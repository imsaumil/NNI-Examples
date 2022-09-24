# Create structure file and check code.

## RESULT SUMMARY ##

```
ORIGINAL UN-PRUNED MODEL: 
 VAE(
  (fc1): Linear(in_features=784, out_features=600, bias=True)
  (fc21): Linear(in_features=600, out_features=100, bias=True)
  (fc22): Linear(in_features=600, out_features=100, bias=True)
  (fc3): Linear(in_features=100, out_features=600, bias=True)
  (fc4): Linear(in_features=600, out_features=784, bias=True)
)
====> Test set loss: 135.1325
====> Test set loss: 118.9197
====> Test set loss: 112.7824
====> Test set loss: 109.8977
====> Test set loss: 108.5349
====> Test set loss: 107.4287
====> Test set loss: 106.5929
====> Test set loss: 106.0382
====> Test set loss: 105.5211
====> Test set loss: 105.1355
====> Test set loss: 104.8926
====> Test set loss: 104.6621
====> Test set loss: 104.3693
====> Test set loss: 104.0606
====> Test set loss: 103.9012
PRUNER WRAPPED MODEL: 
 VAE(
  (fc1): Linear(in_features=784, out_features=600, bias=True)
  (fc21): PrunerModuleWrapper(
    (module): Linear(in_features=600, out_features=100, bias=True)
  )
  (fc22): PrunerModuleWrapper(
    (module): Linear(in_features=600, out_features=100, bias=True)
  )
  (fc3): PrunerModuleWrapper(
    (module): Linear(in_features=100, out_features=600, bias=True)
  )
  (fc4): Linear(in_features=600, out_features=784, bias=True)
) 
fc21  sparsity :  0.6
fc22  sparsity :  0.6
fc3  sparsity :  0.6
C:\Users\sashah8\Anaconda3\envs\nni_env\lib\site-packages\torch\jit\_trace.py:992: TracerWarning: Trace had nondeterministic nodes. Did you forget call .eval() on your model? Nodes:
	%eps : Float(8, 100, strides=[100, 1], requires_grad=0, device=cuda:0) = aten::randn_like(%std, %40, %41, %42, %43, %44) # C:\Users\sashah8\CS - 791\NNI\vae\model_VAE.py:61:0
This may cause errors in trace checking. To disable trace checking, pass check_trace=False to torch.jit.trace()
  _check_trace(
C:\Users\sashah8\Anaconda3\envs\nni_env\lib\site-packages\torch\jit\_trace.py:992: TracerWarning: Output nr 1. of the traced function does not match the corresponding output of the Python function. Detailed error:
Tensor-likes are not close!
Mismatched elements: 5036 / 6272 (80.3%)
Greatest absolute difference: 0.4302244782447815 at index (7, 549) (up to 1e-05 allowed)
Greatest relative difference: 3.910687439142421 at index (3, 689) (up to 1e-05 allowed)
  _check_trace(
[2022-09-24 01:46:29] start to speedup the model
C:\Users\sashah8\Anaconda3\envs\nni_env\lib\site-packages\torch\jit\_trace.py:992: TracerWarning: Output nr 1. of the traced function does not match the corresponding output of the Python function. Detailed error:
Tensor-likes are not close!
Mismatched elements: 4995 / 6272 (79.6%)
Greatest absolute difference: 0.2843446433544159 at index (5, 494) (up to 1e-05 allowed)
Greatest relative difference: 2.425027228903965 at index (1, 123) (up to 1e-05 allowed)
  _check_trace(
no multi-dimension masks found.
[2022-09-24 01:46:29] infer module masks...
[2022-09-24 01:46:29] Update mask for .aten::view.5
[2022-09-24 01:46:29] Update mask for fc1
[2022-09-24 01:46:29] Update mask for .aten::relu.6
[2022-09-24 01:46:29] Update mask for fc21
[2022-09-24 01:46:29] Update mask for fc22
[2022-09-24 01:46:29] Update mask for .aten::mul.7
[2022-09-24 01:46:29] Update mask for .aten::exp.8
[2022-09-24 01:46:29] Update mask for .aten::randn_like.9
[2022-09-24 01:46:29] Update mask for .aten::mul.10
[2022-09-24 01:46:29] Update mask for .aten::add.11
[2022-09-24 01:46:29] Update mask for fc3
[2022-09-24 01:46:29] Update mask for .aten::relu.12
[2022-09-24 01:46:29] Update mask for fc4
[2022-09-24 01:46:29] Update mask for .aten::sigmoid.13
[2022-09-24 01:46:29] Update the indirect sparsity for the .aten::sigmoid.13
[2022-09-24 01:46:29] Update the indirect sparsity for the fc4
[2022-09-24 01:46:29] Update the indirect sparsity for the .aten::relu.12
[2022-09-24 01:46:29] Update the indirect sparsity for the fc3
[2022-09-24 01:46:29] Update the indirect sparsity for the .aten::add.11
[2022-09-24 01:46:30] Update the indirect sparsity for the fc21
[2022-09-24 01:46:30] Update the indirect sparsity for the .aten::mul.10
[2022-09-24 01:46:30] Update the indirect sparsity for the .aten::randn_like.9
[2022-09-24 01:46:30] Update the indirect sparsity for the .aten::exp.8
[2022-09-24 01:46:30] Update the indirect sparsity for the .aten::mul.7
[2022-09-24 01:46:30] Update the indirect sparsity for the fc22
[2022-09-24 01:46:30] Update the indirect sparsity for the .aten::relu.6
[2022-09-24 01:46:30] Update the indirect sparsity for the fc1
[2022-09-24 01:46:30] Update the indirect sparsity for the .aten::view.5
[2022-09-24 01:46:30] resolve the mask conflict
[2022-09-24 01:46:30] replace compressed modules...
[2022-09-24 01:46:30] replace module (name: fc1, op_type: Linear)
[2022-09-24 01:46:30] replace linear with new in_features: 784, out_features: 600
C:\Users\sashah8\Anaconda3\envs\nni_env\lib\site-packages\torch\_tensor.py:1083: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at  C:\cb\pytorch_1000000000000\work\build\aten\src\ATen/core/TensorBody.h:482.)
  return self._grad
[2022-09-24 01:46:30] replace module (name: fc21, op_type: Linear)
[2022-09-24 01:46:30] replace linear with new in_features: 600, out_features: 70
[2022-09-24 01:46:30] replace module (name: fc22, op_type: Linear)
[2022-09-24 01:46:30] replace linear with new in_features: 600, out_features: 70
[2022-09-24 01:46:30] replace module (name: fc3, op_type: Linear)
[2022-09-24 01:46:30] replace linear with new in_features: 70, out_features: 360
[2022-09-24 01:46:30] replace module (name: fc4, op_type: Linear)
[2022-09-24 01:46:30] replace linear with new in_features: 360, out_features: 784
[2022-09-24 01:46:30] speedup done
PRUNED MODEL: 
 VAE(
  (fc1): Linear(in_features=784, out_features=600, bias=True)
  (fc21): Linear(in_features=600, out_features=70, bias=True)
  (fc22): Linear(in_features=600, out_features=70, bias=True)
  (fc3): Linear(in_features=70, out_features=360, bias=True)
  (fc4): Linear(in_features=360, out_features=784, bias=True)
)
====> Test set loss: 125.0589
====> Test set loss: 125.0402
====> Test set loss: 125.0407
====> Test set loss: 124.9597
====> Test set loss: 125.0433
====> Test set loss: 124.9425
====> Test set loss: 124.9915
====> Test set loss: 124.9710
====> Test set loss: 125.0235
====> Test set loss: 125.0355
====> Test set loss: 125.0044
====> Test set loss: 124.8415
====> Test set loss: 125.0979
====> Test set loss: 125.0193
====> Test set loss: 125.0515
```
