# kuzushiji with MLX

This is supposed to be a playground to figure out how to use MLX in C++, while learning a bit about C++ along the way.
The implementation follows the initial MNIST NN implementation of the [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) book.

The data in this repository is copied from [kmnist](https://github.com/rois-codh/kmnist).

If anybody would like to help me optimize this code, I'd be more than grateful. Until then I'll try myself.

## Installing

```bash
cmake .
make
```

## Running

```bash
bin/kuzushiji
```

## Optimization

`measure.sh` is executed which runs the binary 10 times and records the passed seconds for each execution.

| Release | Average Time (s) | Time / Epoch (s) |
|---------|------------------|------------------|
| `0.0.1` |     `1070.3`     |      `35.67`     |
| `0.0.2` |     `767.3`      |      `25.57`     |


## Changes

### `0.0.2` [suggestions by @angeloskath](https://github.com/ml-explore/mlx/issues/1767#issuecomment-2594566745)
 - default stream gpu
 - drop adding arbitrary streams to operations 
 - evaluate list of weights and biases
 - one eval call after epoch
 - one eval call after calculating all outputs
 
#### Notes

Although there was a significant speedup (~40%), it was not as much as I had hoped for. Letting basically the same code from my initial implementation run using `Eigen::MatrixXd` as a drop-in replacement for `mx::array`, the code finished in ~60 seconds: 2 seconds per epoch (running the code once). I will publish the Eigen implementation in a branch: `eigen`.
I had hoped to use `mlx` as a BLAS library with GPU support for Macs with silicon chips: something that seems to be hard to come by in C++ (or maybe I didn't look hard enough). I do think though that this is not what `mlx` is trying to be. *This may be obvious in retrospect*. I'm quite sure using the framework the way it's intended to be used would yield a significant speedup over my naive implementation of a neural network with Eigen.

I'd like to thank @angeloskath again for taking the time to look over my code and making suggestions.

### `0.0.1` inital implementation
 - default stream `mx::Device::cpu`
 - using two `mx::Device::gpu` streams arbitrarily
 - `eval` called after 100 batches and in `evaluate` on every `mx::array`
