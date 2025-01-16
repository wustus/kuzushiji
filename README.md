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


## Changes

### `0.0.1` inital implementation
 - default stream `mx::Device::cpu`
 - using two `mx::Device::gpu` streams arbitrarily
 - `eval` called after 100 batches and in `evaluate` on every `mx::array`
