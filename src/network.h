
#ifndef network_h
#define network_h

#include <mlx/mlx.h>
#include <vector>

namespace mx = mlx::core;

class Network {
private:
    std::vector<int> sizes;
    int num_layers;
    std::vector<mx::array> weights;
    std::vector<mx::array> biases;
public:
    Network(std::vector<int> sizes);
    mx::array sigmoid_prime(mx::array a);
    mx::array feed_forward(mx::array a);
    int max_index(mx::array* a);
    int evaluate(std::vector<std::tuple<mx::array, mx::array>>* data);
    std::tuple<std::vector<mx::array>, std::vector<mx::array>> backprop(mx::array x, mx::array y);
    void update_batch(std::vector<std::tuple<mx::array, mx::array>> batch, double eta);
    void stochastic_gradient_descent(
        std::vector<std::tuple<mx::array, mx::array>> data,
        int epochs,
        int batch_size,
        double eta,
        std::vector<std::tuple<mx::array, mx::array>>* test_data
    );
};

#endif
