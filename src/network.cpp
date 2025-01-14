
#include "network.h"

#include <cstddef>
#include <mlx/dtype.h>
#include <mlx/stream.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <mlx/random.h>
#include <mlx/ops.h>

Network::Network(std::vector<int> sizes): s2{new_stream(mx::Device::gpu)}, s3{new_stream(mx::Device::gpu)} {

    this->sizes = sizes;
    this->num_layers = sizes.size();

    std::random_device rd;
    std::mt19937 gen(rd());;
    std::normal_distribution<> dist(0.0, 1.0);

    for (int i=1; i!=num_layers; i++) {
        int s=sizes[i], ps=sizes[i-1];

        mx::array b = mx::random::normal({s,1});
        biases.push_back(b);

        mx::array w = mx::random::normal({s,ps});
        weights.push_back(w);
    }
    mx::default_stream(mx::Device::cpu);
}


mx::array Network::sigmoid_prime(mx::array a) {
    mx::array s = mx::sigmoid(a);
    return s * (1 - s);
}


mx::array Network::feed_forward(mx::array a) {

    for (int i=0; i!=weights.size(); i++) {
        mx::array w=weights[i], b=biases[i];
        a = mx::sigmoid(mx::matmul(w, a, mx::Device::gpu) + b);
    }

    return a;
}


int Network::max_index(mx::array* a) {

    a->eval();
    float* data = a->data<float>();
    size_t size = a->size();

    float max = 0;
    int max_index = 0;

    for (int i=0; i!=size; i++) {
        if (data[i] > max) {
            max = data[i];
            max_index = i;
        }
    }

    return max_index;
}


int Network::evaluate(std::vector<std::tuple<mx::array, mx::array>>* test_data) {

    int correct = 0;
    for (auto& [x,y] : *test_data) {
        mx::array out = feed_forward(x);
        correct += max_index(&out) == max_index(&y) ? 1 : 0;
    }

    return correct;
}


std::tuple<std::vector<mx::array>, std::vector<mx::array>> Network::backprop(mx::array x, mx::array y) {

    std::vector<mx::array> nabla_b, nabla_w;

    for (int i=0; i!=biases.size(); i++) {
        auto& w=weights[i], b=biases[i];
        nabla_b.push_back(mx::zeros({b.shape()[0], b.shape()[1]}, mx::float32));
        nabla_w.push_back(mx::zeros({w.shape()[0], w.shape()[1]}, mx::float32));
    }

    mx::array a=mx::array({}), z=mx::array({});
    a = x;
    std::vector<mx::array> as,zs{};

    as.push_back(a);

    for (int i=0; i!=biases.size(); i++) {
        auto& w=weights[i], b=biases[i];
        z = mx::add(mx::matmul(w, a, s2), b, s2);
        zs.push_back(z);
        a = mx::sigmoid(z, s2);
        as.push_back(a);
    }

    mx::array delta = mx::subtract(a, y, s2) * sigmoid_prime(zs[zs.size()-1]);
    nabla_b[nabla_b.size()-1] = delta;
    nabla_w[nabla_w.size()-1] = delta * mx::transpose(as[as.size()-2]);

    for (int i=2; i!=num_layers; i++) {
        z = zs[zs.size()-i];
        mx::array sp = sigmoid_prime(z);
        delta = mx::matmul(mx::transpose(weights[weights.size()-i+1]), delta, s2) * sp;
        nabla_b[nabla_b.size()-i] = delta;
        nabla_w[nabla_w.size()-i] = mx::matmul(delta, mx::transpose(as[as.size()-i-1]), s2);
    }


    return std::tuple<std::vector<mx::array>, std::vector<mx::array>>{nabla_b, nabla_w};
}


void Network::update_batch(std::vector<std::tuple<mx::array, mx::array>> batch, double eta) {

    std::vector<mx::array> nabla_b, nabla_w;

    for (int i=0; i!=biases.size(); i++) {
        auto& w=weights[i], b=biases[i];
        nabla_b.push_back(mx::zeros({b.shape()[0], b.shape()[1]}, mx::float32));
        nabla_w.push_back(mx::zeros({w.shape()[0], w.shape()[1]}, mx::float32));
    }

    for (const auto& [x,y] : batch) {
        auto [delta_nabla_b, delta_nabla_w] = backprop(x,y);
        for (int i=0; i!=nabla_b.size(); i++) {
            nabla_b[i] = mx::add(nabla_b[i], delta_nabla_b[i], s2);
            nabla_w[i] = mx::add(nabla_w[i], delta_nabla_w[i], s3);
        }
    }

    for (int i=0; i!=biases.size(); i++) {
        biases[i] = mx::subtract(biases[i], (eta / batch.size()) * nabla_b[i], s2);
        weights[i] = mx::subtract(weights[i], (eta / batch.size()) * nabla_w[i], s3);
    }
}


void Network::stochastic_gradient_descent(
    std::vector<std::tuple<mx::array, mx::array>> data,
    int epochs,
    int batch_size,
    double eta,
    std::vector<std::tuple<mx::array, mx::array>>* test_data)
{
    int n = data.size();
    for (int i=0; i!=epochs; i++) {
        std::shuffle(data.begin(), data.end(), std::mt19937(std::random_device{}()));
        for (int j=0; j<n; j+=batch_size) {
            std::vector<std::tuple<mx::array, mx::array>> batch(
                data.begin() + j,
                data.begin() + std::min(j+int(batch_size), n)
            );
            update_batch(batch, eta);
            if (j % 1000 == 0) {
                for (int i=0; i!=weights.size(); i++) {
                    weights[i].eval();
                    biases[i].eval();
                }
            }
        }

        if (test_data != nullptr) {
            std::cout << "Epoch " << i << ": " << evaluate(test_data) << " / " << test_data->size() << std::endl;
        } else {
            std::cout << "Epoch " << i << std::endl;
        }
    }
}
