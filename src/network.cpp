
#include "network.h"

#include <cstddef>
#include <random>
#include <algorithm>
#include <iostream>

Network::Network(std::vector<int> sizes) {

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
    mx::default_stream(mx::Device::gpu);
}


mx::array Network::sigmoid_prime(mx::array a) {
    mx::array s = mx::sigmoid(a);
    return s * (1 - s);
}


mx::array Network::feed_forward(mx::array a) {

    for (int i=0; i!=weights.size(); i++) {
        mx::array w=weights[i], b=biases[i];
        a = mx::sigmoid(mx::matmul(w, a) + b);
    }

    return a;
}


int Network::max_index(mx::array* a) {

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
    std::vector<mx::array> outs{};
    std::vector<mx::array*> ys{};
    for (auto& [x,y] : *test_data) {
        outs.push_back(feed_forward(x));
        ys.push_back(&y);
    }

    mx::eval(outs);
    for (int i=0; i!=outs.size(); i++) {
        mx::array& out=outs[i], y=*ys[i];
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
        z = mx::add(mx::matmul(w, a), b);
        zs.push_back(z);
        a = mx::sigmoid(z);
        as.push_back(a);
    }

    mx::array delta = mx::subtract(a, y) * sigmoid_prime(zs[zs.size()-1]);
    nabla_b[nabla_b.size()-1] = delta;
    nabla_w[nabla_w.size()-1] = delta * mx::transpose(as[as.size()-2]);

    for (int i=2; i!=num_layers; i++) {
        z = zs[zs.size()-i];
        mx::array sp = sigmoid_prime(z);
        delta = mx::matmul(mx::transpose(weights[weights.size()-i+1]), delta) * sp;
        nabla_b[nabla_b.size()-i] = delta;
        nabla_w[nabla_w.size()-i] = mx::matmul(delta, mx::transpose(as[as.size()-i-1]));
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
            nabla_b[i] = mx::add(nabla_b[i], delta_nabla_b[i]);
            nabla_w[i] = mx::add(nabla_w[i], delta_nabla_w[i]);
        }
    }

    for (int i=0; i!=biases.size(); i++) {
        biases[i] = mx::subtract(biases[i], (eta / batch.size()) * nabla_b[i]);
        weights[i] = mx::subtract(weights[i], (eta / batch.size()) * nabla_w[i]);
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
        }

        std::vector<mx::array> arr(weights.begin(), weights.end());
        arr.insert(arr.end(), biases.begin(), biases.end());
        mx::eval(arr);

        if (test_data != nullptr) {
            std::cout << "Epoch " << i << ": " << evaluate(test_data) << " / " << test_data->size() << std::endl;
        } else {
            std::cout << "Epoch " << i << std::endl;
        }
    }
}
