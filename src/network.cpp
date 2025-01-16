
#include "network.h"
#include "Eigen/src/Core/Matrix.h"
#include <cmath>
#include <iostream>
#include <random>

Network::Network(std::vector<int> sizes) {

    this->num_layers = sizes.size();
    this->sizes = sizes;
    
    this->biases = std::vector<Eigen::VectorXd>();
    this->weights = std::vector<Eigen::MatrixXd>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<>dist(0.0, 1.0);

    for (int i=1; i!=sizes.size(); i++) {
        Eigen::VectorXd v = Eigen::VectorXd::NullaryExpr(sizes[i], [&]() { return dist(gen); });
        this->biases.push_back(v);

        int s = sizes.at(i-1), ns = sizes.at(i);

        Eigen::MatrixXd m = Eigen::MatrixXd::NullaryExpr(ns, s, [&]() { return dist(gen); });
        this->weights.push_back(m);
    }
}


double Network::sigmoid(double v) {
    if (v > 50.0) return 1.0;
    if (v < -50.0) return 0.0;
    return 1.0 / (1.0 + exp(-v));
}


Eigen::VectorXd Network::sigmoidv(Eigen::VectorXd v) {

    for (size_t i=0; i!=v.size(); i++) {
        v[i] = sigmoid(v[i]);
    }

    return v;
}


Eigen::MatrixXd Network::sigmoidm(Eigen::MatrixXd m) {

    for (size_t r=0; r!=m.rows(); r++) {
        for (size_t c=0; c!=m.cols(); c++) {
            m(r,c) = sigmoid(m(r,c));
        }
    }
    
    return m;
}


Eigen::VectorXd Network::sigmoid_prime(Eigen::VectorXd v) {

    for (size_t i=0; i!=v.size(); i++) {
        double s = sigmoid(v[i]);
        v[i] = s * (1 - s);
    }

    return v;
}


Eigen::VectorXd Network::feed_forward(Eigen::VectorXd a) {

    for (int s=0; s!=biases.size(); s++) {
        Eigen::VectorXd b = biases.at(s);
        Eigen::MatrixXd w = weights.at(s);
        a = sigmoidv((w * a).colwise() + b);
    }

    return a;
}


std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> Network::backprop(
            Eigen::VectorXd x, 
            Eigen::VectorXd y)
{
    std::vector<Eigen::VectorXd> nabla_b;
    std::vector<Eigen::MatrixXd> nabla_w;

    for (Eigen::VectorXd b : biases) {
        nabla_b.push_back(Eigen::VectorXd::Zero(b.size()));
    }

    for (Eigen::MatrixXd w : weights) {
        nabla_w.push_back(Eigen::MatrixXd::Zero(w.rows(), w.cols()));
    }

    Eigen::VectorXd activation = x;
    std::vector<Eigen::VectorXd> activations = std::vector<Eigen::VectorXd>{activation};
    std::vector<Eigen::VectorXd> zs;

    for (int i=0; i!=biases.size(); i++) {
        Eigen::VectorXd b = biases.at(i);
        Eigen::MatrixXd w = weights.at(i);
        Eigen::VectorXd z = (w * activation).colwise() + b;
        zs.push_back(z);
        activation = sigmoidv(z);
        activations.push_back(activation);
    }

    Eigen::VectorXd delta = (activation - y).cwiseProduct(sigmoid_prime(zs[zs.size()-1]));
    nabla_b.at(nabla_b.size()-1) = delta;
    nabla_w.at(nabla_w.size()-1) = delta * activations[activations.size()-2].transpose();

    for (int i=2; i!=num_layers; i++) {
        Eigen::VectorXd z = zs[zs.size()-i];
        delta = (weights[weights.size()-i+1].transpose() * delta).cwiseProduct(sigmoid_prime(z));
        nabla_b[nabla_b.size()-i] = delta;
        nabla_w[nabla_w.size()-i] = delta * activations[activations.size()-i-1].transpose();
    }

    return std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>>{nabla_b, nabla_w};
}


void Network::update_mini_batch(std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> mini_batch, double eta) {

    std::vector<Eigen::VectorXd> nabla_b;
    std::vector<Eigen::MatrixXd> nabla_w;

    for (Eigen::VectorXd b : biases) {
        nabla_b.push_back(Eigen::VectorXd::Zero(b.size()));
    }

    for (Eigen::MatrixXd w : weights) {
        nabla_w.push_back(Eigen::MatrixXd::Zero(w.rows(), w.cols()));
    }

    for (std::tuple<Eigen::VectorXd, Eigen::VectorXd> t : mini_batch) {
        Eigen::VectorXd x = std::get<0>(t);
        Eigen::VectorXd y = std::get<1>(t);

        std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> bp = backprop(x, y);
        std::vector<Eigen::VectorXd> delta_nabla_b = std::get<0>(bp);
        std::vector<Eigen::MatrixXd> delta_nabla_w = std::get<1>(bp);

        int i = 0;
        for (Eigen::VectorXd& b : nabla_b) {
            b += delta_nabla_b[i++];
        }

        i = 0;
        for (Eigen::MatrixXd& w : nabla_w) {
            w += delta_nabla_w[i++];
        }
    }

    int i = 0;
    for (Eigen::VectorXd& b : biases) {
        Eigen::VectorXd nb = nabla_b[i++];
        b -= (eta / mini_batch.size()) * nb;
    }

    i = 0;
    for (Eigen::MatrixXd& w : weights) {
        Eigen::MatrixXd nw = nabla_w[i++];
        w -= (eta / mini_batch.size()) * nw;
    }
}


int Network::evaluate(std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> test_data) {

    int success = 0;

    for (auto t : test_data) {
        Eigen::VectorXd in = std::get<0>(t);
        Eigen::VectorXd res = std::get<1>(t);
        Eigen::VectorXd out = feed_forward(in);

        Eigen::Index ri, oi;
        res.maxCoeff(&ri);
        out.maxCoeff(&oi);

        success += ri == oi ? 1 : 0;
    }

    return success;
}


void Network::stochastic_gradient_descent(
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> data,
            size_t epochs,
            size_t mini_batch_size,
            double eta,
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>>* test_data) 
{

    auto rng = std::default_random_engine{};

    for (int i=0; i!=epochs; i++) {
        std::ranges::shuffle(data, rng);
        std::vector<std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>>> mini_batches;
        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> mb;
        for (auto t : data) {
            mb.push_back(t);
            if (mb.size() == mini_batch_size) {
                mini_batches.push_back(mb);
                mb = std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>>();
            }
        }

        if (mb.size() > 0) {
            mini_batches.push_back(mb);
        }

        int c = 0;
        for (auto mb : mini_batches) {
            update_mini_batch(mb, eta);
        }

        if (test_data != nullptr) {
            int success = evaluate(*test_data);
            std::cout << "Epoch " << i << " - " << success << " / " << test_data->size() << std::endl;
        } else {
            std::cout << "Epoch " << i << std::endl;
        }
    }
}
