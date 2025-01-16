
#ifndef network_h
#define network_h

#include <vector>
#include <Eigen/Dense>

class Network {
private:
    int num_layers;
    std::vector<int> sizes;
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::MatrixXd> weights;
public:
    Network(std::vector<int> sizes);
    double sigmoid(double v);
    Eigen::VectorXd sigmoidv(Eigen::VectorXd v);
    Eigen::MatrixXd sigmoidm(Eigen::MatrixXd m);
    Eigen::VectorXd sigmoid_prime(Eigen::VectorXd v);
    Eigen::VectorXd feed_forward(Eigen::VectorXd a);

    std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backprop(
            Eigen::VectorXd x, 
            Eigen::VectorXd y);
    void update_mini_batch(std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> mini_batch, double eta);
    int evaluate(std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> test_data);
    void stochastic_gradient_descent(
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> data,
            size_t epochs,
            size_t mini_batch_size,
            double eta,
            std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>>* test_data);
};

#endif
