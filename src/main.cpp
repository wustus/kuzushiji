
#include "reader.h"
#include "network.h"
#include <cstdlib>
#include <iostream>

int main() {
    Network n(std::vector<int>{784, 30, 10});
    std::cout << "Reading Labels..." << std::endl;
    auto labels = read_labels("data/labels");
    std::cout << "Reading Images..." << std::endl;
    auto images = read_data("data/images");

    std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd>> data, test_data;

    for (int i=0; i!=images.size(); i++) {
        auto im = images[i];
        auto la = labels[i];
        if (i < 50000) {
            data.push_back(std::tuple<Eigen::VectorXd, Eigen::VectorXd>{im, la});
        } else {
            test_data.push_back(std::tuple<Eigen::VectorXd, Eigen::VectorXd>{im, la});
        }
    }

    Eigen::initParallel();
    std::cout << Eigen::nbThreads() << " threads." << std::endl;
    
    std::cout << "Start Training..." << std::endl;

    int epochs = 30, bs = 10;
    double eta = 3;

    const char* EPOCH_ENV = std::getenv("EPOCH");

    if (EPOCH_ENV != nullptr) {
        epochs = atoi(EPOCH_ENV);
    }

    const char* BATCH_SIZE_ENV = std::getenv("BATCH_SIZE");

    if (BATCH_SIZE_ENV != nullptr) {
        bs = atoi(BATCH_SIZE_ENV);
    }

    const char* ETA_ENV = std::getenv("ETA");

    if (ETA_ENV != nullptr) {
        eta = atof(ETA_ENV);
    }

    std::cout << "--------------------" << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Batch Size: " << bs << std::endl;
    std::cout << "Eta: " << eta << std::endl;
    std::cout << "--------------------" << std::endl;

    n.stochastic_gradient_descent(data, epochs, bs, eta, &test_data);
    return 0;
}
