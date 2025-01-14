
#include "reader.h"
#include "network.h"

int main() {

    Network n(std::vector<int>{784, 30, 10});
    
    std::vector<mx::array> la=read_labels("data/labels"), im=read_images("data/images");
    
    std::vector<std::tuple<mx::array, mx::array>> data, test_data;

    for (int i=0; i!=im.size(); i++) {
        if (i<50000) {
            data.push_back(std::tuple<mx::array, mx::array>{im[i], la[i]});
            continue;
        }
        test_data.push_back(std::tuple<mx::array, mx::array>{im[i], la[i]});
    }

    const char* EPOCHS_VAR=std::getenv("EPOCH");
    const char* BATCH_SIZE_VAR=std::getenv("BATCH");
    const char* ETA_VAR=std::getenv("ETA");

    int epochs=30, batch_size=10;
    double eta=1.0;

    if (EPOCHS_VAR != nullptr) {
        epochs = atoi(EPOCHS_VAR);
    }

    if (BATCH_SIZE_VAR != nullptr) {
        batch_size = atoi(BATCH_SIZE_VAR);
    }

    if (ETA_VAR != nullptr) {
        eta = atof(ETA_VAR);
    }

    n.stochastic_gradient_descent(data, epochs, batch_size, eta, &test_data);

    return 0;
}
