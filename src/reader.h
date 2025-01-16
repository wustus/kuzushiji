#include <cstddef>
#include <Eigen/Dense>

#ifndef reader_h
#define reader_h

typedef unsigned char uchar;

uint flip_bytes(char* buf, size_t size);

std::vector<Eigen::VectorXd> read_labels(const char* path);
std::vector<Eigen::VectorXd> read_data(const char* path);

#endif
