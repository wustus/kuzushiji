
#include "reader.h"
#include "Eigen/src/Core/Matrix.h"
#include <iostream>
#include <errno.h>
#include <sys/fcntl.h>
#include <unistd.h>

uint flip_bytes(uchar* buf, size_t size) {
    int flipped = 0;

    if (size < 1 || size > 4) {
        std::cout << "Can only flip between 1 and 4 bytes." << std::endl;
        exit(1);
    }

    for (int i=0; i!=size; i++) {
        flipped |= buf[size - (i+1)] << (8 * i);
    }

    return flipped;
}

std::vector<Eigen::VectorXd> read_labels(const char *path) {

    int fd;
    std::vector<Eigen::VectorXd> res;

    if (fd=open(path, O_RDONLY); fd<0) {
        std::cerr << "Unable to open file: " << strerror(errno) << std::endl;
        exit(1);
    }

    int r_bytes;
    uchar buf[8];

    std::memset(buf, 0, 8);

    if (r_bytes=read(fd, buf, 4); r_bytes<4) {
        std::cerr << "Could not read magic number." << std::endl;
        exit(1);
    }

    uint magic_number = flip_bytes(buf, 4);
    if (magic_number != 2049) {
        std::cerr << "Magic Number != 2049" << std::endl;
        exit(1);
    }

    std::memset(buf, 0, 8);

    if (r_bytes=read(fd, buf, 4); r_bytes<4) {
        std::cerr << "Could not read number of items." << std::endl;
        exit(1);
    }

    uint n_items = flip_bytes(buf, 4);

    std::memset(buf, 0, 8);
    int c=1;

    while (r_bytes=read(fd, buf, 1), r_bytes>0) {
        Eigen::VectorXd v(10);
        v[int(buf[0])] = 1.0;
        res.push_back(v);
    }

    close(fd);

    return res;
}


std::vector<Eigen::VectorXd> read_data(const char* path) {

    int fd;
    std::vector<Eigen::VectorXd> res;

    if (fd=open(path, O_RDONLY); fd<0) {
        std::cerr << "Unable to open file: " << strerror(errno) << std::endl;
        exit(1);
    }

    int r_bytes;
    uchar buf[1024];

    if (r_bytes=read(fd, buf, 4); r_bytes<4) {
        std::cerr << "Could not read magic number." << std::endl;
        exit(1);
    }

    uint magic_number = flip_bytes(buf, 4);
    if (magic_number != 2051) {
        std::cerr << "Magic Number != 2051" << std::endl;
        exit(1);
    }

    uint n_images, n_row, n_col;

    // Number of Images
    std::memset(buf, 0, 1024);
    if (r_bytes=read(fd, buf, 4); r_bytes<4) {
        std::cerr << "Could not read number of images." << std::endl;
        exit(1);
    }

    n_images = flip_bytes(buf, 4);;

    if (n_images != 60000) {
        std::cerr << "Number of Images != 60000" << std::endl;
        exit(1);
    }

    // Number of Rows
    std::memset(buf, 0, 1024);
    if (r_bytes=read(fd, buf, 4); r_bytes<4) {
        std::cerr << "Could not read number of rows." << std::endl;
        exit(1);
    }

    n_row = flip_bytes(buf, 4);;

    if (n_row != 28) {
        std::cerr << "Number of Rows != 28" << std::endl;
        exit(1);
    }

    // Number of Cols
    std::memset(buf, 0, 1024);
    if (r_bytes=read(fd, buf, 4); r_bytes<4) {
        std::cerr << "Could not read number of cols." << std::endl;
        exit(1);
    }

    n_col = flip_bytes(buf, 4);;

    if (n_col != 28) {
        std::cerr << "Number of Rows != 28" << std::endl;
        exit(1);
    }

    int n_pixels = n_row * n_col;
    while(r_bytes=read(fd, buf, n_pixels), r_bytes==n_pixels) {
        Eigen::VectorXd img(n_pixels);
        for (int i=0; i!=n_pixels; i++) {
            double v = double(buf[i]) / 255.0f;
            img[i] = v;
        }
        res.push_back(img);
        std::memset(buf, 0, 1024);
    }

    close(fd);

    return res;
}
