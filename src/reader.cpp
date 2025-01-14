
#include "reader.h"

#include <cassert>
#include <iostream>
#include <fcntl.h>
#include <mlx/dtype.h>
#include <unistd.h>
#include <mlx/ops.h>
#include <sys/fcntl.h>
#include <errno.h>

int flip_bytes(unsigned char* buf, int size) {

    int res = 0;

    for (int i=0; i!=size; i++) {
        res |= buf[size-(i+1)] << (8*i);
    }

    return res;
}

std::vector<mx::array> read_labels(const char* path) {

    int fd, res;
    if (fd=open(path, O_RDONLY); fd<0) {
        std::cerr << "Could not open file: " << strerror(errno) << std::endl;
        close(fd);
        exit(1);
    }

    unsigned char buf[1024];
    std::memset(buf, 0, 1024);

    if (res=read(fd, buf, 4); res<0) {
        std::cerr << "Could not read magic number from file: " << strerror(errno) << std::endl;
        close(fd);
        exit(1);
    }

    int magic_number, n_items;

    magic_number = flip_bytes(buf, 4);
    assert(magic_number == 2049);

    std::memset(buf, 0, 1024);
    if (res=read(fd, buf, 4); res<0) {
        std::cerr << "Could not read number of items from file: " << strerror(errno) << std::endl;
        close(fd);
        exit(1);
    }

    n_items = flip_bytes(buf, 4);
    assert(n_items == 60000);

    std::vector<mx::array> labels{};

    std::memset(buf, 0, 1024);
    while (read(fd, buf, 1000) > 0) {

        for (int i=0; i!=1000; i++) {
            float data[10];
            std::memset(data, 0, 10*sizeof(float));
            data[int(buf[i])] = 1.0f;
            mx::array m = mx::array(data, {10,1}, mx::float32);
            labels.push_back(m);
        }

        std::memset(buf, 0, 1024);
    }

    return labels;
}


std::vector<mx::array> read_images(const char* path) {

    int fd, res;
    if (fd=open(path, O_RDONLY); fd<0) {
        std::cerr << "Could not open file: " << strerror(errno) << std::endl;
        close(fd);
        exit(1);
    }

    unsigned char buf[1024];
    std::memset(buf, 0, 1024);

    int magic_number, n_items, n_rows, n_cols;


    std::memset(buf, 0, 1024);
    if (res=read(fd, buf, 16); res<0) {
        std::cerr << "Could not read numbers from file: " << strerror(errno) << std::endl;
        close(fd);
        exit(1);
    }

    magic_number = flip_bytes(buf, 4);
    assert(magic_number == 2051);

    n_items = flip_bytes(buf+4, 4);
    assert(n_items == 60000);

    n_rows = flip_bytes(buf+8, 4);
    assert(n_rows == 28);

    n_cols = flip_bytes(buf+12, 4);
    assert(n_cols == 28);

    std::vector<mx::array> images{};

    std::memset(buf, 0, 1024);

    int r_pixels=0, n_pixels=n_rows*n_cols;
    while (r_pixels=read(fd, buf, n_pixels), r_pixels==n_pixels) {
        float data[n_pixels];
        std::memset(data, 0, n_pixels*sizeof(float));
        for (int i=0; i!=n_pixels; i++) {
            data[i] = float(buf[i]) / 255.0f;
        }
        
        mx::array m = mx::array(data, {n_rows*n_cols, 1}, mx::float32);
        images.push_back(m);

        std::memset(buf, 0, 1024);
    }
    close(fd);

    return images;
}
