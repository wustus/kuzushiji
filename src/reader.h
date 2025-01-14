
#ifndef reader_h
#define reader_h

#include <mlx/array.h>
#include <vector>

namespace mx = mlx::core;

std::vector<mx::array> read_labels(const char* path);
std::vector<mx::array> read_images(const char* path);

#endif
