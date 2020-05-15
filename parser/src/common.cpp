#include "common.h"
#include <glog/logging.h>

Halide::Buffer<float> LoadBufferFromTensor(const caffe2::TensorProto& tensor) {
    int width = 0, height = 0, channel = 0, number = 0;
    std::cout << tensor.name() << ": " << tensor.dims_size();
    if (tensor.dims_size() == 4) {
        // why this order?
        channel = tensor.dims(1);
        width = tensor.dims(3);
        height = tensor.dims(2);
        number = tensor.dims(0);
    } else if (tensor.dims_size() == 2) {
        number = tensor.dims(0);
        channel = tensor.dims(1);
    } else {
        channel = tensor.dims(0);
    }
    
    
    if (tensor.dims_size() == 4) {
        Halide::Buffer<float> image(number, channel, height, width);
        int idx = 0;
        for (int n = 0; n < number; ++n) {
            for (int c = 0; c < channel; ++c) {
                for (int j = 0; j < height; ++j) {
                    for (int i = 0; i < width; ++i) {
                        image(n, c, j, i) = tensor.float_data(idx);
                        ++idx;
                    }
                }
            }
        }
        return image;
    } else if (tensor.dims_size() == 2) {
        Halide::Buffer<float> image(number, channel);
        int idx = 0;
        for (int n = 0; n < number; ++n) {
            for (int c = 0; c < channel; ++c) {
                image(n, c) = tensor.float_data(idx);
                ++idx;
            }
        }
        return image;
    } else {
        Halide::Buffer<float> image(channel);
        for (int c = 0; c < channel; ++c) {
            image(c) = tensor.float_data(c);
        }
        return image;
    }
    //std::cout << tensor.name() << " finished";

    //LOG(INFO) << "Finished LoadingImageFromTensor";

    //return image;
}