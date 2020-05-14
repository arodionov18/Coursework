#include "common.h"
#include <glog/logging.h>

Halide::Buffer<float> LoadBufferFromTensor(const caffe2::TensorProto& tensor) {
    int width = 0, height = 0, channel = 0, number = 0;
    std::cout << "everything is good" << std::endl;

    if (tensor.dims().size() == 4) {
        // why this order?
        channel = tensor.dims(1);
        width = tensor.dims(3);
        height = tensor.dims(2);
        number = tensor.dims(0);
    } else {
        channel = tensor.dims(0);
    }
    std::cout << "stil food" << std::endl;
    Halide::Buffer<float> image(number, channel, height, width);

    int idx = 0;
    if (tensor.dims().size() == 4) {
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
    } else {
        for (int c = 0; c < channel; ++c) {
            image(1, c, 1, 1) = tensor.float_data(idx);
            ++idx;
        }
    }

    LOG(INFO) << "Finished LoadingImageFromTensor";

    return image;
}