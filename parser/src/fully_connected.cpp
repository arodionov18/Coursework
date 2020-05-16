#include "fully_connected.h"
#include "common.h"
#include <glog/logging.h>

FCLayer::FCLayer(const caffe2::TensorProto& w, const caffe2::TensorProto& b, std::shared_ptr<AbstractLayer> input, int schedule) : AbstractLayer(input) {
    auto input_layer_ = input_layer;
    
    Halide::RDom r(0, input_layer_->out_dim_size(3));

    std::cout << "FC: W: " << w.dims_size() << std::endl;
    for (int i = 0; i < w.dims_size(); ++i) {
        std::cout << "dim: " << i << ", " << w.dims(i) << std::endl;
    }

    std::cout << "FC: b: " << b.dims_size() << std::endl;
    for (int i = 0; i < b.dims_size(); ++i) {
        std::cout << "dim: " << i << ", " << b.dims(i) << std::endl;
    }

    params.push_back(LoadBufferFromTensor(w));
    params.push_back(LoadBufferFromTensor(b));

    forward(n, z, y, x) = sum(params[0](x, r.x) * input_layer_->forward(n, z, y, r.x));
    forward(n, z, y, x) += params[1](z);
}

void FCLayer::back_propagate(Halide::Func dout) {
    // LOG_ASSERT(false) << "NOT IMPLEMENTED YET" << std::endl;
}

int FCLayer::out_dims() const {
    return 4;
}

int FCLayer::out_dim_size(int i) const {
    // LOG_ASSERT(i < 4) << "Wrong number";

    return input_layer->out_dim_size(i);
}