#include "fully_connected.h"
#include "common.h"
#include <glog/logging.h>

FCLayer::FCLayer(const caffe2::TensorProto& w, const caffe2::TensorProto& b, std::weak_ptr<AbstractLayer> input, int schedule) : AbstractLayer(input) {
    auto input_layer_ = input_layer.lock();
    
    Halide::RDom r(0, input_layer_->out_dim_size(0));

    params.push_back(LoadBufferFromTensor(w));
    params.push_back(LoadBufferFromTensor(b));

    forward(x, y, z, n) = sum(params[0](x, r.x) * input_layer_->forward(r.x, y, z, n));
    forward(x, y, z, n) += params[1](x);
}

void FCLayer::back_propagate(Halide::Func dout) {
    // LOG_ASSERT(false) << "NOT IMPLEMENTED YET" << std::endl;
}

int FCLayer::out_dims() const {
    return 4;
}

int FCLayer::out_dim_size(int i) const {
    // LOG_ASSERT(i < 4) << "Wrong number";

    return input_layer.lock()->out_dim_size(i);
}