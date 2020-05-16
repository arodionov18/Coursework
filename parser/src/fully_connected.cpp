#include "fully_connected.h"
#include "common.h"
#include <glog/logging.h>

FCLayer::FCLayer(const caffe2::TensorProto& w, const caffe2::TensorProto& b, std::shared_ptr<AbstractLayer> input, int schedule) : AbstractLayer(input) {
    assert(input_layer->out_dims() == 2);
    //Halide::RDom r(0, input_layer->out_dim_size(1));

    num_samples = input_layer->out_dim_size(0);
    out_width = w.dims(0); // (M, K) * (N, K)^T = (M, N)

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

    RDom r(0, num_samples);
    forward(x, y) = params[1](x);
    forward(x, y) += params[0](r.x, x) * input_layer->forward(r.x, y);

    if (schedule) {
        forward.compute_root().fuse(x, y, par).parallel(par);
        forward.update().fuse(x, y, par).parallel(par);
    }
}

void FCLayer::back_propagate(Halide::Func dout) {
    // LOG_ASSERT(false) << "NOT IMPLEMENTED YET" << std::endl;
}

int FCLayer::out_dims() const {
    return 2;
}

int FCLayer::out_dim_size(int i) const {
    // LOG_ASSERT(i < 4) << "Wrong number";
    int size = 0;
    if (i == 0) {
        size = num_samples;
    } else {
        size = out_width;
    }
    return size;
}