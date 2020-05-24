#include "fully_connected.h"
#include "common.h"
#include <glog/logging.h>

FCLayer::FCLayer(const caffe2::TensorProto& w, const caffe2::TensorProto& b, std::shared_ptr<AbstractLayer> input, int schedule) : AbstractLayer(input) {
    assert(input_layer->out_dims() == 2);
    //Halide::RDom r(0, input_layer->out_dim_size(1));
    // forward.trace_stores();

    num_samples = input_layer->out_dim_size(0);
    // std::cerr << input_layer->out_dim_size(1) << std::endl;
    assert(input_layer->out_dim_size(1) == w.dims(1));
    out_width = w.dims(0); // (M, K) * (N, K)^T = (M, N)
    //                        (1, K) * (4096, K)^T = (1, 4096)

    // std::cerr << "FC: W: " << w.dims_size() << std::endl;
    for (int i = 0; i < w.dims_size(); ++i) {
        // std::cerr << "dim: " << i << ", " << w.dims(i) << std::endl;
    }

    // std::cerr << "FC: b: " << b.dims_size() << std::endl;
    for (int i = 0; i < b.dims_size(); ++i) {
        // std::cerr << "dim: " << i << ", " << b.dims(i) << std::endl;
    }

    params.push_back(LoadBufferFromTensor(w));
    params.push_back(LoadBufferFromTensor(b));

    RDom r(0, input_layer->out_dim_size(1));
    forward(x, y) = params[1](y);
    forward(x, y) += input_layer->forward(x, r.x) * params[0](y, r.x);

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