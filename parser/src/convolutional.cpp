#include "convolutional.h"
#include "common.h"
#include "caffe2.pb.h"
#include <glog/logging.h>

ConvolutionalLayer::ConvolutionalLayer(const caffe2::TensorProto &w,
                                       const caffe2::TensorProto &b,
                                       const caffe2::OperatorDef &op,
                                       std::shared_ptr<AbstractLayer> input,
                                       int schedule) : AbstractLayer(input)
{
    // LOG_ASSERT(!input_layer.expired()) << "input layer expired";
    auto input_layer_ = input_layer;
    assert(input_layer_->out_dims() == 4);
    num_samples = input_layer_->out_dim_size(0);
    in_ch = input_layer_->out_dim_size(1);
    in_h = input_layer_->out_dim_size(2);
    in_w = input_layer_->out_dim_size(3);
    std::cout << "Convolutional layer" << std::endl;
    std::cout << "Params dims: " << w.dims_size() << std::endl;
    for (int i = 0; i < w.dims_size(); ++i) {
        std::cout << "dim " << i << ", " << w.dims(i) << std::endl;
    }
    f_w = w.dims(3);
    f_h = w.dims(2);
    num_f = w.dims(0);
    params.push_back(LoadBufferFromTensor(w));
    params.push_back(LoadBufferFromTensor(b));
    assert(num_f == b.dims(0));

    pad = op.arg(1).i(); // уточнить
    stride = op.arg(0).i();

    forward_clamp = Halide::BoundaryConditions::constant_exterior(
        input_layer_->forward, 0.0f,
        0, num_samples,
        0, in_ch,
        0, in_h,
        0, in_w);

    Halide::RDom r(0, f_w, 0, f_h, 0, in_ch);

    forward(n, z, y, x) = params[1](z);
    forward(n, z, y, x) += params[0](z, r.z, r.y, r.x) * forward_clamp(n, r.z,
                                                                       y * stride + r.y - pad,
                                                                       x * stride + r.x - pad);
    
    if (schedule) {
        o_block_size = 16;
        y_block_size = 32;
        vec_len = 8;
        forward.update().reorder(y, x, r.z);
        // blocking spatially with vectorization
        // forward_clamp.compute_at(f_simple, n);
        forward.compute_root();
        forward.fuse(z, n, par).parallel(par);
        forward.update().reorder(x, y, r.z);
        forward.update().split(y, y, y_t, y_block_size);
        forward.update().split(z, z, z_t, o_block_size);
        forward.update().reorder(y_t, z_t, y, r.z, z);
        forward.update().vectorize(x, vec_len);
        forward.update().fuse(z, n, par).parallel(par);
        //forward.update().fuse(y, par, par).parallel(par);
        forward.update().unroll(r.x);
        forward.update().unroll(r.y);
        // There are performance implications to this and seems to
        // be incompatible with some schedules. Have to investigate
        // this more closely.
        //forward_clamp.compute_at(forward, n);
        forward_clamp.compute_at(forward, z_t);
    }
}

void ConvolutionalLayer::back_propagate(Halide::Func dout) {
    // NOT IMPLEMENTED
}

int ConvolutionalLayer::out_dims() const {
    return 4;
}

int ConvolutionalLayer::out_dim_size(int i) const {
    int size = 0;
    if (i == 3)
      size = (1 + (in_w + 2 * pad - f_w) / stride);
    else if (i == 2)
      size = (1 + (in_h + 2 * pad - f_h) / stride);
    else if (i == 1)
      size = num_f;
    else if (i == 0)
      size = num_samples;
    return size;
}