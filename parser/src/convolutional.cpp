#include "convolutional.h"
#include "common.h"
#include "caffe2.pb.h"
#include <glog/logging.h>

ConvolutionalLayer::ConvolutionalLayer(const caffe2::TensorProto &w,
                                       const caffe2::TensorProto &b,
                                       const caffe2::OperatorDef &op,
                                       std::weak_ptr<AbstractLayer> input,
                                       int schedule) : AbstractLayer(input)
{
    LOG_ASSERT(!input_layer.expired()) << "input layer expired";
    auto input_layer_ = input_layer.lock();
    assert(input_layer_->out_dims() == 4);
    num_samples = input_layer_->out_dim_size(3);
    in_ch = input_layer_->out_dim_size(2);
    in_h = input_layer_->out_dim_size(1);
    in_w = input_layer_->out_dim_size(0);

    params.push_back(LoadBufferFromTensor(w));
    params.push_back(LoadBufferFromTensor(b));

    num_f = params[0].extent(3);
    f_w = params[0].width();
    f_h = params[0].height();

    pad = op.arg(2).i(); // уточнить
    stride = op.arg(0).i();

    forward_clamp = Halide::BoundaryConditions::constant_exterior(
        input_layer_->forward, 0.f,
        0, in_w,
        0, in_h);

    Halide::RDom r(0, f_w, 0, f_h, 0, in_ch);

    forward(x, y, z, n) = params[1](z);
    forward(x, y, z, n) += params[0](r.x, r.y, r.z, z) * forward_clamp(x * stride + r.x - pad,
                                                 y * stride + r.y - pad,
                                                 r.z, n);
    
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
    if (i == 0)
      size = (1 + (in_w + 2 * pad - f_w) / stride);
    else if (i == 1)
      size = (1 + (in_h + 2 * pad - f_h) / stride);
    else if (i == 2)
      size = num_f;
    else if (i == 3)
      size = num_samples;
    return size;
}