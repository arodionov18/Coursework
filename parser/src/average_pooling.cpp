#include "average_pooling.h"

using namespace Halide;

AveragePoolingLayer::AveragePoolingLayer(const caffe2::OperatorDef& op, std::shared_ptr<AbstractLayer> input, int schedule): AbstractLayer(input) {

    num_samples = input_layer->out_dim_size(0);
    in_ch = input_layer->out_dim_size(1);
    in_h = input_layer->out_dim_size(2);
    in_w = input_layer->out_dim_size(3);

    stride = op.arg(0).i();
    p_l = op.arg(1).i();
    p_t = op.arg(2).i();
    p_r = op.arg(3).i();
    p_b = op.arg(4).i();
    kernel = op.arg(5).i();

    Func forward_clamp = BoundaryConditions::constant_exterior(input_layer->forward, 0.0f,
                                                                 0, num_samples,
                                                                 0, in_ch,
                                                                 0, in_h,
                                                                 0, in_w);

    RDom r(0, kernel, 0, kernel);
    forward(n, z, y, x) = sum(forward_clamp(n, z,
                                            y * stride + r.y - p_l,
                                            x * stride + r.x - p_t)) / (kernel * kernel * 1.0f);
    
    if (schedule) {
        forward.vectorize(x, vec_len);
        forward.compute_root().fuse(n, z, par).parallel(par);
    }
    //forward.bound(z, 0, in_ch);

}

void AveragePoolingLayer::back_propagate(Func dout) {
    // LOG_ASSERT(dout.defined()) << "dout is not defined yet";
}

int AveragePoolingLayer::out_dims() const {
    return 4;
}

int AveragePoolingLayer::out_dim_size(int i) const {
    // LOG_ASSERT(i < 4) << "Wrong input";

    int size = 0; // NRVO
    if (i == 3) {
        size = 1 + ((in_w - kernel + p_l + p_r) / stride);
    } else if (i == 2) {
        size = 1 + ((in_h - kernel + p_l + p_r) / stride);
    } else if (i == 1) {
        size = in_ch;
    } else {
        size = num_samples;
    }
    return size;
}