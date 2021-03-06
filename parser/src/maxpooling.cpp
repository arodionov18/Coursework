#include "maxpooling.h"

using namespace Halide;

MaxPoolingLayer::MaxPoolingLayer(const caffe2::OperatorDef& op, std::shared_ptr<AbstractLayer> input, int schedule) : AbstractLayer(input) {
    auto layer = input_layer;
    // LOG_ASSERT(layer->out_dims() == 4);
    //forward.trace_stores();

    num_samples = layer->out_dim_size(0);
    in_ch = layer->out_dim_size(1);
    in_h = layer->out_dim_size(2);
    in_w = layer->out_dim_size(3);

    p_h = op.arg(2).i();
    p_w = op.arg(2).i(); // CHECK THIS NORMAL
    pad = op.arg(1).i();
    stride = op.arg(0).i();

    //assert((in_h - p_h + 2 * pad) % stride == 0);// << "Bad Hpad";
    // assert((in_w - p_w + 2 * pad) % stride == 0);// << "Bad Wpad";

    Func forward_clamp = BoundaryConditions::constant_exterior(layer->forward, 0.0f,
                                                                 0, num_samples,
                                                                 0, in_ch,
                                                                 0, in_h,
                                                                 0, in_w);

    RDom r(0, p_w, 0, p_h);
    forward(n, z, y, x) = maximum(forward_clamp(n, z,
                                                 y * stride + r.y - pad,
                                                 x * stride + r.x - pad));
    //forward.bound(z, 0, in_ch);
    if (schedule) {
        forward.vectorize(x, vec_len);
        forward.compute_root().fuse(n, z, par).parallel(par);
    }
}

void MaxPoolingLayer::back_propagate(Func dout) {
    // LOG_ASSERT(dout.defined()) << "dout is not defined yet";

    if (!f_in_grad.defined()) {
        auto layer = input_layer;

        Func pool_argmax;
        RDom r1(0, p_w, 0, p_h);
        pool_argmax(x, y, z, n) = argmax(layer->forward(x * stride + p_w, 
                                                        y * stride + p_h,
                                                        z, n));
        pool_argmax.compute_root();
        RDom r2(0, out_dim_size(0), 0, out_dim_size(1));
        f_in_grad(x, y, z, n) = cast(dout.output_types()[0], 0);

        Expr x_bin = clamp(r2.x * stride + pool_argmax(r2.x, r2.y, z, n)[0], 0, in_w);
        Expr y_bin = clamp(r2.y * stride + pool_argmax(r2.x, r2.y, z, n)[1], 0, in_h);

        f_in_grad(x_bin, y_bin, z, n) += dout(r2.x, r2.y, z, n);
        layer->back_propagate(f_in_grad);
    }
}

int MaxPoolingLayer::out_dims() const {
    return 4;
}

int MaxPoolingLayer::out_dim_size(int i) const {
    // LOG_ASSERT(i < 4) << "Wrong input";

    int size = 0; // NRVO
    if (i == 3) {
        size = 1 + ((in_w - p_w + 2 * pad) / stride);
    } else if (i == 2) {
        size = 1 + ((in_h - p_h + 2 * pad) / stride);
    } else if (i == 1) {
        size = in_ch;
    } else {
        size = num_samples;
    }
    return size;
}