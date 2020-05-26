#include "lrn.h"

using namespace Halide;

LRNLayer::LRNLayer(const caffe2::OperatorDef& op, std::shared_ptr<AbstractLayer> input) : AbstractLayer(input) {
    auto layer = input_layer;
    
    in_w = layer->out_dim_size(3);
    in_h = layer->out_dim_size(2);
    in_ch = layer->out_dim_size(1);
    num_samples = layer->out_dim_size(0);

    // Across-chanel lrn
    size = op.arg(0).i();
    alpha = op.arg(1).f();
    beta = op.arg(2).f();
    bias = op.arg(3).f();

    Func clamped = BoundaryConditions::constant_exterior(layer->forward, 0.0f, 0, num_samples, 0, in_ch, 0, in_h, 0, in_w);

    RDom r(0, size);

    Func normalizer;
    normalizer(n, z, y, x) = sum(pow(clamped(n, max(min(0, z - size / 2 + r.x), in_ch), y, x), 2.0f));

    forward(n, z, y, x) = clamped(n, z, y, x) / pow((bias + alpha * normalizer(n, z, y, x)), beta);

    //forward.bound(z, 0, in_ch);
    //clamped.compute_root();
    //normalizer.compute_root().parallel(n);
    forward.compute_root().parallel(n);
}

void LRNLayer::back_propagate(Func dout) {
    assert(false); // NOT IMPLIMENTED
}

int LRNLayer::out_dims() const {
    return 4;
}

int LRNLayer::out_dim_size(int i) const {
    switch (i)
    {
    case 0:
        return num_samples;

    case 1:
        return in_ch;
    
    case 2:
        return in_h;
    
    case 3:
        return in_w;
    
    default:
        assert(false);
        break;
    }
}