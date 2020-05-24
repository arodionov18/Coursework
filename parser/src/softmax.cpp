#include "softmax.h"

using namespace Halide;

SoftmaxLayer::SoftmaxLayer(std::shared_ptr<AbstractLayer> input, int schedule) : AbstractLayer(input) {
    //forward.trace_stores();
    auto layer = input_layer;
    // LOG_ASSERT(layer->out_dims() == 2);
    // std::cerr << "Softmax: " << layer->out_dims() << std::endl;
    for (int i = 0; i < layer->out_dims(); ++i) {
        // std::cerr << "Dim " << i << ", " << layer->out_dim_size(i) << std::endl;
    }

    num_classes = layer->out_dim_size(1);
    num_samples = layer->out_dim_size(0);

    Func exp_max, expo, normalizer;
    RDom r(0, num_classes);
    exp_max(n) = maximum(layer->forward(n, r.x));
    expo(n, in_dim) = exp(layer->forward(n, in_dim) - exp_max(n));
    normalizer(n) = cast<float>(0);
    normalizer(n) += expo(n, r.x);
    forward(n, in_dim) = expo(n, in_dim) / normalizer(n);

    if (schedule) {
        exp_max.compute_at(forward, n);
        expo.compute_at(forward, n);
        normalizer.compute_at(forward, n);
        forward.compute_root().parallel(n);
    }
}

void SoftmaxLayer::back_propagate(Func labels) {
    if (!f_in_grad.defined()) {
        // LOG_ASSERT(labels.defined()) << "labels is not defined yet";
        // LOG_ASSERT(forward.defined()) << "forward is not defined yet";

        Expr label = clamp(labels(n), 0, num_classes - 1);
        Expr t = (forward(in_dim, n) - 1) / num_samples;
        Expr f = (forward(in_dim, n) / num_samples);
        f_in_grad(in_dim, n) = select(in_dim == label, t, f);
        input_layer->back_propagate(f_in_grad);
    }
}

Func SoftmaxLayer::loss(Func labels) {
    // LOG_ASSERT(labels.defined()) << "labels is not defined yet";
    // LOG_ASSERT(labels.dimensions() == 1) << "dimensions is not right";

    Var x;
    Func loss_p;
    RDom r(0, num_samples);
    loss_p(x) = cast(forward.output_types()[0], 0);
    loss_p(0) += -log(forward(clamp(labels(r.x), 0, num_classes - 1), r.x)) / num_samples;
    return loss_p;
}

int SoftmaxLayer::out_dims() const {
    return 2;
}

int SoftmaxLayer::out_dim_size(int i) const {
    // LOG_ASSERT(i < 2) << "wrong input";
    int size = 0;
    if (i == 1) {
        size = num_classes;
    } else if (i == 0) {
        size = num_samples;
    }
    return size;
}