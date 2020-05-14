#include "softmax.h"

using namespace Halide;

SoftmaxLayer::SoftmaxLayer(std::weak_ptr<AbstractLayer> input, int schedule) : AbstractLayer(input) {
    auto layer = input_layer.lock();
    // LOG_ASSERT(layer->out_dims() == 2);

    num_classes = layer->out_dim_size(0);
    num_samples = layer->out_dim_size(1);

    Func exp_max, expo, normalizer;
    RDom r(0, num_classes);
    exp_max(n) = maximum(layer->forward(r.x, n));
    expo(in_dim) = exp(layer->forward(in_dim, n) - exp_max(n));
    normalizer(n) = cast(layer->forward.output_types()[0], 0);
    normalizer(n) += expo(r.x, n);
    forward(in_dim, n) = expo(in_dim, n) / normalizer(n);

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
        input_layer.lock()->back_propagate(f_in_grad);
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
    if (i == 0) {
        size = num_classes;
    } else if (i == 1) {
        size = num_samples;
    }
    return size;
}