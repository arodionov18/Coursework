#include "flatten.h"

using namespace Halide;

FlattenLayer::FlattenLayer(std::weak_ptr<AbstractLayer> input) : AbstractLayer(input) {
    auto layer = input_layer.lock();
    // LOG_ASSERT(layer->out_dims() >= 2 && layer->out_dims() <= 4);

    if (layer->out_dims() == 2) {
        out_width = layer->out_dim_size(0);
        forward(x, n) = layer->forward(x, n);
    } else if (layer->out_dims() == 3) {
        int w = layer->out_dim_size(0);
        int h = layer->out_dim_size(1);
        out_width = w * h;
        forward(x, n) = layer->forward(x % w, x / w, n);
    } else if (layer->out_dims() == 4) {
        int w = layer->out_dim_size(0);
        int h = layer->out_dim_size(1);
        int c = layer->out_dim_size(2);
        out_width = w * h * c;

        forward(x, n) = layer->forward(x % w, (x / w) % h, x / (w * h), n);
    }

    forward.compute_root().parallel(n); // check with schedule
}

void FlattenLayer::back_propagate(Func dout) {
    // LOG_ASSERT(dout.defined()) << "dout is not defined yet";

    if (!f_in_grad.defined()) {
        auto layer = input_layer.lock();
        if (layer->out_dims() == 2) {
            f_in_grad(x, n) = dout(x, n);
        } else if (layer->out_dims() == 3) {
            int w = layer->out_dim_size(0);
            f_in_grad(x, y, n) = dout(y * w + x, n);
        } else if (layer->out_dims() == 4) {
            int w = layer->out_dim_size(0);
            int h = layer->out_dim_size(1);
            f_in_grad(x, y, z, n) = dout(z * w * h + y * w + x, n);
        }
        layer->back_propagate(f_in_grad);
    }
}

int FlattenLayer::out_dims() const {
    return 2;
}

int FlattenLayer::out_dim_size(int i) const {
    // LOG_ASSERT(i < 2) << "Wrong index";
    int size = 0;
    if (i == 0) {
        size = out_width;
    } else {
        size = num_samples;
    }
    return size; // NRVO;
}