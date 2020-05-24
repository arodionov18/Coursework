#include "flatten.h"

using namespace Halide;

FlattenLayer::FlattenLayer(std::shared_ptr<AbstractLayer> input) : AbstractLayer(input) {
    auto layer = input_layer;
    // LOG_ASSERT(layer->out_dims() >= 2 && layer->out_dims() <= 4);
    // forward.trace_stores();
    num_samples = layer->out_dim_size(0);

    if (layer->out_dims() == 2) {
        out_width = layer->out_dim_size(1);
        forward(n, x) = layer->forward(n, x);
    } else if (layer->out_dims() == 3) {
        int w = layer->out_dim_size(2);
        int c = layer->out_dim_size(1);
        out_width = w * c;
        forward(n, x) = layer->forward(n, x / w, x % w);
    } else if (layer->out_dims() == 4) {
        int w = layer->out_dim_size(3);
        int h = layer->out_dim_size(2);
        int c = layer->out_dim_size(1);
        out_width = w * h * c;
        /*Halide::Buffer<float> debug_output = layer->forward.realize({1, c, h, w});
        for (size_t i = 0; i < c; ++i) {
            // std::cerr << std::endl << "C: " << c << std::endl;
            for (size_t j = 0; j < h; ++j) {
                for (size_t k = 0; k < w; ++k) {
                    // std::cerr << debug_output(0, i, j, k) << " ";
                }
                // std::cerr << std::endl;
            } 
        }*/

        forward(n, x) = layer->forward(n, x / (w * h), (x / w) % h, x % w);
    }

    forward.compute_root().parallel(n); // check with schedule
}

void FlattenLayer::back_propagate(Func dout) {
    // LOG_ASSERT(dout.defined()) << "dout is not defined yet";

    if (!f_in_grad.defined()) {
        auto layer = input_layer;
        if (layer->out_dims() == 2) {
            f_in_grad(n, x) = dout(n, x);
        } else if (layer->out_dims() == 3) {
            int w = layer->out_dim_size(2);
            f_in_grad(n, y, x) = dout(n, y * w + x);
        } else if (layer->out_dims() == 4) {
            int w = layer->out_dim_size(3);
            int h = layer->out_dim_size(2);
            f_in_grad(n, z, y, x) = dout(n, z * w * h + y * w + x);
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
    if (i == 1) {
        size = out_width;
    } else {
        size = num_samples;
    }
    return size;
}