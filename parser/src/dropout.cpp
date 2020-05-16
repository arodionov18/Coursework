#include "dropout.h"

using namespace Halide;

DropoutLayer::DropoutLayer(const caffe2::OperatorDef& op, std::shared_ptr<AbstractLayer> input) : AbstractLayer(input) {
    auto layer = input_layer;

    ratio = op.arg(0).f(); // ratio
    is_test = op.arg(1).i(); // phase

    Expr scale = 1.0f / (1.0f - ratio);

    switch (layer->out_dims())
    {
    case 1:
        mask(x) = select(random_float() > ratio, scale, 0.0f);
        if (is_test) {
            forward(x) = layer->forward(x);
        } else {
            forward(x) = mask(x) * layer->forward(x);
        }
        break;

    case 2:
        mask(x, y) = select(random_float() > ratio, scale, 0.0f);
        if (is_test) {
            forward(x, y) = layer->forward(x, y);
        } else {
            forward(x, y) = mask(x, y) * layer->forward(x, y);
        }
        break;

    case 3:
        mask(z, y, x) = select(random_float() > ratio, scale, 0.0f);
        if (is_test) {
            forward(z, y, x) = layer->forward(z, y, x);
        } else {
            forward(z, y, x) = mask(z, y, x) * layer->forward(z, y, x);
        }
        break;
    
    case 4:
        mask(n, z, y, x) = select(random_float() > ratio, scale, 0.0f);
        if (is_test) {
            forward(n, z, y, x) = layer->forward(n, z, y, x);
        } else {
            forward(n, z, y, x) = mask(n, z, y, x) * layer->forward(n, z, y, x);
        }
        break;

    default:
        // LOG_ASSERT(false) << "too many dimensions";
        break;
    }

    mask.compute_root();
}

void DropoutLayer::back_propagate(Func dout) {
    // LOG_ASSERT(dout.defined()) << "dout is not defined yet";
    // LOG_ASSERT(!is_test) << "BackProp should not be called during testing";
    if (!f_in_grad.defined()) {
        auto layer = input_layer;
        switch(layer->out_dims())
        {
        case 1:
            f_in_grad(x) = dout(x) * mask(x);
            break;

        case 2:
            f_in_grad(x, y) = dout(x, y) * mask(x, y);
            break;

        case 3:
            f_in_grad(z, y, x) = dout(z, y, x) * mask(z, y, x);
            break;
        
        case 4:
            f_in_grad(n, z, y, x) = dout(n, z, y, x) * mask(n, z, y, x);
            break;
        
        default:
            // LOG_ASSERT(false) << "too many dimensions";
            break;
        }
        layer->back_propagate(f_in_grad);
    }
}

int DropoutLayer::out_dims() const {
    return input_layer->out_dims();
}

int DropoutLayer::out_dim_size(int i) const {
    return input_layer->out_dim_size(i);
}