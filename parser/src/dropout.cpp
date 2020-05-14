#include "dropout.h"

using namespace Halide;

DropoutLayer::DropoutLayer(const caffe2::OperatorDef& op, std::weak_ptr<AbstractLayer> input) : AbstractLayer(input) {
    auto layer = input_layer.lock();

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
        mask(x, y, z) = select(random_float() > ratio, scale, 0.0f);
        if (is_test) {
            forward(x, y, z) = layer->forward(x, y, z);
        } else {
            forward(x, y, z) = mask(x, y, z) * layer->forward(x, y, z);
        }
        break;
    
    case 4:
        mask(x, y, z, n) = select(random_float() > ratio, scale, 0.0f);
        if (is_test) {
            forward(x, y, z, n) = layer->forward(x, y, z, n);
        } else {
            forward(x, y, z, n) = mask(x, y, z, n) * layer->forward(x, y, z, n);
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
        auto layer = input_layer.lock();
        switch(layer->out_dims())
        {
        case 1:
            f_in_grad(x) = dout(x) * mask(x);
            break;

        case 2:
            f_in_grad(x, y) = dout(x, y) * mask(x, y);
            break;

        case 3:
            f_in_grad(x, y, z) = dout(x, y, z) * mask(x, y, z);
            break;
        
        case 4:
            f_in_grad(x, y, z, n) = dout(x, y, z, n) * mask(x, y, z, n);
            break;
        
        default:
            // LOG_ASSERT(false) << "too many dimensions";
            break;
        }
        layer->back_propagate(f_in_grad);
    }
}

int DropoutLayer::out_dims() const {
    return input_layer.lock()->out_dims();
}

int DropoutLayer::out_dim_size(int i) const {
    return input_layer.lock()->out_dim_size(i);
}