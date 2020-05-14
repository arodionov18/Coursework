#include "relu.h"

using namespace Halide;

ReluLayer::ReluLayer(std::weak_ptr<AbstractLayer> input) : AbstractLayer(input) {
    auto layer = input_layer.lock();
    switch (layer->out_dims())
    {
    case 1:
        forward(x) = max(0, layer->forward(x));
        break;
    case 2:
        forward(x, y) = max(0, layer->forward(x, y));
    case 3:
        forward(x, y, z) = max(0, layer->forward(x, y, z));
    case 4:
        forward(x, y, z, n) = max(0, layer->forward(x, y, z, n));
    default:
        // LOG_ASSERT(false) << "Bad input dimensions";
        break;
    }
}

void ReluLayer::back_propagate(Func dout) {
    // LOG_ASSERT(dout.defined()) << "dout is not defined yet";
    if (!f_in_grad.defined()) {
        auto layer = input_layer.lock();
        switch (layer->out_dims())
        {
        case 1:
            f_in_grad(x) = dout(x) * select(layer->forward(x) > 0, 1, 0);
            break;
        case 2:
            f_in_grad(x, y) = dout(x, y) * select(layer->forward(x, y) > 0, 1, 0);
            break;
        case 3:
            f_in_grad(x, y, z) = dout(x, y, z) * select(layer->forward(x, y, z) > 0, 1, 0);
            break;
        case 4:
            f_in_grad(x, y, z, n) = dout(x, y, z, n) * select(layer->forward(x, y, z, n) > 0, 1, 0);
            break;
        
        default:
            // LOG_ASSERT(false) << "Bad input dimensions";
            break;
        }
        layer->back_propagate(f_in_grad);
    }
}

int ReluLayer::out_dims() const {
    return input_layer.lock()->out_dims();
}

int ReluLayer::out_dim_size(int i) const {
    return input_layer.lock()->out_dim_size(i);
}