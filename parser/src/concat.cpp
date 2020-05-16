#include "concat.h"
#include <glog/logging.h>

using namespace Halide;

ConcatLayer::ConcatLayer(const std::vector<std::shared_ptr<AbstractLayer>>& inputs) : AbstractLayer(inputs[0]) {
    //default order NCHW, concat on axis C

    int offset = 0;
    // forward(x, y, z, n) = 0.0f; why?
    auto input = inputs[0];
    in_w = input->out_dim_size(3);
    in_h = input->out_dim_size(2);
    in_ch = input->out_dim_size(1);
    num_samples = input->out_dim_size(0);

    for (const auto& layer_w : inputs) {
        auto layer = layer_w;
        RDom r(0, layer->out_dim_size(1));
        forward(n, offset + r, y, x) = layer->forward(n, r, y, x);
        offset += layer->out_dim_size(2);
    }
}

void ConcatLayer::back_propagate(Func dout) {
    // LOG_ASSERT(false) << "NOT IMPLEMENTED";
}

int ConcatLayer::out_dims() const {
    return 4;
}

int ConcatLayer::out_dim_size(int i) const {
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
        // LOG_ASSERT(false) << "BAD INPUT";
        break;
    }
}