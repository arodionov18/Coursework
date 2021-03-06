#include "concat.h"
#include <glog/logging.h>

using namespace Halide;

ConcatLayer::ConcatLayer(const std::vector<std::shared_ptr<AbstractLayer>>& inputs) : AbstractLayer(inputs[0]) {
    //default order NCHW, concat on axis C

    // forward(x, y, z, n) = 0.0f; why?
    auto input = inputs[0];
    in_w = input->out_dim_size(3);
    in_h = input->out_dim_size(2);
    in_ch = 0;
    num_samples = input->out_dim_size(0);

    forward(n, z, y, x) = 0.0f;

    int offset = 0;
    for (const auto& layer_w : inputs) {
        in_ch += layer_w->out_dim_size(1);
        RDom r(0, layer_w->out_dim_size(1));
        forward(n, offset + r.x, y, x) += layer_w->forward(n, r.x, y, x);
        offset += layer_w->out_dim_size(1);
    }

    forward.compute_root();
    forward.update().parallel(x);
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