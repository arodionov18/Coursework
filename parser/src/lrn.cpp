#include "lrn.h"

using namespace Halide;

LRNLayer::LRNLayer(const caffe2::OperatorDef& op, std::shared_ptr<AbstractLayer> input) : AbstractLayer(input) {
    auto layer = input_layer;
    
    Func clamped = BoundaryConditions::constant_exterior(layer->forward, 0.0f, 0, layer->out_dim_size(0), 0, layer->out_dim_size(1), 0, layer->out_dim_size(2));

    Func activation;
    Func normilizer;

    in_w = layer->out_dim_size(0);
    in_h = layer->out_dim_size(1);
    in_ch = layer->out_dim_size(2);
    num_samples = layer->out_dim_size(3);

    
}