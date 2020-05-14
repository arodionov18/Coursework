#pragma once

#include "layer.h"
#include "common.h"
#include "Halide.h"

using namespace Halide;

class FlattenLayer: public AbstractLayer {
public:
    Var x, y, z, n;

    int out_width;
    int num_samples;

    FlattenLayer(std::weak_ptr<AbstractLayer> input);
    
    void back_propagate(Func dout) override;

    int out_dims() const override;

    int out_dim_size(int i) const override;
};