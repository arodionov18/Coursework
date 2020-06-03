#pragma once

#include "layer.h"
#include "common.h"
#include "Halide.h"

using namespace Halide;

class ReluLayer: public AbstractLayer {
public:
    Var x, y, z, n;

    ReluLayer(std::shared_ptr<AbstractLayer> input);

    void back_propagate(Func dout) override;

    int out_dims() const;

    int out_dim_size(int i) const;
};