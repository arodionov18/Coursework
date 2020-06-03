#pragma once

#include "common.h"
#include "layer.h"
#include "Halide.h"

using namespace Halide;

class ConcatLayer: public AbstractLayer {
public:
    Var x, y, z, n;
    int num_samples, in_ch, in_h, in_w;

    ConcatLayer(const std::vector<std::shared_ptr<AbstractLayer>>& inputs);

    void back_propagate(Func dout) override;

    int out_dims() const override;

    int out_dim_size(int i) const override;


private:

};