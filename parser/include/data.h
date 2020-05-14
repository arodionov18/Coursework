#pragma once

#include "common.h"
#include "Halide.h"
#include "layer.h"

using namespace Halide;

struct ImageInfo {
    int h, w, channels, num_samples;
};

class DataLayer: public AbstractLayer {
public:
    int in_h, in_w, in_ch, num_samples;

    Var x, y, z, n;

    DataLayer(Buffer<float> img, const ImageInfo& info);

    void back_propagate(Func dout) override;

    int out_dims() const;

    int out_dim_size(int i) const;

};