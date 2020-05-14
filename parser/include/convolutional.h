#pragma once

#include "layer.h"

class ConvolutionalLayer : public AbstractLayer {
public:
    Halide::Var x, y, z, n;

    int num_samples, in_ch, in_h, in_w;

    int num_f, f_h, f_w, pad, stride;

    float reg;

    Halide::Var y_t, z_t, par; // scheduling

    int o_block_size;
    int y_block_size;
    int vec_len;

    Halide::Func forward_clamp;

    ConvolutionalLayer(const caffe2::TensorProto& w, const caffe2::TensorProto& b,
     const caffe2::OperatorDef& op, std::weak_ptr<AbstractLayer> input, int schedule = true);

    void back_propagate(Halide::Func dout) override;

    int out_dims() const override;

    int out_dim_size(int i) const override;
};