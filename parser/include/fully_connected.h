#pragma once

#include "common.h"
#include "layer.h"


class FCLayer : public AbstractLayer {
public:
    Halide::Var x, y, z, n;

    FCLayer(const caffe2::TensorProto& w, const caffe2::TensorProto& b, std::weak_ptr<AbstractLayer> input, int schedule = true);

    void back_propagate(Halide::Func dout) override;

    int out_dims() const override;

    int out_dim_size(int i) const override;

private:

};