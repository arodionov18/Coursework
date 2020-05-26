#include "layer.h"
#include "common.h"
#include "Halide.h"

using namespace Halide;

class AveragePoolingLayer: public AbstractLayer {
public:
    Var x, y, z, n;

    int num_samples, in_ch, in_h, in_w;

    int p_l, p_r, p_t, p_b, stride;
    int kernel;
    // scheduling params
    Var par;
    int vec_len = 8;

    AveragePoolingLayer(const caffe2::OperatorDef& op, std::shared_ptr<AbstractLayer> input, int schedule = 1);

    void back_propagate(Func dout) override;

    int out_dims() const;

    int out_dim_size(int i) const;
};