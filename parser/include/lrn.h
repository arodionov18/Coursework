#include "common.h"
#include "layer.h"
#include "Halide.h"

using namespace Halide;

class LRNLayer: public AbstractLayer {
public:
    Var x, y, z, n;

    int num_samples, in_ch, in_h, in_w;

    int size;
    float alpha, beta, bias;

    LRNLayer(const caffe2::OperatorDef& op, std::weak_ptr<AbstractLayer> input);

    void back_propagate(Func dout) override;

    int out_dims() const override;

    int out_dim_size(int i) const override;
};