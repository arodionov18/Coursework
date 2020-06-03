#include "common.h"
#include "layer.h"
#include "Halide.h"

using namespace Halide;

class DropoutLayer: public AbstractLayer {
public:
    Var x, y, z, n;

    float ratio;
    int is_test;

    Func mask;

    DropoutLayer(const caffe2::OperatorDef& op, std::shared_ptr<AbstractLayer> input);

    void back_propagate(Func dout) override;

    int out_dims() const override;

    int out_dim_size(int i) const override;
};