#include "common.h"
#include "layer.h"
#include "Halide.h"

using namespace Halide;

class SoftmaxLayer: public AbstractLayer {
public:
    Var in_dim, n;

    int num_classes, num_samples;

    SoftmaxLayer(std::shared_ptr<AbstractLayer> input, int schedule = 1);

    Func loss(Func labels);

    void back_propagate(Func labels) override;

    int out_dims() const override;

    int out_dim_size(int i) const override;
};