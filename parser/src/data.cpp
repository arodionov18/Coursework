#include "data.h"

using namespace Halide;

DataLayer::DataLayer(Buffer<float> img, const ImageInfo& info): AbstractLayer(std::shared_ptr<AbstractLayer>()) {
    in_h = info.h;
    in_w = info.w;
    in_ch = info.channels;
    num_samples = info.num_samples;

    //Func data = BoundaryConditions::constant_exterior(img, 0.f, 0, in_w, in_h);
    img.add_dimension();
    Func data = BoundaryConditions::constant_exterior(img, 0.f, 0, in_w, 0, in_h, 0, in_ch, 0, 1);
    forward(x, y, z, n) = data(x, y, z, n);
}

void DataLayer::back_propagate(Func dout) {
    return;
}

int DataLayer::out_dims() const {
    return 4;
}

int DataLayer::out_dim_size(int i) const {
    switch (i)
    {
    case 3:
        return in_w;
    case 2:
        return in_h;
    case 1:
        return in_ch;
    case 0:
        return num_samples;
    default:
        // LOG_ASSERT(false) << "BAD INPUT";
        break;
    }
}