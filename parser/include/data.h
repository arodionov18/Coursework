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

private:
    const int kImageSize = 224; // Optimal size for pretrained vgg19

    Halide::Buffer<float> rescale(const Halide::Buffer<float>& image, const ImageInfo& info);

    Halide::Buffer<float> crop_center(const Halide::Buffer<float>& image, const ImageInfo& info);

    Halide::Buffer<float> convert_to_nchw(const Halide::Buffer<float>& image, const ImageInfo& info);

    Halide::Buffer<float> convert_to_bgr(const Halide::Buffer<float>& image, const ImageInfo& info);
};