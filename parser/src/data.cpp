#include "data.h"
#include "halide_image_io.h"

using namespace Halide;
// img in CHW format
DataLayer::DataLayer(Buffer<float> img, const ImageInfo& info): AbstractLayer(std::shared_ptr<AbstractLayer>()) {

    //Func data = BoundaryConditions::constant_exterior(img, 0.f, 0, in_w, in_h);
    //forward.trace_stores();
    img.add_dimension();
    Buffer<float> rescaled_image = rescale(convert_to_nchw(img, info), info);
    ImageInfo rescaled_info;
    rescaled_info.num_samples = rescaled_image.extent(0);
    rescaled_info.channels = rescaled_image.extent(1);
    rescaled_info.h = rescaled_image.extent(2);
    rescaled_info.w = rescaled_image.extent(3);

    std::cout << "rescaled image info: " << rescaled_info.num_samples << " " << rescaled_info.channels << " " << rescaled_info.h << " " << rescaled_info.w << std::endl;

    Buffer<float> cropped_image = crop_center(convert_to_bgr(rescaled_image, rescaled_info), rescaled_info);

    /*// ##########################
    std::cout << "INPUT IMG\n";
    for (size_t i = 0; i < kImageSize; ++i) {
        for (size_t j = 0; j < kImageSize; ++j) {
            std::cout << cast<int>(std::max(cropped_image(0, 1, i, j), cropped_image(0, 0, i, j))) << " ";
        }
        std::cout << std::endl;
    }
    // ##########################*/

    in_h = kImageSize;
    in_w = kImageSize;
    in_ch = info.channels;
    num_samples = info.num_samples;

    Func data = BoundaryConditions::constant_exterior(cropped_image, 0.0f, 0, num_samples, 0, in_ch, 0, in_h, 0, in_w);
    forward(n, z, y, x) = 255.0f * data(n, z, y, x) - 128.0f;
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

Halide::Buffer<float> transform(const Halide::Buffer<float>& image, const ImageInfo& info, int new_height, int new_width) {
    Halide::Buffer<float> new_image(info.num_samples, info.channels, new_height, new_width);
    Var x, y, c, n;
    float w_ratio = info.w / static_cast<float>(new_width);
    float h_ratio = info.h / static_cast<float>(new_height);
    new_image(n, c, y, x) = image(n, c, Halide::cast<int>(y * h_ratio), Halide::cast<int>(x * w_ratio));
    return new_image;
}

Halide::Buffer<float> DataLayer::rescale(const Halide::Buffer<float>& image, const ImageInfo& info) {
    float aspect = info.w / static_cast<float>(info.h);
    if (aspect > 1) { // wide image
        return transform(image, info, aspect * kImageSize, kImageSize);
    } else if (aspect < 1) { // tall image
        return transform(image, info, kImageSize, kImageSize / aspect);
    } else {
        return transform(image, info, kImageSize, kImageSize);
    }
}

Halide::Buffer<float> DataLayer::crop_center(const Halide::Buffer<float>& image, const ImageInfo& info) {
    int start_x = info.w / 2 - kImageSize / 2;
    int start_y = info.h / 2 - kImageSize / 2;
    Buffer<float> result(info.num_samples, info.channels, kImageSize, kImageSize);
    Var c, n;
    RDom r(0, kImageSize, 0, kImageSize);
    result(n, c, r.y, r.x) = image(n, c, start_y + r.y, start_x + r.x);
    return result;
}

Halide::Buffer<float> DataLayer::convert_to_nchw(const Halide::Buffer<float>& image, const ImageInfo& info) {
    Halide::Buffer<float> result(info.num_samples, info.channels, info.h, info.w);
    Var x, y, c, n;
    result(n, c, y, x) = image(x, y, c, n); // normalization
    return result;
}

Halide::Buffer<float> DataLayer::convert_to_bgr(const Halide::Buffer<float>& image, const ImageInfo& info) {
    Halide::Buffer<float> result(info.num_samples, info.channels, info.h, info.w);
    Var x, y, c, n;
    result(n, c, y, x) = image(n, info.channels - c - 1, y, x);
    return result;
}