#include "data.h"
#include "halide_image_io.h"

using namespace Halide;

template <class T>
void PrintDebugImage(const Buffer<T>& img, int i) {
    std::cerr << "INPUT_IMG_DEBUG: " << i << std::endl;
    if (img.dimensions() == 4) {
        for (size_t i = 0; i < img.extent(0); ++i)
        {
            for (size_t j = 0; j < img.extent(1); ++j)
            {
                for (size_t k = 0; k < img.extent(2); ++k)
                {
                    for (size_t m = 0; m < img.extent(3); ++m)
                    {
                        std::cerr << img(i, j, k, m) << " ";
                    }
                    std::cerr << std::endl;
                }
                std::cerr << "##" << std::endl;
            }
        }
    } else if (img.dimensions() == 3) {
        for (size_t i = 0; i < img.extent(2); ++i)
        {
            for (size_t j = 0; j < img.extent(1); ++j)
            {
                for (size_t k = 0; k < img.extent(2); ++k)
                {
                    std::cerr << img(k, j, i) << " ";
                }
                std::cerr << std::endl;
            }
            std::cerr << "##" << std::endl;
        }
    }
    std::cerr << std::endl << std::endl;
}

void DataLayer::LoadNewImage(Buffer<float> img, const ImageInfo& info) {
    img.add_dimension();
    Buffer<float> rescaled_image = rescale(convert_to_nchw(img, info), info);
    ImageInfo rescaled_info;
    rescaled_info.num_samples = rescaled_image.extent(0);
    rescaled_info.channels = rescaled_image.extent(1);
    rescaled_info.h = rescaled_image.extent(2);
    rescaled_info.w = rescaled_image.extent(3);

    // PrintDebugImage(rescaled_image, 11);

    std::cerr << "rescaled image info: " << rescaled_info.num_samples << " " << rescaled_info.channels << " " << rescaled_info.h << " " << rescaled_info.w << std::endl;

    Buffer<float> cropped_image = crop_center(convert_to_bgr(rescaled_image, rescaled_info), rescaled_info);

    //PrintDebugImage(cropped_image, 12);

    in_h = kImageSize;
    in_w = kImageSize;
    in_ch = info.channels;
    num_samples = info.num_samples;

    Func data = BoundaryConditions::constant_exterior(cropped_image, 0.0f, 0, num_samples, 0, in_ch, 0, in_h, 0, in_w);
    forward(n, z, y, x) = data(n, z, y, x) - 128.0f;
    forward.compute_root().parallel(z);
}

// img in CHW format
DataLayer::DataLayer(Buffer<float> img, const ImageInfo& info): AbstractLayer(std::shared_ptr<AbstractLayer>()) {

    //Func data = BoundaryConditions::constant_exterior(img, 0.f, 0, in_w, in_h);
    //forward.trace_stores();
    //PrintDebugImage(img, 10);
    LoadNewImage(img, info);
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
    float w_ratio = info.w / static_cast<float>(new_width);
    float h_ratio = info.h / static_cast<float>(new_height);
    for (int i = 0; i < info.num_samples; ++i) {
        for (int j = 0; j < info.channels; ++j) {
            for (int k = 0; k < new_height; ++k) {
                for (int m = 0; m < new_width; ++m) {
                    new_image(i, j, k, m) = image(i, j, static_cast<int>(k * h_ratio), static_cast<int>(m * w_ratio));
                }
            }
        }
    }
    
    return new_image;
}

Halide::Buffer<float> DataLayer::rescale(const Halide::Buffer<float>& image, const ImageInfo& info) {
    // PrintDebugImage(image, 20);
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
    for (int i = 0; i < info.num_samples; ++i) {
        for (int j = 0; j < info.channels; ++j) {
            for (int k = 0; k < kImageSize; ++k) {
                for (int m = 0; m < kImageSize; ++m) {
                    result(i, j, k, m) = image(i, j, start_y + k, start_x + m);
                }
            }
        }
    }
    return result;
}

Halide::Buffer<float> DataLayer::convert_to_nchw(const Halide::Buffer<float>& image, const ImageInfo& info) {
    Halide::Buffer<float> result(info.num_samples, info.channels, info.h, info.w);
    for (int i = 0; i < info.num_samples; ++i) {
        for (int j = 0; j < info.channels; ++j) {
            for (int k = 0; k < info.h; ++k) {
                for (int m = 0; m < info.w; ++m) {
                    result(i, j, k, m) = image(m, k, j, i);
                }
            }
        }
    }
    return result;
}

Halide::Buffer<float> DataLayer::convert_to_bgr(const Halide::Buffer<float>& image, const ImageInfo& info) {
    Halide::Buffer<float> result(info.num_samples, info.channels, info.h, info.w);
    for (int i = 0; i < info.num_samples; ++i) {
        for (int j = 0; j < info.channels; ++j) {
            for (int k = 0; k < info.h; ++k) {
                for (int m = 0; m < info.w; ++m) {
                    result(i, j, k, m) = image(i, info.channels - j - 1, k, m);
                }
            }
        }
    }
    return result;
}