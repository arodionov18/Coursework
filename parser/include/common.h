#pragma once

#include "caffe2.pb.h"
#include "Halide.h"
#include <glog/logging.h>

Halide::Buffer<float> LoadBufferFromTensor(const caffe2::TensorProto& tensor);
