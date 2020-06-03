#pragma once

#include "common.h"
#include "layer.h"

#include <vector>
#include <memory>
#include <map>

namespace parser {
namespace net {

using std::string;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::map;

class Net {
public:
    explicit Net(const string& network_description, bool binary);

    void LoadWeights(const string& network_weights, bool binary);

    void Init();

    void ReadImage(const string& image_path);

    string name;

    Halide::Buffer<float> Forward(float& loss);

    //const vector<Blob<Dtype>*>& Forward(DType* loss = NULL);

    std::pair<float, int> GetResults();

    void Backward();

    void Reshape();
    void Update();

protected:
    string name_;
    map<string, shared_ptr<AbstractLayer>> net_outputs;
    map<string, caffe2::TensorProto> net_tensors;

    caffe2::NetDef network_def;
    caffe2::TensorProtos weights;

private:
    void ParseAndCreateNetwork(const caffe2::NetDef& network, const caffe2::TensorProtos& tensor);
};

} // net
} // parser