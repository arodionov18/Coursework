#include "net.h"
//#include "include/io.h"
#include "layer.h"
#include "convolutional.h"
#include "layer_list.h"
#include <iostream>
#include <glog/logging.h>
#include <fstream>
#include "halide_image_io.h"
using namespace Halide::Tools;

namespace parser {
namespace net {

using std::string;
using std::unique_ptr;

std::pair<caffe2::TensorProto, caffe2::TensorProto> GetTensors(const caffe2::TensorProtos& tensors, const std::string& wname, const std::string& bname) {
    int count = 0;
    std::pair<caffe2::TensorProto, caffe2::TensorProto> result;
    std::cout << tensors.protos_size() << std::endl;
    std::cout << tensors.DebugString() << std::endl;
    std::cout << "#####" << std::endl;
    std::cout << tensors.SerializeAsString() << std::endl;
    for (int t_idx = 0; t_idx < tensors.protos_size() && count < 2; ++t_idx) {
        const caffe2::TensorProto& tensor = tensors.protos(t_idx);
        if (tensor.name() == wname) {
            std::cout << tensor.dims_size() << std::endl;
            std::cout << tensor.DebugString() << std::endl;
            //result.first = tensor;
            ++count;
        } else if (tensor.name() == bname) {
            //result.second = tensor;
            ++count;
        }
        std::cout << tensor.name() << std::endl;
    }
    return result;
}

void Net::ParseAndCreateNetwork(const caffe2::NetDef& network, const caffe2::TensorProtos& tensor) {
    LOG(INFO) << "Create network";
    std::cout << "Create network";
    std::cout << network.op_size() << std::endl;
    //network.PrintDebugString();
    for (int i = 0; i < network.op_size(); ++i) {
        auto layer_def = network.op(i);

        if (layer_def.type() == "Conv") {
            string input_name = layer_def.input(0); // name, w_name, b_name
            string output_name = layer_def.output(0);

            // LOG_ASSERT(net_outputs.find(input_name) == net_outputs.end());

            std::weak_ptr<AbstractLayer> input = net_outputs[input_name];

            string w_name = layer_def.input(1);
            string b_name = layer_def.input(2);
            std::cout << w_name << " " << b_name << std::endl;

            auto Wb = GetTensors(tensor, w_name, b_name);
            std::cout << "WW" << Wb.first.dims_size() << std::endl;
            net_tensors[w_name] = Wb.first;
            net_tensors[b_name] = Wb.second;
            net_outputs[output_name] = std::make_shared<ConvolutionalLayer>(net_tensors[w_name], net_tensors[b_name], layer_def, net_outputs[input_name]);
        } else if (layer_def.type() == "Relu") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);
            net_outputs[output_name] = std::make_shared<ReluLayer>(net_outputs[input_name]);
            
        } else if (layer_def.type() == "MaxPool") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);
            net_outputs[output_name] = std::make_shared<MaxPoolingLayer>(layer_def, net_outputs[input_name]);
            
        } else if (layer_def.type() == "AveragePool") {

        } else if (layer_def.type() == "Dropout") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<DropoutLayer>(layer_def, net_outputs[input_name]);
        } else if (layer_def.type() == "FC") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            string w_name = layer_def.input(1);
            string b_name = layer_def.input(2);

            auto Wb = GetTensors(tensor, w_name, b_name);
            net_tensors[w_name] = Wb.first;
            net_tensors[b_name] = Wb.second;
            net_outputs[output_name] = std::make_shared<FCLayer>(net_tensors[w_name], net_tensors[b_name], net_outputs[input_name]);
        } else if (layer_def.type() == "LRN") {

        } else if (layer_def.type() == "DepthConcat") {

        } else if (layer_def.type() == "Softmax") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<SoftmaxLayer>(net_outputs[input_name]);
        } else if (layer_def.type() == "Flatten") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<FlattenLayer>(net_outputs[input_name]);
        } else if (layer_def.type() == "Concat") {
            std::vector<std::weak_ptr<AbstractLayer>> inputs;
            for (int l_idx = 0; l_idx < layer_def.input_size(); ++l_idx) {
                auto iter = net_outputs.find(layer_def.input(l_idx));
                // LOG_ASSERT(iter != net_outputs.end()) << "input layer is not defined yet";
                inputs.push_back(iter->second);
            }
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<ConcatLayer>(inputs);
        } else {
            // LOG_ASSERT(false) << "Unimplemented layer" << std::endl;
            return;
        }
    }
    LOG(INFO) << "Network build finished with success";
}


Net::Net(const string& network_description, bool binary) {
    auto flags = std::ios::in;
    if (binary) {
        flags = flags | std::ios::binary;
    }
    std::fstream net_input(network_description, flags);

    network_def.ParseFromIstream(&net_input);
}

void Net::LoadWeights(const string& network_weights, bool binary) {
    auto flags = std::ios::in;
    if (binary) {
        flags = flags | std::ios::binary;
    }
    std::fstream input_weights(network_weights, flags);
    //caffe2::ExternalDataProto data;
    //data.ParseFromIstream(&input_weights);
    //std::cout << data.source_type() << std::endl;
    weights.ParseFromIstream(&input_weights);
    std::cout << "Weigths " << weights.protos_size() << std::endl;
    weights.PrintDebugString();
}

void Net::ReadImage(const std::string& image_path) {
    ImageInfo params{0,0,0,0};
    Halide::Buffer<float> image = load_and_convert_image(image_path);
    
    net_outputs["data"] = std::make_shared<DataLayer>(image, params);
}

void Net::Init() {
    ParseAndCreateNetwork(network_def, weights);
    for (auto elem : net_outputs) {
        std::cout << elem.first << std::endl;
    }
    //std::cout << net_outputs["prob"]->out_dims();
    std::cout << "GOOD" << std::endl;
}

} // net
} // parser