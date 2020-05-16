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
    //std::cout << tensors.protos_size() << std::endl;
    for (int t_idx = 0; t_idx < tensors.protos_size() && count < 2; ++t_idx) {
        const caffe2::TensorProto& tensor = tensors.protos(t_idx);
        if (tensor.name() == wname) {
            result.first = tensor;
            ++count;
        } else if (tensor.name() == bname) {
            result.second = tensor;
            ++count;
        }
        //std::cout << tensor.name() << std::endl;
    }
    assert(count == 2);
    return result;
}

void Net::ParseAndCreateNetwork(const caffe2::NetDef& network, const caffe2::TensorProtos& tensor) {
    LOG(INFO) << "Create network";
    std::cout << "Create network" << std::endl;
    std::cout << network.op_size() << std::endl;
    //network.PrintDebugString();
    for (int i = 0; i < network.op_size(); ++i) {
        auto layer_def = network.op(i);

        if (layer_def.type() == "Conv") {
            string input_name = layer_def.input(0); // name, w_name, b_name
            string output_name = layer_def.output(0);

            // LOG_ASSERT(net_outputs.find(input_name) == net_outputs.end());

            std::shared_ptr<AbstractLayer> input = net_outputs[input_name];

            string w_name = layer_def.input(1);
            string b_name = layer_def.input(2);
            std::cout << w_name << " " << b_name << std::endl;

            auto Wb = GetTensors(tensor, w_name, b_name);
            std::cout << "WW" << Wb.first.dims_size() << std::endl;
            net_tensors[w_name] = Wb.first;
            net_tensors[b_name] = Wb.second;
            net_outputs[output_name] = std::make_shared<ConvolutionalLayer>(net_tensors[w_name], net_tensors[b_name], layer_def, net_outputs[input_name]);
            std::cout << "##NN## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cout << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cout << "##EE##" << std::endl;
        } else if (layer_def.type() == "Relu") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);
            net_outputs[output_name] = std::make_shared<ReluLayer>(net_outputs[input_name]);
            std::cout << "##NN## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cout << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cout << "##EE##" << std::endl;
            
        } else if (layer_def.type() == "MaxPool") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);
            net_outputs[output_name] = std::make_shared<MaxPoolingLayer>(layer_def, net_outputs[input_name]);
            std::cout << "##NN## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cout << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cout << "##EE##" << std::endl;
        } else if (layer_def.type() == "AveragePool") {

        } else if (layer_def.type() == "Dropout") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<DropoutLayer>(layer_def, net_outputs[input_name]);
            std::cout << "##NN## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cout << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cout << "##EE##" << std::endl;
        } else if (layer_def.type() == "FC") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            string flatten_name = input_name + "_FLATTEN";
            net_outputs[flatten_name] = std::make_shared<FlattenLayer>(net_outputs[input_name]);

            string w_name = layer_def.input(1);
            string b_name = layer_def.input(2);

            auto Wb = GetTensors(tensor, w_name, b_name);

            net_tensors[w_name] = Wb.first;
            net_tensors[b_name] = Wb.second;
            net_outputs[output_name] = std::make_shared<FCLayer>(net_tensors[w_name], net_tensors[b_name], net_outputs[flatten_name]);
            std::cout << "##NN## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cout << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cout << "##EE##" << std::endl;       
        } else if (layer_def.type() == "LRN") {

        } else if (layer_def.type() == "DepthConcat") {

        } else if (layer_def.type() == "Softmax") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            string flatten_name = input_name + "_FLATTEN";
            net_outputs[flatten_name] = std::make_shared<FlattenLayer>(net_outputs[input_name]);

            net_outputs[output_name] = std::make_shared<SoftmaxLayer>(net_outputs[flatten_name]);
        } else if (layer_def.type() == "Flatten") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<FlattenLayer>(net_outputs[input_name]);
            std::cout << "##NN## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cout << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cout << "##EE##" << std::endl;
        } else if (layer_def.type() == "Concat") {
            std::vector<std::shared_ptr<AbstractLayer>> inputs;
            for (int l_idx = 0; l_idx < layer_def.input_size(); ++l_idx) {
                auto iter = net_outputs.find(layer_def.input(l_idx));
                // LOG_ASSERT(iter != net_outputs.end()) << "input layer is not defined yet";
                inputs.push_back(iter->second);
            }
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<ConcatLayer>(inputs);
            std::cout << "##NN## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cout << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cout << "##EE##" << std::endl;
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
    caffe2::NetDef def;
    def.ParsePartialFromIstream(&input_weights);
    weights.InitAsDefaultInstance();
    for (int op_idx = 0; op_idx < def.op_size(); ++op_idx) {
        auto operation_def = def.op(op_idx);
        caffe2::TensorProto* tensor = weights.add_protos();
        tensor->InitAsDefaultInstance();
        tensor->set_name(operation_def.output(0));
        tensor->clear_dims();
        for (int i = 0; i < operation_def.arg(0).ints_size(); ++i) {
            tensor->add_dims(operation_def.arg(0).ints(i));
        }
        tensor->clear_float_data();
        for (int i = 0; i < operation_def.arg(1).floats_size(); ++i) {
            tensor->add_float_data(operation_def.arg(1).floats(i));
        }
    }
}

void Net::ReadImage(const std::string& image_path) {
    ImageInfo params{190, 190, 3, 1};// ~~190 {64,128,3,1};
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