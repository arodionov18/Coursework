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
    //std::cerr << tensors.protos_size() << std::endl;
    for (int t_idx = 0; t_idx < tensors.protos_size() && count < 2; ++t_idx) {
        const caffe2::TensorProto& tensor = tensors.protos(t_idx);
        if (tensor.name() == wname) {
            result.first = tensor;
            ++count;
        } else if (tensor.name() == bname) {
            result.second = tensor;
            ++count;
        }
        //std::cerr << tensor.name() << std::endl;
    }
    assert(count == 2);
    return result;
}

void Net::ParseAndCreateNetwork(const caffe2::NetDef& network, const caffe2::TensorProtos& tensor) {
    LOG(INFO) << "Create network";
    // std::cerr << "Create network" << std::endl;
    // std::cerr << network.op_size() << std::endl;
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
            //std::cerr << w_name << " " << b_name << std::endl;

            auto Wb = GetTensors(tensor, w_name, b_name);
            // std::cerr << "WW" << Wb.first.dims_size() << std::endl;
            net_tensors[w_name] = Wb.first;
            net_tensors[b_name] = Wb.second;
            net_outputs[output_name] = std::make_shared<ConvolutionalLayer>(net_tensors[w_name], net_tensors[b_name], layer_def, net_outputs[input_name]);
            /*std::cerr << "##CONV## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cerr << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cerr << input_name << " " << output_name << std::endl;
            std::cerr << "##EE##" << std::endl;*/
        } else if (layer_def.type() == "Relu") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);
            net_outputs[output_name] = std::make_shared<ReluLayer>(net_outputs[input_name]);
            /*std::cerr << "##RELU## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cerr << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cerr << "##EE##" << std::endl;
            */
        } else if (layer_def.type() == "MaxPool") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);
            net_outputs[output_name] = std::make_shared<MaxPoolingLayer>(layer_def, net_outputs[input_name]);
            /*std::cerr << "##MAX## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cerr << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cerr << "##EE##" << std::endl;*/
        } else if (layer_def.type() == "AveragePool") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);
            net_outputs[output_name] = std::make_shared<AveragePoolingLayer>(layer_def, net_outputs[input_name]);
            /*std::cerr << "##AVG## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cerr << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cerr << "##EE##" << std::endl;*/
        } else if (layer_def.type() == "Dropout") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<DropoutLayer>(layer_def, net_outputs[input_name]);
            /*std::cerr << "##DROP## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cerr << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cerr << "##EE##" << std::endl;*/
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
            net_outputs[output_name] = std::make_shared<FCLayer>(layer_def, net_tensors[w_name], net_tensors[b_name], net_outputs[flatten_name]);
            /*std::cerr << "##FC## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cerr << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cerr << "##EE##" << std::endl;*/       
        } else if (layer_def.type() == "LRN") {
            string input_name = layer_def.input(0);
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<LRNLayer>(layer_def, net_outputs[input_name]);
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
            /*std::cerr << "##FLAT## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cerr << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cerr << "##EE##" << std::endl;*/
        } else if (layer_def.type() == "Concat") {
            std::vector<std::shared_ptr<AbstractLayer>> inputs;
            for (int l_idx = 0; l_idx < layer_def.input_size(); ++l_idx) {
                auto iter = net_outputs.find(layer_def.input(l_idx));
                // LOG_ASSERT(iter != net_outputs.end()) << "input layer is not defined yet";
                inputs.push_back(iter->second);
            }
            string output_name = layer_def.output(0);

            net_outputs[output_name] = std::make_shared<ConcatLayer>(inputs);
            /*std::cerr << "##CONC## " <<  net_outputs[output_name]->out_dims() << std::endl;
            for (int tt = 0; tt < net_outputs[output_name]->out_dims(); ++tt) {
                std::cerr << "\t" << net_outputs[output_name]->out_dim_size(tt) << std::endl;
            }
            std::cerr << "##EE##" << std::endl;*/
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
    def.ParseFromIstream(&input_weights);
    //float max_weight = 0;
    //weights.InitAsDefaultInstance();
    for (int op_idx = 0; op_idx < def.op_size(); ++op_idx) {
        auto operation_def = def.op(op_idx);
        caffe2::TensorProto* tensor = weights.add_protos();
        //tensor->InitAsDefaultInstance();
        tensor->set_name(operation_def.output(0));
        //tensor->clear_dims();
        for (int i = 0; i < operation_def.arg(0).ints_size(); ++i) {
            tensor->add_dims(operation_def.arg(0).ints(i));
        }
        //tensor->clear_float_data();
        for (int i = 0; i < operation_def.arg(1).floats_size(); ++i) {
            // max_weight = std::max(max_weight, abs(operation_def.arg(1).floats(i)));
            tensor->add_float_data(operation_def.arg(1).floats(i));
        }
    }
    // std::cerr << "Maximum weight is: " << max_weight << std::endl;
}

void Net::ReadImage(const std::string& image_path) {
    ImageInfo params;//{190, 190, 3, 1};// ~~190 {64,128,3,1};
    Halide::Buffer<uint8_t> image = load_image(image_path);
    /*std::cerr << "INPUT IMG\n";
    for (size_t i = 0; i < image.extent(0); ++i) {
        for (size_t j = 0; j < image.extent(1); ++j) {
            std::cerr << "[" << image(i,j,0) << "," << image(i,j,1) << "," << image(i,j,2) << "]\t";
        }
        std::cerr << std::endl;
    }*/
    params.num_samples = 1;
    params.w = image.extent(0);
    params.h = image.extent(1);
    params.channels = image.extent(2);
    Halide::Buffer<float> fimage(params.w, params.h, params.channels);
    // std::cerr << "inp_image" << std::endl;
    for (size_t i = 0; i < params.w; ++i) {
        for (size_t j = 0; j < params.h; ++j) {
            for (size_t c = 0; c < params.channels; ++c) {
                fimage(i, j, c) = 1.0f * image(i, j, c);
                //std::cerr << "[" << fimage(i, j, c) << ", " << image(i, j, c) << "]" <<  " ";
            }
        }
    }
    //std::cerr << std::endl;
    
    net_outputs["data"] = std::make_shared<DataLayer>(fimage, params);
}

void Net::Init() {
    ParseAndCreateNetwork(network_def, weights);
    /*for (auto elem : net_outputs) {
        std::cerr << elem.first << std::endl;
    }*/
    //std::cerr << net_outputs["prob"]->out_dims();
    //std::cerr << "GOOD" << std::endl;
}

std::pair<float, int> Net::GetResults() {
    int num_samples, num_classes;
    num_samples = net_outputs["prob"]->out_dim_size(0);
    num_classes = net_outputs["prob"]->out_dim_size(1);
    Halide::Buffer<float> results = net_outputs["prob"]->forward.realize(num_samples, num_classes);

    /*Halide::Buffer<float> tmp_results = net_outputs["prob"]->input_layer->forward.realize(1, 1000);
    for (size_t i = 0; i < 1000; ++i) {
        std::cerr << tmp_results(0,i) << ", ";
    }
    std::cerr << std::endl;*/
    float max_prob = 0;
    int max_prob_class = -1;
    float min_prob = 1;
    int min_prob_class = -1;
    //std::cerr << "Probs: " << std::endl;
    for (size_t i = 0; i < num_classes; ++i) {
        //std::cerr << results(0, i) << std::endl;
        if (results(0, i) > max_prob) {
            max_prob_class = i;
            max_prob = results(0, i);
        }
        if (results(0, i) < min_prob) {
            min_prob_class = i;
            min_prob = results(0, i);
        }
    }
    //std::cerr << "Min prob: " << min_prob << ", class: " << min_prob_class << std::endl;*/
    return {max_prob, max_prob_class};
}

} // net
} // parser