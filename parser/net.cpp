#include "net.h"
#include "io.h"
#include "layer.h"
#include <iostream>

namespace parser {
namespace net {

unique_ptr<AbstractLayer> ParseAndCreateLayer(NetParameter* parameters) {
    CHECK(!parameters.has_type()) << "Wrong layer";
    switch (parameters.type()) {
        case "DATA":
            auto params = GetDataParams(parameters);
            break;

        case "CONVOLUTIONAL":
            auto params = GetConvolutionalParams(parameters);
            break;
        
        case "RELU":
            auto params = GetReluParams(parameters);
            break;

        case "POOLING":
            auto params = GetPoolingParams(parameters);
            break;
        
        case "INNER_PRODUCT":
            auto params = GetInnerProductParams(parameters);
            break;
        
        case "ACCURACY":
            auto params = GetAccuracyParams(parameters);
            break;

        case "DROPOUT":
            auto params = GetDropoutParams(parameters);
            break;
        
        case "SOFTMAX_LOSS":
            auto params = GetSoftMaxParams(parameters);
            break;

        
        default:
            std::cerr << "ERROR: UNKNOWN LAYER TYPE " << parameters.type() << " WITH NAME " << parameters.name() << std::endl;
            break;
    }
}

Net::Net(const NetParameter& parameters) {
    init(parameters);
}

Net::Net(const string& filename, bool binary) {
    NetParameter parameters;
    // TODO: ReadNetParametersFromFile(filename, binary);
    init(parameters);
}

void Net::init(const NetParameter& parameters) {
    this->name = parameters.name();
    // TODO
    for (size_t i = 0; i < parameters.layer_size(); ++i) {
        auto& layer = param.layer(i);
        this->layers.push_back(ParseAndCreateLayer(layer));
    }
}

} // net
} // parser