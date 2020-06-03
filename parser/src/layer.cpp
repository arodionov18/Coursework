#include "layer.h"


//  AbstractLayer
AbstractLayer::AbstractLayer(std::shared_ptr<AbstractLayer> input) {
    input_layer = input;
}