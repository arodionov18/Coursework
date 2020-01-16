#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "layer.h"

//----------------------------------------------------------
class ILayerProducer {
public:
    virtual ~ILayerProducer() = default;
    virtual std::unique_ptr<AbstractLayer> Produce(const AbstractLayerParams& params) = 0;
};
// ---------------------------------------------------------
template<class Layer>
class LayerProducer : public ILayerProducer {
public:
    virtual std::unique_ptr<Layer> Produce() override {
        return std::make_unique<Layer>();
    }
};

class LayerRegistry {
public:
    template<class Layer>
    void RegisterLayer(const std::string& layer_name) {
        producers_.insert({layer_name, std::make_unique<LayerProducer<Layer>>()});
    }
private:
    std::unordered_map<std::string, std::unique_ptr<ILayerProducer>> producers_;
};