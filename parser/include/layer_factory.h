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
    virtual std::unique_ptr<AbstractLayer> Produce(LayerParameter* params) = 0;
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

class LayerFactory {
public:

    std::string CreateLayer(LayerParameter parameters); // Create new layer

    bool IsLayerExists() const;

    void DeleteLayer(const std::string& layerName);

    std::unique_ptr<AbstractLayer> GetLayer(const std::string& layerName);

private:
    std::unordered_map<std::string, std::unique_ptr<AbstractLayer>> layers_;
}