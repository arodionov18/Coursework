#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

struct AbstractLayerParams {
    std::string name;
    std::string type;
    std::string prev_layer;
    std::string next_layer;

    AbstractLayerParams(const std::string& name_, const std::string& type_, const std::string& prev_layer_, const std::string& next_layer_) 
                        : name(name_), type(type_), prev_layer(prev_layer_), next_layer(next_layer_) {
    }

    AbstractLayerParams(const AbstractLayerParams& rhs) : name(rhs.name), type(rhs.type), prev_layer(rhs.prev_layer), next_layer(rhs.next_layer) {
    }
};


class AbstractLayer {
public:
    virtual ~AbstractLayer() = default;
    AbstractLayer(std::unique_ptr<AbstractLayerParams> params) : params_(std::move(params)) {
    }

protected:
    std::unique_ptr<AbstractLayerParams> params_;
};


struct WeightFiller {
    std::string type;
    double std;

    WeightFiller(const std::string& type_, double std_) : type(type_), std(std_) {
    }

    WeightFiller(const WeightFiller& rhs) : type(rhs.type), std(rhs.std) {
    }
};

struct ConvolutionalParams : public AbstractLayerParams {
    bool bias_term;
    int num_output;
    int pad;
    int kernel_size;
    int stride;
    WeightFiller weight_filler;

    ConvolutionalParams(const AbstractLayerParams& params, bool bias_term_, int num_output_,
                        int pad_, int kernel_size_, int stride_, const WeightFiller& weight_filler_)
                        : AbstractLayerParams(params), bias_term(bias_term_), num_output(num_output_),
                        pad(pad_), kernel_size(kernel_size_), stride(stride_),
                        weight_filler(weight_filler_) {
    }
};

struct BatchNormParams : public AbstractLayerParams {
    bool use_global_stats;
    double eps;

    BatchNormParams(const AbstractLayerParams& params, bool stats, double eps_)
                    : use_global_stats(stats), eps(eps_), AbstractLayerParams(params) {
    }
};

struct ScaleParams : public AbstractLayerParams {
    bool bias_term;

    ScaleParams(const AbstractLayerParams& params, bool term) : AbstractLayerParams(params), bias_term(term) {
    }
};

enum Pool { MAX };

struct PoolingParams : public AbstractLayerParams {
    Pool pool;
    int kernel_size = 3;
    int stride = 2;

    PoolingParams(AbstractLayerParams base_params, Pool pool_, int kernel_size_, int stride_)
                : AbstractLayerParams(base_params), pool(pool_), kernel_size(kernel_size_), stride(stride_) {
    }
};

class ConvolutionalLayer : public AbstractLayer {
public:
    ConvolutionalLayer(std::unique_ptr<AbstractLayerParams> params) : AbstractLayer(std::move(params)) {
    }
};