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

    AbstractLayerParams(const std::string& name_, const std::string& type_, const std::string& prev_layer_, const std::string& next_layer_);

    AbstractLayerParams(const AbstractLayerParams& rhs);
};


class AbstractLayer {
public:
    virtual ~AbstractLayer() = default;
    AbstractLayer(std::unique_ptr<AbstractLayerParams> params);
protected:
    std::unique_ptr<AbstractLayerParams> params_;
};


struct WeightFiller {
    std::string type;
    double std;

    WeightFiller(const std::string& type_, double std_);

    WeightFiller(const WeightFiller& rhs);
};

struct ConvolutionalParams : public AbstractLayerParams {
    bool bias_term;
    int num_output;
    int pad;
    int kernel_size;
    int stride;
    WeightFiller weight_filler;

    ConvolutionalParams(const AbstractLayerParams& params, bool bias_term_, int num_output_,
                        int pad_, int kernel_size_, int stride_, const WeightFiller& weight_filler_);
};

struct BatchNormParams : public AbstractLayerParams {
    bool use_global_stats;
    double eps;

    BatchNormParams(const AbstractLayerParams& params, bool stats, double eps_);
};

struct ScaleParams : public AbstractLayerParams {
    bool bias_term;

    ScaleParams(const AbstractLayerParams& params, bool term);
};

enum Pool { MAX };

struct PoolingParams : public AbstractLayerParams {
    Pool pool;
    int kernel_size = 3;
    int stride = 2;

    PoolingParams(AbstractLayerParams base_params, Pool pool_, int kernel_size_, int stride_);
};

class ConvolutionalLayer : public AbstractLayer {
public:
    ConvolutionalLayer(std::unique_ptr<AbstractLayerParams> params);

private:
    // unique_ptr<> params;
};

class BatchNormLayer : public AbstractLayer {
public:
    BatchNormLayer(std::unique_ptr<AbstractLayerParams> params);

private:
    // unique_ptr<> params;
};

class ScaleLayer : public AbstractLayer {
public:
    ScaleLayer(std::unique_ptr<AbstractLayerParams> params);

private:
};

class ReLuLayer : public AbstractLayer {
public:
    ReLuLayer(std::unique_ptr<AbstractLayerParams> params);
};

class PoolingLayer: public AbstractLayer {
public:
    PoolingLayer(std::unique_ptr<AbstractLayerParams> params);
};
