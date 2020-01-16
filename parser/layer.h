#pragma once

#include "Halide.h"
#include "generated/caffe.pb.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

using Network::LayerParameter;

class AbstractLayer {
public:
    virtual ~AbstractLayer() = default;
    AbstractLayer(LayerParameter* params);

    Halide::Func forward;
    int x, y, z, w;

protected:
    LayerParameter m_params;
    Halide::Var i, j, k, l;
};

class ConvolutionalLayer : public AbstractLayer {
public:
    ConvolutionalLayer(LayerParameter* params);

private:
    LayerParameter m_params;
};

class BatchNormLayer : public AbstractLayer {
public:
    BatchNormLayer(LayerParameter* params);

private:
    LayerParameter m_params;
};

class ScaleLayer : public AbstractLayer {
public:
    ScaleLayer(LayerParameter* params);

private:
    LayerParameter m_params;
};

class ReLuLayer : public AbstractLayer {
public:
    ReLuLayer(LayerParameter* params);

private:
    LayerParameter m_params;
};

class PoolingLayer: public AbstractLayer {
public:
    PoolingLayer(LayerParameter* params);

private:
    LayerParameter m_params;
};


/*struct AbstractLayerParams {
    std::string name;
    std::string type;
    std::string prev_layer;
    std::string next_layer;

    AbstractLayerParams(const std::string& name_, const std::string& type_, const std::string& prev_layer_, const std::string& next_layer_);

    AbstractLayerParams(const AbstractLayerParams& rhs);
};


enum LayerType {
    DATA,
    CONVOLUTIONAL,
    RELU,
    POOLING,
    INNER_PRODUCT,
    ACCURACY,
    DROP_OUT,
    SOFTMAX_LOSS
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
*/