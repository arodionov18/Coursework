#include "layer.h"

#include "Halide.h"

using namespace Halide;

//#############   Layer Parameters  ######################

//  AbstractLayerParams
AbstractLayerParams::AbstractLayerParams(const std::string& name_, const std::string& type_, const std::string& prev_layer_, const std::string& next_layer_) 
                        : name(name_), type(type_), prev_layer(prev_layer_), next_layer(next_layer_) {
    }

AbstractLayerParams::AbstractLayerParams(const AbstractLayerParams& rhs) : name(rhs.name), type(rhs.type), prev_layer(rhs.prev_layer), next_layer(rhs.next_layer) {
}

//  WeightFiller
WeightFiller::WeightFiller(const std::string& type_, double std_) : type(type_), std(std_) {
}

WeightFiller::WeightFiller(const WeightFiller& rhs) : type(rhs.type), std(rhs.std) {
}

//  ConvolutionalParams
ConvolutionalParams::ConvolutionalParams(const AbstractLayerParams& params, bool bias_term_, int num_output_,
                        int pad_, int kernel_size_, int stride_, const WeightFiller& weight_filler_)
                        : AbstractLayerParams(params), bias_term(bias_term_), num_output(num_output_),
                        pad(pad_), kernel_size(kernel_size_), stride(stride_),
                        weight_filler(weight_filler_) {
}

//  BatchNormParams
BatchNormParams::BatchNormParams(const AbstractLayerParams& params, bool stats, double eps_)
                    : use_global_stats(stats), eps(eps_), AbstractLayerParams(params) {
}

//  ScaleParams
ScaleParams::ScaleParams(const AbstractLayerParams& params, bool term) : AbstractLayerParams(params), bias_term(term) {
}

//  PoolingParams
PoolingParams::PoolingParams(AbstractLayerParams base_params, Pool pool_, int kernel_size_, int stride_)
                : AbstractLayerParams(base_params), pool(pool_), kernel_size(kernel_size_), stride(stride_) {
}

//#############   Layers  ######################

//  AbstractLayer
AbstractLayer::AbstractLayer(std::unique_ptr<AbstractLayerParams> params) : params_(std::move(params)) {
}

ConvolutionalLayer::ConvolutionalLayer(std::unique_ptr<AbstractLayerParams> params) : AbstractLayer(std::move(params)) {
}

