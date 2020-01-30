#include "layer.h"

#include "Halide.h"

using namespace Halide;

//  AbstractLayer
AbstractLayer::AbstractLayer() {
}

DataLayer::DataLayer(const LayerParameter& params) : m_params(params) {
}

ConvolutionalLayer::ConvolutionalLayer(const LayerParameter& params) : m_params(params) {
}

BatchNormLayer::BatchNormLayer(const LayerParameter& params) : m_params(params) {
}

ScaleLayer::ScaleLayer(const LayerParameter& params) : m_params(params) {
}

ReLuLayer::ReLuLayer(const LayerParameter& params) : m_params(params) {
}

PoolingLayer::PoolingLayer(const LayerParameter& params) : m_params(params) {
}

InnerProductLayer::InnerProductLayer(const LayerParameter& params) : m_params(params) {
}

AccuracyLayer::AccuracyLayer(const LayerParameter& params) : m_params(params) {
}

DropOutLayer::DropOutLayer(const LayerParameter& params) : m_params(params) {
}

SoftMaxLayer::SoftMaxLayer(const LayerParameter& params) : m_params(params) {
}