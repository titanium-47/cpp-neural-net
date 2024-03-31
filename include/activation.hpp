#pragma once
#include <math.h>

enum Activation {
    SIGMOID,
    RELU,
    LEAKYRELU,
    NONE
};

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float dsigmoid(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

float relu(float x) {
    return x > 0 ? x : 0;
}

float drelu(float x) {
    return x > 0 ? 1 : 0;
}

float leaky_relu(float x) {
    return x > 0 ? x : 0.01 * x;
}

float dleaky_relu(float x) {
    return x > 0 ? 1 : 0.01;
}