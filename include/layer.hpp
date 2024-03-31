#pragma once

#include "activation.hpp"
#include "time.h"

class Layer
{
private:
    float **weights;
    float *biases;
    float *inputValues;
    float *outputValues;
    float **deltas;
    float *biasDeltas;

    float applyActivation(float input)
    {
        switch (activation)
        {
        case SIGMOID:
            return sigmoid(input);
        case RELU:
            return relu(input);
        case LEAKYRELU:
            return leaky_relu(input);
        case NONE:
            return input;
        }
    }

    float applyDerivative(float input)
    {
        switch (activation)
        {
        case SIGMOID:
            return dsigmoid(input);
        case RELU:
            return drelu(input);
        case LEAKYRELU:
            return dleaky_relu(input);
        case NONE:
            return 1;
        }
    }

public:
    Activation activation;
    size_t size;
    Layer *nextLayer;
    bool isInput;
    Layer(Activation activation, size_t size, Layer *nextLayer, bool isInput)
    {
        this->activation = activation;
        this->size = size;
        this->nextLayer = nextLayer;
        this->isInput = isInput;
    }
    Layer(Activation activation, size_t size, bool isInput)
    {
        this->activation = activation;
        this->size = size;
        this->isInput = isInput;
        this->nextLayer = nullptr;
    }

    void initialize()
    {
        srand(time(0));
        if (nextLayer != nullptr)
        {
            weights = new float *[size];
            deltas = new float *[size];
            for (size_t i = 0; i < size; i++)
            {
                weights[i] = new float[nextLayer->size];
                deltas[i] = new float[nextLayer->size];
                for (size_t j = 0; j < nextLayer->size; j++)
                {
                    weights[i][j] = 2*((float)rand() / ((float)RAND_MAX) - 0.5);
                    deltas[i][j] = 0;
                }
            }
        }
        if (!isInput)
        {
            biases = new float[size];
            biasDeltas = new float[size];
            for (size_t i = 0; i < size; i++)
            {
                biases[i] = 2*((float)rand() / ((float)RAND_MAX) - 0.5);
                biasDeltas[i] = 0;
            }
        }
    }

    float *forwardPropagate(float *input)
    {
        inputValues = input;
        float *output;
        if (!isInput)
        {
            for (size_t i = 0; i < size; i++)
            {
                input[i] += biases[i];
                input[i] = applyActivation(input[i]);
            }
        }
        outputValues = input;

        if (nextLayer != nullptr)
        {
            output = new float[nextLayer->size];
            for (size_t i = 0; i < nextLayer->size; i++)
            {
                output[i] = 0;
                for (size_t j = 0; j < size; j++)
                {
                    output[i] += weights[j][i] * input[j];
                }
            }
            return output;
        }
        output = input;
        return output;
    }

    float *backPropagateOutput(float *desiredOutput)
    {
        float *partialBias = new float[size];
        for (size_t i = 0; i < size; i++)
        {
            partialBias[i] = applyDerivative(inputValues[i]) * (desiredOutput[i] - outputValues[i]);
            biasDeltas[i] += partialBias[i];
            // biases[i] += learningRate * partialBias[i];
        }
        return partialBias;
    }

    float *backPropagate(float *partialBiasIn)
    {
        for (size_t i = 0; i < nextLayer->size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                deltas[j][i] += partialBiasIn[i] * outputValues[j];
                //weights[j][i] += deltas[j][i] * learningRate;
            }
        }
        if (isInput)
        {
            return nullptr;
        }
        float *partialBiasOut = new float[size];
        for (size_t i = 0; i < size; i++)
        {
            float sumWeights = 0;
            for (size_t j = 0; j < nextLayer->size; j++)
            {
                sumWeights += weights[i][j] * partialBiasIn[j];
            }
            partialBiasOut[i] = sumWeights * applyDerivative(inputValues[i]);
            biasDeltas[i] += partialBiasOut[i];
            //biases[i] += learningRate * partialBiasOut[i];
        }
        return partialBiasOut;
    }

    float *backPropagateOutput(float *desiredOutput, float learningRate)
    {
        float *partialBias = new float[size];
        for (size_t i = 0; i < size; i++)
        {
            partialBias[i] = applyDerivative(inputValues[i]) * (desiredOutput[i] - outputValues[i]);
            //biasDeltas[i] += partialBias[i];
            biases[i] += learningRate * partialBias[i];
        }
        return partialBias;
    }

    float *backPropagate(float *partialBiasIn, float learningRate)
    {
        float *partialBiasOut = new float[size];
        if (!isInput)
        {
            for (size_t i = 0; i < size; i++)
            {
                float sumWeights = 0;
                for (size_t j = 0; j < nextLayer->size; j++)
                {
                    sumWeights += weights[i][j] * partialBiasIn[j];
                }
                partialBiasOut[i] = sumWeights * applyDerivative(inputValues[i]);
                //biasDeltas[i] += partialBiasOut[i];
                biases[i] += learningRate * partialBiasOut[i];
            }
        }
        for (size_t i = 0; i < nextLayer->size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                //deltas[j][i] += partialBiasIn[i] * outputValues[j];
                weights[j][i] += partialBiasIn[i] * outputValues[j] * learningRate;
            }
        }
        return partialBiasOut;
    }

    void applyDeltas(float learningRate) {
        if (nextLayer != nullptr) {
            for (size_t i = 0; i<size; i++) {
                for (size_t j = 0; j<nextLayer->size; j++) {
                    weights[i][j] += deltas[i][j]*learningRate;
                    deltas[i][j] = 0;
                }
            }
        }
        if (!isInput) {
            for (size_t i = 0; i<size; i++) {
                biases[i] += biasDeltas[i]*learningRate;
                biasDeltas[i] = 0;
            }
        }
    }

    void setWeights(float** w) {
        weights = w;
    }

    void setBiases(float* b) {
        biases = b;
    }

    void printWeights() {
        if (nextLayer == nullptr) {
            return;
        }
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextLayer->size; j++) {
                printf("W_%d_%d: %f, ", i, j, weights[i][j]);
            }
        }
        std::cout<<std::endl;
    }

    void printBiases() {
        if (isInput) {
            return;
        }
        for (int i = 0; i<size; i++) {
            printf("B%d: %f, ", i, biases[i]);
        }
        std::cout<<std::endl;
    }
};