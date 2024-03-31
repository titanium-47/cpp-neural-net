#include <iostream>
#include <math.h>
#include "activation.hpp"
#include "nn.hpp"

int main(int, char **)
{
    NeuralNet *ann = new NeuralNet(new int[4]{2, 4, 4, 1}, 4, new Activation[3]{Activation::RELU, Activation::LEAKYRELU, Activation::SIGMOID});
    float **samplesX = new float *[4];
    for (int i = 0; i < 4; i++)
    {
        samplesX[i] = new float[2];
    }
    samplesX[0][0] = 0.0f;
    samplesX[0][1] = 0.0f;
    samplesX[1][0] = 0.0f;
    samplesX[1][1] = 1.0f;
    samplesX[2][0] = 1.0f;
    samplesX[2][1] = 0.0f;
    samplesX[3][0] = 1.0f;
    samplesX[3][1] = 1.0f;

    float **samplesY = new float *[4];
    for (int i = 0; i < 4; i++)
    {
        samplesY[i] = new float[1];
    }

    samplesY[0][0] = 0.0f;
    samplesY[1][0] = 1.0f;
    samplesY[2][0] = 1.0f;
    samplesY[3][0] = 0.2f;

    ann->train(samplesX, samplesY, 4, 20000, 0.1);
    std::cout << *ann->test(new float[2]{0.0f, 0.0f}) << std::endl;
    std::cout << *ann->test(new float[2]{1.0f, 0.0f}) << std::endl;
    std::cout << *ann->test(new float[2]{0.0f, 1.0f}) << std::endl;
    std::cout << *ann->test(new float[2]{1.0f, 1.0f}) << std::endl;
    return 1;
}
