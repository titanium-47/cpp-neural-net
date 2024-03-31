#pragma once
#include "activation.hpp"
#include "layer.hpp"

class NeuralNet {
    private:
            Layer** layers;
            int numLayers;
            float** trainingX;
            float** trainingY;
            int samples;
            float* stepResult;

            void shuffle(int arr[], int length) {
                srand(time(0));
                for (int i = 0; i<length; i++) {
                    int ni = (int)((((float)rand())/(RAND_MAX+1))*length);
                    int temp = arr[i];
                    arr[i] = arr[ni];
                    arr[ni] = temp;
                }
            } 

        public:
            NeuralNet(int* neurons, int numLayers, Activation* activations) {
                this->numLayers = numLayers;
                layers = new Layer*[numLayers];
                layers[numLayers-1] = new Layer(activations[numLayers-2], neurons[numLayers-1], false);
                for (int i = numLayers-2; i>0; i--) {
                    layers[i] = new Layer(activations[i-1], neurons[i], layers[i+1], false);
                }
                layers[0] = new Layer(NONE, neurons[0], layers[1], true);
                for (int i = 0; i<numLayers; i++) {
                    layers[i]->initialize();
                    layers[i]->printBiases();
                    layers[i]->printWeights();
                }
            }

            void train(float** x, float** y, int samples, int epochs, float learningRate) {
                this->samples = samples;
                int shuffled[samples] = { 0 };
                for (int i = 0; i<samples; i++) {
                    shuffled[i] = i;
                }
                // for(int i = 0; i<samples; i++) {
                //     std::cout<<shuffled[i]<<std::endl;
                // }
                this->trainingX = x;
                this->trainingY = y;
                for (int epoch = 0; epoch<epochs; epoch++) {
                    shuffle(shuffled, samples);
                    for (int sample = 0; sample < samples; sample++) {
                        stepResult = trainingX[shuffled[sample]];
                        for (int i = 0; i<numLayers; i++) {
                            stepResult = layers[i]->forwardPropagate(stepResult);
                        }
                        stepResult = layers[numLayers-1]->backPropagateOutput(trainingY[shuffled[sample]], learningRate);
                        for (int i = numLayers - 2; i>=0; i--) {
                            stepResult = layers[i]->backPropagate(stepResult, learningRate);
                        }
                    }
                    // for (int i = 0; i<numLayers; i++) {
                    //     layers[i]->applyDeltas(learningRate);
                    // }
                    // printf("Epoch %d\n", epoch);
                    // printWB();
                }
                printWB();
            }

            void setWeights(float** weights[]) {
                for (int i = 0; i<numLayers-1; i++) {
                    layers[i]->setWeights(weights[i]);
                }
            }

            void setBiases(float* biases[]) {
                for (int i = 1; i<numLayers; i++) {
                    layers[i]->setBiases(biases[i-1]);
                }
            }

            float* test(float* input) {
                stepResult = input;
                for (int i = 0; i<numLayers; i++) {
                    stepResult = layers[i]->forwardPropagate(stepResult);
                }
                return stepResult;
            }

            void printWB() {
                for (int i = 0; i<numLayers; i++) {
                    layers[i]->printBiases();
                    layers[i]->printWeights();
                }
                std::cout<<std::endl;
            }
};