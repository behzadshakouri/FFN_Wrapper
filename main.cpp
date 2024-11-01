//Our main

#include <iostream>
#include <mlpack.hpp>
#include "ffnwrapper.h"
#include <BTCSet.h>

using namespace mlpack;
using namespace std;


int main()
{

    FFNWrapper F;
    F.DataProcess();
    F.Train();
    F.Test();
    F.Shifter();


    return 0;
}

//Our main


















//Test2
/*
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <armadillo>
#include <iostream>
#include <cmath>

using namespace mlpack;
//using namespace mlpack::ann;
using namespace arma;

double targetFunction(double x) {
    return sin(x);
}

int main()
{
        // Generate training data.
        const int numPoints = 1000;
        mat X(1, numPoints);
        mat Y(1, numPoints);

        for (int i = 0; i < numPoints; ++i) {
            double x = (i / static_cast<double>(numPoints)) * 10; // Range from 0 to 10
            X(0, i) = x;
            Y(0, i) = targetFunction(x);
        }

        // Define the model.
        FFN<MeanSquaredError> model;
        model.Add<Linear>(10); // Input layer with 1 neuron, hidden layer with 10 neurons
        model.Add<ReLU>();        // Activation function
        model.Add<Linear>(1); // Output layer with 1 neuron

        // Train the model.
        model.Train(X, Y); // 1000 iterations

        // Test the model.
        mat testInput(1, 10);
        mat testOutput;

        for (int i = 0; i < 10; ++i) {
            testInput(0, i) = i; // Testing points from 0 to 9
        }

        model.Predict(testInput, testOutput);

        // Output the results.
        std::cout << "Input\tPredicted\tActual" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << testInput(0, i) << "\t" << testOutput(0, i) << "\t\t" << targetFunction(testInput(0, i)) << std::endl;
        }

        return 0;
    }

    //Test2
*/
