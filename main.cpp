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
    F.Prediction.save("prediction.txt",arma::file_type::raw_ascii);
    CTimeSeriesSet<double> PredictionTS(F.Prediction,0.01);
    PredictionTS.writetofile("PredictionTS.csv");

    CTimeSeriesSet<double> InputTimeSeries("/home/behzad/Projects/FFNWrapper/output_c.txt",true);

    vector<int> inputcolumns;
    vector<vector<int>> lags;
    vector<int> lag1; lag1.push_back(0); lag1.push_back(2);
    vector<int> lag2; lag1.push_back(1); lag1.push_back(3);
    lags.push_back(lag1);
    lags.push_back(lag2);
    inputcolumns.push_back(1); // Input 1: D(2): Settling element (1)_Coagulant:external_mass_flow_timeseries
    inputcolumns.push_back(49); // Input 2: CV(50): Reactor (1)_Solids:inflow_concentration



    arma::mat x = InputTimeSeries.ToArmaMatShifter(inputcolumns, lags);

    CTimeSeriesSet<double> output(x,0.01);
    output.writetofile("Shifted.tx");

    // Split the labels from the training set and testing set respectively.
    // Decrement the labels by 1, so they are in the range 0 to (numClasses - 1).

    /*
    arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
    arma::mat testLabels = testData.row(testData.n_rows - 1) - 1;
    trainData.shed_row(trainData.n_rows - 1);
    testData.shed_row(testData.n_rows - 1);
    */
    /*arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
    arma::mat testLabels = testData.row(testData.n_rows - 1) - 1;
    trainData.shed_row(trainData.n_rows - 1);
    testData.shed_row(testData.n_rows - 1);*/

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
