//Our main

#include <iostream>
#include <mlpack.hpp>
#include "ffnwrapper.h"
#include <BTCSet.h>
#include "modelcreator.h"
#include <QDebug>

using namespace mlpack;
using namespace std;


int main()
{

    // Defining Model Structure
    CModelStructure mymodelstruct;
    mymodelstruct.n_layers = 1;
    mymodelstruct.n_nodes = {4};
    /*
    mymodelstruct.dt=0.01;
    */

    /*
    mymodelstruct.inputcolumns.push_back(1);
    mymodelstruct.n_input_layers=4;
    mymodelstruct.activation_function="Sigmoid";
    mymodelstruct.n_output_layers=1;
    */

    /*mymodelstruct.input_lag_multiplier = 5;
    modelCreator.max_number_of_nodes_in_layers = 7;
    mymodelstruct.n_layers = 2;
    mymodelstruct.n_nodes = {3,2};
    modelCreator.SetParameters(&mymodelstruct);
    */

#ifdef Arash
    mymodelstruct.inputaddress="/home/arash/Projects/FFNWrapper/observedoutput.txt";
    mymodelstruct.testaddress="/home/arash/Projects/FFNWrapper/observedoutput.txt";
#else
    mymodelstruct.inputaddress="/home/behzad/Projects/Settling_Models/observedoutput.txt";
    mymodelstruct.testaddress="/home/behzad/Projects/Settling_Models/observedoutput.txt";
#endif

    // Defining Inputs
    mymodelstruct.inputcolumns.push_back(0); // Input 1: D(2): Settling element (1)_Coagulant:external_mass_flow_timeseries
    mymodelstruct.inputcolumns.push_back(1); // Input 2: CV(50): Reactor (1)_Solids:inflow_concentration

    // Defining Output(s)
    mymodelstruct.outputcolumns.push_back(2); // Output: V(11): Settling element (1)_Solids:concentration

    //Lags definition
    vector<int> lag1; lag1.push_back(0); lag1.push_back(20); lag1.push_back(50);
    vector<int> lag2; lag2.push_back(0); lag2.push_back(10); lag2.push_back(30);
    mymodelstruct.lags.push_back(lag1);
    mymodelstruct.lags.push_back(lag2);

    //Model creator
    ModelCreator modelCreator;
    modelCreator.lag_frequency = 3;
    modelCreator.maximum_superficial_lag = 3;
    modelCreator.total_number_of_columns = 2;
    modelCreator.max_number_of_layers = 2;
    modelCreator.max_lag_multiplier = 6;

    for (int i=0; i<10; i++)

    {

    modelCreator.CreateRandomModelStructure(&mymodelstruct);

    CModelStructure mymodelstruct2;
    CModelStructure mymodelstruct3(mymodelstruct);
    modelCreator.CreateRandomModelStructure(&mymodelstruct2);

    qDebug()<<"Model2 ?= Model1"<<(mymodelstruct2==mymodelstruct);
    qDebug()<<"Model3 ?= Model1"<<(mymodelstruct3==mymodelstruct);

    qDebug()<<mymodelstruct2.ParametersToString();

    // Running FFNWrapper
    if (mymodelstruct.ValidLags())
    {   FFNWrapper F;
        F.ModelStructure = mymodelstruct;
        F.Initiate();
        F.Training();
        F.Testing();
        F.PerformanceMetrics();
        qDebug()<<mymodelstruct.ParametersToString() << "MSE = " << F.nMSE << "R2 = " << F._R2;
        F.DataSave();
        //data::Save("model.xml","model", F);
    }
    else
        i--;
    }

    return 0;
}


















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
