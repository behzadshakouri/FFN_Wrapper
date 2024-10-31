#include "ffnwrapper.h"
#include <mlpack.hpp>

using namespace mlpack;
using namespace std;
using namespace arma;

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <armadillo>
using namespace mlpack::ann;


FFNWrapper::FFNWrapper():FFN<>()
{

}

FFNWrapper::FFNWrapper(const FFNWrapper &rhs):FFN<>(rhs)
{
    ModelStructure = rhs.ModelStructure;
    data = rhs.data;

}

FFNWrapper& FFNWrapper::operator=(const FFNWrapper& rhs)
{
    FFN<>::operator=(rhs);
    ModelStructure = rhs.ModelStructure;
    data = rhs.data;

    return *this;
}
FFNWrapper::~FFNWrapper()
{

}

bool FFNWrapper::DataProcess()
{
    // Load the whole data (OpenHydroQual output).
    CTimeSeriesSet<double> InputTimeSeries("/home/behzad/Projects/FFNWrapper/output_c.txt",true);

    //columns.push_back(1); // A: //t It's not reading this column!!!
    inputcolumns.push_back(1); // Input 1: D(2): Settling element (1)_Coagulant:external_mass_flow_timeseries
    inputcolumns.push_back(49); // Input 2: CV(50): Reactor (1)_Solids:inflow_concentration
    outputcolumns.push_back(10); // Output: V(11): Settling element (1)_Solids:concentration

    data = new CTimeSeriesSet<double>(InputTimeSeries);
    data->writetofile("data.csv");

    //Working with arma
    A.load("/home/behzad/Projects/FFNWrapper/output_c.txt",arma::file_type::auto_detect);
    A.save("A.csv", arma::file_type::raw_ascii);

    return true;
}

bool FFNWrapper::Train()
{

    // Getting data
    mat TrainInputData = data->ToArmaMat(inputcolumns);
    mat TrainOutputData = data->ToArmaMat(outputcolumns);

    TrainInputData.save("TrainInputData.csv", arma::file_type::raw_ascii);
    TrainOutputData.save("TrainOutputData.csv", arma::file_type::raw_ascii);

    // Initialize the network
    model.Add<Linear>(2); // Connection Layer: InputData to Hidden Layer with 2 Neurons
    model.Add<ReLU>(); // Activation Funchion
    model.Add<Linear>(1); // Output Layer with 1 Neuron

    // Train the model
    model.Train(TrainInputData, TrainOutputData);

    return true;
}

bool FFNWrapper::Test()
{
    // Getting data
    mat TestInputData = data->ToArmaMat(inputcolumns);
    mat TestOutputData = data->ToArmaMat(outputcolumns);

    // Use the Predict method to get the predictions.

    model.Predict(TestInputData, Prediction);
    cout << "Prediction:" << Prediction;

    /*arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);

    // Find index of max prediction for each data point and store in "prediction"
    for (size_t i = 0; i < predictionTemp.n_cols; ++i)
    {
        prediction(i) = arma::as_scalar(arma::find(
            arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1));
    }*/

    /*
    Compute the error between predictions and testLabels,
    now that we have the desired predictions.*/

    /*size_t correct = arma::accu(prediction == OutputData);
    double classificationError = 1 - double(correct) / InputData.n_cols;

    // Print out the classification error for the testing dataset.
    std::cout << "Classification Error for the Test set: " << classificationError << std::endl;*/

    return true;
}
