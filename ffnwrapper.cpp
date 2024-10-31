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

    inputcolumns.push_back(1); // Input 1: D(2): Settling element (1)_Coagulant:external_mass_flow_timeseries
    inputcolumns.push_back(49); // Input 2: CV(50): Reactor (1)_Solids:inflow_concentration
    outputcolumns.push_back(10); // Output: V(11): Settling element (1)_Solids:concentration

    data = new CTimeSeriesSet<double>(InputTimeSeries);
    data->writetofile("data.csv");

    return true;
}

bool FFNWrapper::Train()
{

    // Getting data
    mat TrainInputData = data->ToArmaMat(inputcolumns);
    mat TrainOutputData = data->ToArmaMat(outputcolumns);

    //Data Checking
    TrainInputData.save("TrainInputData.csv", arma::file_type::raw_ascii);
    TrainOutputData.save("TrainOutputData.csv", arma::file_type::raw_ascii);

    CTimeSeriesSet<double> TrainDataTS(TrainInputData,0.01);
    TrainDataTS.writetofile("TrainDataTS.txt");

    CTimeSeriesSet<double> OutputDataTS(TrainOutputData,0.01);
    OutputDataTS.writetofile("TrainOutputDataTS.txt");


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
    mat Prediction;

    model.Predict(TestInputData, Prediction);
    cout << "Prediction:" << Prediction;

    Prediction.save("Prediction.txt",arma::file_type::raw_ascii);

    CTimeSeriesSet<double> PredictionTS(Prediction,0.01);
    PredictionTS.writetofile("PredictionTS.csv");

    /* Compute the error between predictions and testLabels,
    now that we have the desired predictions.*/

    /*size_t correct = arma::accu(prediction == OutputData);
    double classificationError = 1 - double(correct) / InputData.n_cols;

    // Print out the classification error for the testing dataset.
    std::cout << "Classification Error for the Test set: " << classificationError << std::endl;*/

    return true;
}

bool FFNWrapper::Shifter()
{
    CTimeSeriesSet<double> InputTimeSeries("/home/behzad/Projects/FFNWrapper/output_c.txt",true);

    //Lags definition
    vector<vector<int>> lags;
    vector<int> lag1; lag1.push_back(0); lag1.push_back(2);
    vector<int> lag2; lag1.push_back(1); lag1.push_back(3);
    lags.push_back(lag1);
    lags.push_back(lag2);

    //Shifting by lags definition (Inputs)
    mat x = InputTimeSeries.ToArmaMatShifter(inputcolumns, lags);

    CTimeSeriesSet<double> ShiftedInputs(x,0.01);
    ShiftedInputs.writetofile("ShiftedInputs.txt");

    //Shifting by lags definition (Outputs)
    mat y = InputTimeSeries.ToArmaMatShifter(outputcolumns, lags);

    CTimeSeriesSet<double> ShiftedOutputs(y,0.01);
    ShiftedOutputs.writetofile("ShiftedOutputs.txt");

    return true;

}
