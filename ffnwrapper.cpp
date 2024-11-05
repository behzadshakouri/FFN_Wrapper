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
    ModelStructure.dt=0.01;

    // Load the whole data (OpenHydroQual output).
    CTimeSeriesSet<double> InputTimeSeries("/home/behzad/Projects/FFNWrapper/output_c.txt",true);

    ModelStructure.inputcolumns.push_back(1); // Input 1: D(2): Settling element (1)_Coagulant:external_mass_flow_timeseries
    ModelStructure.inputcolumns.push_back(49); // Input 2: CV(50): Reactor (1)_Solids:inflow_concentration
    ModelStructure.outputcolumns.push_back(10); // Output: V(11): Settling element (1)_Solids:concentration

    data = new CTimeSeriesSet<double>(InputTimeSeries);
    data->writetofile("data.csv");

    //Lags definition
    vector<int> lag1; lag1.push_back(0); lag1.push_back(20); lag1.push_back(50);
    vector<int> lag2; lag2.push_back(0); lag2.push_back(10); lag1.push_back(30);
    ModelStructure.lags.push_back(lag1);
    ModelStructure.lags.push_back(lag2);

    Shifter();

    return true;
}

bool FFNWrapper::Shifter()
{
    CTimeSeriesSet<double> InputTimeSeries("/home/behzad/Projects/FFNWrapper/output_c.txt",true);

    //Shifting by lags definition (Inputs)
    TrainInputData = InputTimeSeries.ToArmaMatShifter(ModelStructure.inputcolumns, ModelStructure.lags);

    CTimeSeriesSet<double> ShiftedInputs(TrainInputData,ModelStructure.dt,ModelStructure.lags);
    ShiftedInputs.writetofile("ShiftedInputs.txt");

    //Shifting by lags definition (Outputs)
    TrainOutputData = InputTimeSeries.ToArmaMatShifterOutput(ModelStructure.outputcolumns, ModelStructure.lags);

    CTimeSeriesSet<double> ShiftedOutputs = CTimeSeriesSet<double>::OutputShifter(TrainOutputData,ModelStructure.dt,ModelStructure.lags);
    ShiftedOutputs.writetofile("ShiftedOutputs.txt");

    //Data Checking
    TrainInputData.save("TrainInputData.csv", arma::file_type::raw_ascii);
    TrainOutputData.save("TrainOutputData.csv", arma::file_type::raw_ascii);

    CTimeSeriesSet<double> TrainInputDataTS(TrainInputData,ModelStructure.dt);
    TrainInputDataTS.writetofile("TrainInputDataTS.txt");

    CTimeSeriesSet<double> TrainOutputDataTS(TrainOutputData,ModelStructure.dt);
    TrainOutputDataTS.writetofile("TrainOutputDataTS.txt");

    return true;
}

bool FFNWrapper::Initiate()
{
    DataProcess();

    // Initialize the network
    model.Add<Linear>(6); // Connection Layer : ModelStructure.n_input_layers
    model.Add<Sigmoid>(); // Activation Funchion
   //model.Add<Linear>(3); // Connection Layer 2: ModelStructure.n_input_layers
    //model.Add<Sigmoid>(); // Activation Funchion 2
    model.Add<Linear>(1); // Output Layer : ModelStructure.n_output_layers

    return true;
}

bool FFNWrapper::Train()
{

    // Train the model
    model.Train(TrainInputData, TrainOutputData);

    return true;
}

bool FFNWrapper::Test()
{
    // Use the Predict method to get the predictions.

    model.Predict(TrainInputData, Prediction);
    cout << "Prediction:" << Prediction;

    Prediction.save("Prediction.txt",arma::file_type::raw_ascii);

    CTimeSeriesSet<double> PredictionTS(Prediction,ModelStructure.dt);
    PredictionTS.writetofile("PredictionTS.csv");

    return true;
}
