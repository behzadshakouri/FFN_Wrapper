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


FFNWrapper::FFNWrapper():FFN<MeanSquaredError>()
{

}

FFNWrapper::FFNWrapper(const FFNWrapper &rhs):FFN<MeanSquaredError>(rhs)
{
    ModelStructure = rhs.ModelStructure;
    data = rhs.data;

}

FFNWrapper& FFNWrapper::operator=(const FFNWrapper& rhs)
{
    FFN<MeanSquaredError>::operator=(rhs);
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

    ModelStructure.InputTimeSeries = new CTimeSeriesSet<double>(ModelStructure.inputaddress,true);

    // Writing the data for checking
    data = new CTimeSeriesSet<double>(*ModelStructure.InputTimeSeries);

    Shifter();

    return true;
}

bool FFNWrapper::Shifter()
{
    CTimeSeriesSet<double> InputTimeSeries(ModelStructure.inputaddress,true);

    //Shifting by lags definition (Inputs)
    TrainInputData = InputTimeSeries.ToArmaMatShifter(ModelStructure.inputcolumns, ModelStructure.lags);

    CTimeSeriesSet<double> ShiftedInputs(TrainInputData,ModelStructure.dt,ModelStructure.lags);
    //ShiftedInputs.writetofile("ShiftedInputs.txt");

    //Shifting by lags definition (Outputs)
    TrainOutputData = InputTimeSeries.ToArmaMatShifterOutput(ModelStructure.outputcolumns, ModelStructure.lags);

    CTimeSeriesSet<double> ShiftedOutputs = CTimeSeriesSet<double>::OutputShifter(TrainOutputData,ModelStructure.dt,ModelStructure.lags);
    //ShiftedOutputs.writetofile("ShiftedOutputs.txt");

    return true;
}

bool FFNWrapper::Initiate()
{
    DataProcess();

    // Initialize the network
    for (int layer = 0; layer<ModelStructure.n_layers; layer++)
    {
        Add<Linear>(ModelStructure.n_nodes[layer]); // Connection Layer : ModelStructure.n_input_layers
        Add<Sigmoid>(); // Activation Funchion
    }

   //model.Add<Linear>(3); // Connection Layer 2: ModelStructure.n_input_layers
    //model.Add<Sigmoid>(); // Activation Funchion 2
    Add<Linear>(TrainOutputData.n_rows); // Output Layer : ModelStructure.n_output_layers

    return true;
}

bool FFNWrapper::Training()
{

    // Train the model
    Train(TrainInputData, TrainOutputData);

    return true;
}

bool FFNWrapper::Testing()
{
    // Use the Predict method to get the predictions.

    ModelStructure.TestTimeSeries = new CTimeSeriesSet<double>(ModelStructure.testaddress,true);

    // Writing the data for checking
    data2 = new CTimeSeriesSet<double>(*ModelStructure.TestTimeSeries);

    TestInputData = ModelStructure.TestTimeSeries->ToArmaMatShifter(ModelStructure.inputcolumns, ModelStructure.lags);

    CTimeSeriesSet<double> ShiftedInputs(TestInputData,ModelStructure.dt,ModelStructure.lags);

    TestOutputData = ModelStructure.TestTimeSeries->ToArmaMatShifterOutput(ModelStructure.outputcolumns, ModelStructure.lags);

    CTimeSeriesSet<double> ShiftedOutputs = CTimeSeriesSet<double>::OutputShifter(TestOutputData,ModelStructure.dt,ModelStructure.lags);

    Predict(TestInputData, Prediction);
    cout << "Prediction:" << Prediction;

    return true;
}

bool FFNWrapper::PerformanceMetrics()
{
    CTimeSeriesSet<double> PredictionData (Prediction,ModelStructure.dt,ModelStructure.lags);
    PredictionData.writetofile("Prediction.txt");
    CTimeSeriesSet<double> TargetData = GetOutputData();
    TargetData.writetofile("Target.txt");
    double nMSE = diff2(PredictionData.BTC[0],TargetData.BTC[0])/(norm2(TargetData.BTC[0])/TargetData.BTC[0].n);
    double _R2 = R2(PredictionData.BTC[0],TargetData.BTC[0]);
    cout<<"nMSE = "<<nMSE<<endl;
    cout<<"R2 = "<<_R2<<endl;

    return true;
}


bool FFNWrapper::DataSave()
{
    //Input data checking
    data->writetofile("data.csv");

    // Input/Output matrix checking
    TrainInputData.save("TrainInputData.csv", arma::file_type::raw_ascii);
    TrainOutputData.save("TrainOutputData.csv", arma::file_type::raw_ascii);

    CTimeSeriesSet<double> TrainInputDataTS(TrainInputData,ModelStructure.dt);
    TrainInputDataTS.writetofile("TrainInputDataTS.csv");

    CTimeSeriesSet<double> TrainOutputDataTS(TrainOutputData,ModelStructure.dt);
    TrainOutputDataTS.writetofile("TrainOutputDataTS.csv");

    //Prediction results
    Prediction.save("Prediction.csv",arma::file_type::raw_ascii);

    CTimeSeriesSet<double> PredictionTS(Prediction,ModelStructure.dt);
    PredictionTS.writetofile("PredictionTS.csv");

    TestInputData.save("TestInputData.txt",arma::file_type::raw_ascii);
    TestOutputData.save("TestOutputData.txt",arma::file_type::raw_ascii);

    CTimeSeriesSet<double> TestInputDataTS(TestInputData,ModelStructure.dt);
    TestInputDataTS.writetofile("TestInputDataTS.csv");

    CTimeSeriesSet<double> TestOutputDataTS(TestOutputData,ModelStructure.dt);
    TestOutputDataTS.writetofile("TestOutputDataTS.csv");

    //Performance metrics


    return true;
}

