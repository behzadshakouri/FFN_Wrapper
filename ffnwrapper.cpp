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

#include <QVector>
#include <iostream>
#include <cmath>
#include <gnuplot-iostream.h>

#include <ensmallen.hpp>  // Ensmallen header file

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
    //cout << "Prediction:" << Prediction;

    return true;
}

bool FFNWrapper::PerformanceMetrics()
{
    CTimeSeriesSet<double> PredictionData (Prediction,ModelStructure.dt,ModelStructure.lags);
    PredictionData.writetofile("Prediction.txt");
    CTimeSeriesSet<double> TargetData = GetOutputData();
    TargetData.writetofile("Target.txt");
    nMSE = diff2(PredictionData.BTC[0],TargetData.BTC[0])/(norm2(TargetData.BTC[0])/TargetData.BTC[0].n);
    _R2 = R2(PredictionData.BTC[0],TargetData.BTC[0]);

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

    cout<<"nMSE = "<<nMSE<<endl;
    cout<<"R2 = "<<_R2<<endl;

    return true;
}

/*

bool FFNWrapper:: Plotter()
{
    CTimeSeriesSet<double> Observed(ModelStructure.inputaddress,true);

    CTimeSeriesSet<double> Predicted(ModelStructure.testaddress,true);

    vector<double> d1={1, 2};
    vector<double> d2={2, 4};
    vector<pair<double, double>> plotdata;
    for (int i=0; i<Observed.maxnumpoints(); i++)
    {
        plotdata.push_back(make_pair(Observed.BTC[0].GetT(i),Observed.BTC[0].GetC(i)));
    }


    //plotdata.push_back(make_pair(Observed,Predicted));   // Store (time, value) pairs

    /*

    std::vector<double> time, values1, values2;
    // Generate some dummy data for two time series
    for (double i = 0; i < 100; i += 0.1) {
        time.push_back(i);
        values1.push_back(sin(i / 10.0) * 10.0);  // First time series (sin wave)
        values2.push_back(cos(i / 10.0) * 10.0);  // Second time series (cos wave)
    }

    // Create a vector of pairs to send to Gnuplot for plotting
    std::vector<std::pair<double, double>> data1, data2;

    for (size_t i = 0; i < time.size(); ++i) {
        data1.push_back(std::make_pair(time[i], values1[i]));
        data2.push_back(std::make_pair(time[i], values2[i]));
    }

    plotdata.push_back(make_pair(d1,d2));   // Store (time, value) pairs

    */

/*
    // Create a Gnuplot object
    Gnuplot gp;

    // Set titles and labels
    gp << "set title 'Time Series Plot'\n";
    gp << "set xlabel 'Observed'\n";
    gp << "set ylabel 'Predicted'\n";

    // Plot the data using lines
    gp << "plot '-' with lines title 'Time Series Data'\n";
    //gp.send1d(data1);  // Send the data to Gnuplot

    gp.send1d(plotdata);

    return true;
}
*/


bool FFNWrapper:: Optimizer()
{
/*
    // Define the objective function to minimize (f(x) = x^2 + 4x + 4).
    class QuadraticFunction
    {
    public:
        // Function value at a given point x.
        double Evaluate(const rowvec& parameters)
        {
            // f(x) = x^2 + 4x + 4
            double x = parameters(0);
            return x * x + 4 * x + 4;
        }

        // Gradient of the objective function.
        void Gradient(const rowvec& parameters, rowvec& gradient)
        {
            // Derivative of f(x) = 2x + 4
            double x = parameters(0);
            gradient(0) = 2 * x + 4;
        }
    };
        // Create an instance of the quadratic function.
        QuadraticFunction f;

        // Initial parameters (let's start at x = 10).
        rowvec initialPoint = {10};

        // Create the optimizer (using Stochastic Gradient Descent in this case).
        ens::SGD optimizer(0.1, 1000, 1e-6);

        // Optimize the function using the gradient descent algorithm.
        optimizer.Optimize(f, initialPoint);

        // Output the result.
        std::cout << "Optimal point: " << initialPoint(0) << std::endl;
        std::cout << "Optimal value: " << f.Evaluate(initialPoint) << std::endl;
*/
        return true;
}
