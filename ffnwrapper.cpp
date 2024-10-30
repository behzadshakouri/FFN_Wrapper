#include "ffnwrapper.h"
#include <mlpack.hpp>

using namespace mlpack;
using namespace std;
using namespace arma;

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
    CTimeSeriesSet<double> InputTimeSeries("/home/behzad/Projects/FFNWrapper/output.txt",true);

    //columns.push_back(1); // A: //t It's not reading this column!!!
    inputcolumns.push_back(0); // Input 1: D(4): Settling element (1)_Coagulant:external_mass_flow_timeseries
    inputcolumns.push_back(49); // Input 2: CV(50): Reactor (1)_Solids:inflow_concentration
    outputcolumns.push_back(10); // Output: V(11): Settling element (1)_Solids:concentration

    data = new CTimeSeriesSet<double>(InputTimeSeries);

    return true;
}

bool FFNWrapper::Train()
{

    // Getting data
    mat TrainInputData = data->ToArmaMat(inputcolumns);
    mat TrainOutputData = data->ToArmaMat(outputcolumns);

    // Initialize the network
    model.Add<Linear>(2); // Connection Layer: InputData to Hidden Layer with 8 Neurons
    model.Add<Sigmoid>(); // Activation Function
    model.Add<Linear>(1); // Connection Layer: Hidden Layer to OutputData with 1 Neuron
    model.Add<Sigmoid>(); // Activation Funchion

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

void FFNWrapper::Test2()
{
    arma::mat X = arma::randu<arma::mat>(1, 100);
    arma::mat y = arma::sin(X);

    // Create an FFN model.
    FFN<MeanSquaredError<>, RandomInitialization> model;

    // Add layers to the model.
    model.Add<Linear<> >(1, 32);
    model.Add<ReLULayer<> >();
    model.Add<Linear<> >(32, 1);

    // Train the model.
    model.Train(X, y);

    // Make predictions on new data.
    arma::mat X_test = arma::randu<arma::mat>(1, 10);
    arma::mat y_pred;
    model.Predict(X_test, y_pred);

    return 0;
}
