#include "ffnwrapper.h"
#include <mlpack.hpp>

using namespace mlpack;
using namespace std;

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
    inputcolumns.push_back(98); // Input 2: CV(100): Reactor (1)_Solids:inflow_concentration
    outputcolumns.push_back(20); // Output: V(22): Settling element (1)_Solids:concentration

    //arma::mat InputData = InputTimeSeries.ToArmaMat(inputcolumns);
    //arma::mat OutputData = InputTimeSeries.ToArmaMat(outputcolumns);

    data = &InputTimeSeries;

    return true;
}

bool FFNWrapper::Train()
{

    // Getting data
    arma::mat InputData = data->ToArmaMat(inputcolumns);
    arma::mat OutputData = data->ToArmaMat(outputcolumns);

    // Initialize the network
    FFN<> model;
    model.Add<Linear>(4); // Input Single Layer
    model.Add<Sigmoid>(); // Base Layer
    model.Add<Linear>(3); // Hidden Single Layer
    model.Add<LogSoftMax>(); // Loss Function Layer

    // Train the model
    model.Train(InputData, OutputData);

    return true;
}

bool FFNWrapper::Test()
{
    // Getting data
    arma::mat InputData = data->ToArmaMat(inputcolumns);
    arma::mat OutputData = data->ToArmaMat(outputcolumns);

    // Use the Predict method to get the predictions.
    arma::mat predictionTemp;
    FFN<> model;
    model.Predict(InputData, predictionTemp);

    /*
    Since the predictionsTemp is of dimensions (3 x number_of_data_points)
    with continuous values, we first need to reduce it to a dimension of
    (1 x number_of_data_points) with scalar values, to be able to compare with
    testLabels.

    The first step towards doing this is to create a matrix of zeros with the
    desired dimensions (1 x number_of_data_points).

    In predictionsTemp, the 3 dimensions for each data point correspond to the
    probabilities of belonging to the three possible classes.*/

    arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);

    // Find index of max prediction for each data point and store in "prediction"
    for (size_t i = 0; i < predictionTemp.n_cols; ++i)
    {
        prediction(i) = arma::as_scalar(arma::find(
            arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1));
    }

    /*
    Compute the error between predictions and testLabels,
    now that we have the desired predictions.*/

    size_t correct = arma::accu(prediction == OutputData);
    double classificationError = 1 - double(correct) / InputData.n_cols;

    // Print out the classification error for the testing dataset.
    std::cout << "Classification Error for the Test set: " << classificationError << std::endl;


    return true;
}
