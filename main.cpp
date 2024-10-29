#include <iostream>
#include <mlpack.hpp>
#include "ffnwrapper.h"
#include <BTCSet.h>


using namespace mlpack;
using namespace std;

int main()
{

    //Data.save("Data.csv", arma::file_type::raw_ascii);

    // Load the whole data (OpenHydroQual output).
    CTimeSeriesSet<double> InputTimeSeries("/home/behzad/Projects/FFNWrapper/output.txt",true);

    vector<int> inputcolumns;
    vector<int> outputcolumns;
    //columns.push_back(1); // A: //t It's not reading this column!!!
    inputcolumns.push_back(0); // Input 1: D(4): Settling element (1)_Coagulant:external_mass_flow_timeseries
    inputcolumns.push_back(98); // Input 2: CV(100): Reactor (1)_Solids:inflow_concentration
    outputcolumns.push_back(20); // Output: V(22): Settling element (1)_Solids:concentration

    arma::mat InputData = InputTimeSeries.ToArmaMat(inputcolumns);
    arma::mat OutputData = InputTimeSeries.ToArmaMat(outputcolumns);

    FFNWrapper F;
    F.data = &InputTimeSeries;
    F.inputcolumns = inputcolumns;
    F.outputcolumns = outputcolumns;

    F.Train();
    //F.Test();

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
