#include <iostream>
#include <mlpack.hpp>
#include "ffnwrapper.h"
#include <BTCSet.h>


using namespace mlpack;
using namespace std;

int main()
{


    //Data.save("Data.csv", arma::file_type::raw_ascii);

    FFNWrapper F;
    F.DataProcess();
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
