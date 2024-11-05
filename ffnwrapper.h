#ifndef FFNWRAPPER_H
#define FFNWRAPPER_H

#include <mlpack.hpp>
#include <vector>
#include <BTCSet.h>

using namespace mlpack;
using namespace std;

struct model_structure
{
    double dt;
    int n_input_layers;
    string activation_function;
    int n_output_layers;
    int n_layers;
    vector<int> n_nodes;
    vector<string> node_type;
    vector<vector<int>> lags;
    vector<int> inputcolumns;
    vector<int> outputcolumns;
};

class FFNWrapper : FFN<>
{
public:
    FFNWrapper();
    FFNWrapper(const FFNWrapper &F);
    FFNWrapper& operator=(const FFNWrapper& rhs);
    virtual ~FFNWrapper();
    bool DataProcess();
    mat A;
    FFN <MeanSquaredError> model;
    bool Shifter();
    bool Train();
    bool Test();
    model_structure ModelStructure;    
    CTimeSeriesSet<double> *data;
    CTimeSeriesSet<double> *data2;
    CTimeSeriesSet<double> *data3;
    bool Initiate();
    mat Prediction;
private:
    mat TrainInputData;
    mat TrainOutputData;


};

#endif // FFNWRAPPER_H
