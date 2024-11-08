#ifndef FFNWRAPPER_H
#define FFNWRAPPER_H
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <vector>
#include <BTCSet.h>

using namespace mlpack;
using namespace std;

struct model_structure
{
    CTimeSeriesSet<double> *InputTimeSeries = nullptr; //(string& address, bool& tf);
    CTimeSeriesSet<double> *TestTimeSeries = nullptr; //(string& address, bool& tf);
    double dt;
    string inputaddress;
    string testaddress;
    int n_input_layers;
    string activation_function;
    int n_output_layers;
    int n_layers; //
    vector<int> n_nodes; //
    vector<string> node_type; //      102.302,231,44
    vector<vector<int>> lags; //
    vector<int> inputcolumns; //
    vector<int> outputcolumns;
    int input_lag_multiplier;

};

class FFNWrapper : FFN<MeanSquaredError>
{
public:
    FFNWrapper();
    FFNWrapper(const FFNWrapper &F);
    FFNWrapper& operator=(const FFNWrapper& rhs);
    virtual ~FFNWrapper();
    bool DataProcess();
    mat A;

    bool Shifter();
    bool Initiate();
    bool Training();
    bool Testing();
    bool PerformanceMetrics();
    bool DataSave();
    model_structure ModelStructure;    
    CTimeSeriesSet<double> *data;
    CTimeSeriesSet<double> *data2;
    CTimeSeriesSet<double> GetInputData()
    {
        return CTimeSeriesSet<double>(TestInputData,ModelStructure.dt,ModelStructure.lags);
    }
    CTimeSeriesSet<double> GetOutputData()
    {
        return CTimeSeriesSet<double>(TestOutputData,ModelStructure.dt,ModelStructure.lags);
    }
    mat Prediction;
private:
    mat TrainInputData;
    mat TrainOutputData;
    mat TestInputData;
    mat TestOutputData;


};

#endif // FFNWRAPPER_H
