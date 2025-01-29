#ifndef FFNWrapper_MULTI_H
#define FFNWrapper_MULTI_H
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <vector>
#include <BTCSet.h>
#include "cmodelstructure_multi.h"
#include <gnuplot-iostream.h>

using namespace mlpack;
using namespace std;
using namespace arma;

enum class datacategory {Train, Test};

class FFNWrapper_Multi : FFN<MeanSquaredError>
{
public:
    FFNWrapper_Multi();
    FFNWrapper_Multi(const FFNWrapper_Multi &F);
    FFNWrapper_Multi& operator=(const FFNWrapper_Multi& rhs);
    virtual ~FFNWrapper_Multi();

    bool Initiate(bool dataprocess = true);
    bool DataProcess();
    bool Shifter(datacategory);
    bool Transformation();
    bool Train();
    bool Test();
    bool PerformanceMetrics();
    bool DataSave(datacategory);
    bool Plotter();
    bool Optimizer();
    mat A;
    vector<int> segment_sizes;
    CModelStructure_Multi ModelStructure;
    //CTimeSeriesSet<double> *data = nullptr;
    //CTimeSeriesSet<double> *data2 = nullptr;
    bool silent = true;
    CTimeSeriesSet<double> GetInputData()
    {
        return CTimeSeriesSet<double>(TestInputData,ModelStructure.dt,ModelStructure.lags);
    }
    CTimeSeriesSet<double> GetOutputData()
    {
        return CTimeSeriesSet<double>(TestOutputData,ModelStructure.dt,ModelStructure.lags);

    }

    mat Prediction;
    double nMSE = -999;
    double _R2 = -999;

    //Normalization
    mlpack::data::MinMaxScaler minMaxScaler_tr_i;
    mlpack::data::MinMaxScaler minMaxScaler_tr_o;
    mlpack::data::MinMaxScaler minMaxScaler_te_i;
    mlpack::data::MinMaxScaler minMaxScaler_te_o;


private:
    mat TrainInputData;
    mat TrainOutputData;
    mat TestInputData;
    mat TestOutputData;


};


#endif // FFNWrapper_MULTI_H
