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

enum class datacategory {Train, Test};

class FFNWrapper_Multi : FFN<MeanSquaredError>
{
public:
    FFNWrapper_Multi();
    FFNWrapper_Multi(const FFNWrapper_Multi &F);
    FFNWrapper_Multi& operator=(const FFNWrapper_Multi& rhs);
    virtual ~FFNWrapper_Multi();
    bool DataProcess();
    mat A;

    bool Shifter(datacategory);
    bool Initiate(bool dataprocess = true);
    bool Train();
    bool Test();
    bool PerformanceMetrics();
    bool DataSave(datacategory);
    bool Plotter();
    bool Optimizer();
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


private:
    mat TrainInputData;
    mat TrainOutputData;
    mat TestInputData;
    mat TestOutputData;


};


#endif // FFNWrapper_MULTI_H
