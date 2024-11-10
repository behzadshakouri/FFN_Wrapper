#ifndef FFNWRAPPER_H
#define FFNWRAPPER_H
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <vector>
#include <BTCSet.h>
#include "cmodelstructure.h"

using namespace mlpack;
using namespace std;



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
    CModelStructure ModelStructure;
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
