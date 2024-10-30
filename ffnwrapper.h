#ifndef FFNWRAPPER_H
#define FFNWRAPPER_H

#include <mlpack.hpp>
#include <vector>
#include <BTCSet.h>

using namespace mlpack;
using namespace std;

struct model_structure
{
    int n_layers;
    vector<int> n_nodes;
    vector<string> node_type;
};

class FFNWrapper : FFN<>
{
public:
    FFNWrapper();
    FFNWrapper(const FFNWrapper &F);
    FFNWrapper& operator=(const FFNWrapper& rhs);
    virtual ~FFNWrapper();
    bool DataProcess();
    FFN<> model;
    bool Train();
    bool Test();
    void Test2();
    model_structure ModelStructure;    
    CTimeSeriesSet<double> *data;
    vector<int> inputcolumns;
    vector<int> outputcolumns;
    mat Prediction;
private:


};

#endif // FFNWRAPPER_H
