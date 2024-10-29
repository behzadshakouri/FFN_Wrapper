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
    bool Train();
    model_structure ModelStructure;
    CTimeSeriesSet<double> *input_data;
    CTimeSeriesSet<double> *output_data;

private:


};

#endif // FFNWRAPPER_H
