#ifndef CMODELSTRUCTURE_H
#define CMODELSTRUCTURE_H

#include "BTCSet.h"
#include <string>
#include <QString>

using namespace std;

class CModelStructure
{
public:
    CModelStructure();
    CModelStructure(const CModelStructure &rhs);
    CModelStructure& operator = (const CModelStructure &rhs);
    CTimeSeriesSet<double> *InputTimeSeries = nullptr; //(string& address, bool& tf);
    CTimeSeriesSet<double> *TestTimeSeries = nullptr; //(string& address, bool& tf);
    double dt;
    string trainaddress; //vector<string>
    string testaddress;//vector<string>
    string observedaddress;
    string predictedaddress;
    int realization;
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
    string outputpath;
    bool log_output = false;

    bool operator==(const CModelStructure &m2)
    {
        if (input_lag_multiplier!=m2.input_lag_multiplier)
            return false;
        if (inputcolumns != m2.inputcolumns)
            return false;
        if (n_layers != m2.n_layers)
            return false;
        if (lags.size()!=m2.lags.size())
            return false;
        for (unsigned int i=0; i<lags.size(); i++)
            if (lags[i]!=lags[i])
                return false;
        return true;
    }

    bool WriteToFile(const QString &filename);
    QString ParametersToString();
    bool operator!=(const CModelStructure &m2)
    {
        return !(operator==(m2));
    }

    bool ValidLags()
    {
        for (int i=0; i<lags.size(); i++)
            if (lags[i].size()>0)
                return true;

        return false;
    }

};

#endif // CMODELSTRUCTURE_H
