#include "cmodelstructure.h"
#include <QFile>
#include <QTextStream>

CModelStructure::CModelStructure() //Default constructor
{

}

CModelStructure::CModelStructure(const CModelStructure &rhs) // Copy constructor
{
    InputTimeSeries = rhs.InputTimeSeries;
    TestTimeSeries = rhs.TestTimeSeries;
    activation_function = rhs.activation_function;
    dt = rhs.dt;
    input_lag_multiplier = rhs.input_lag_multiplier;
    trainaddress = rhs.trainaddress;
    inputcolumns = rhs.inputcolumns;
    lags = rhs.lags;
    n_input_layers = rhs.n_input_layers;
    n_layers = rhs.n_layers;
    n_nodes = rhs.n_nodes;
    n_output_layers = rhs.n_output_layers;
    node_type = rhs.node_type;
    outputcolumns = rhs.outputcolumns;
    testaddress = rhs.testaddress;
    observedaddress = rhs.observedaddress;
    predictedaddress = rhs.predictedaddress;
    realization = rhs.realization;
    outputpath = rhs.outputpath;
    log_output = rhs.log_output;


}
CModelStructure& CModelStructure::operator = (const CModelStructure &rhs) // Operator =
{
    InputTimeSeries = rhs.InputTimeSeries;
    TestTimeSeries = rhs.TestTimeSeries;
    activation_function = rhs.activation_function;
    dt = rhs.dt;
    input_lag_multiplier = rhs.input_lag_multiplier;
    trainaddress = rhs.trainaddress;
    inputcolumns = rhs.inputcolumns;
    lags = rhs.lags;
    n_input_layers = rhs.n_input_layers;
    n_layers = rhs.n_layers;
    n_nodes = rhs.n_nodes;
    n_output_layers = rhs.n_output_layers;
    node_type = rhs.node_type;
    outputcolumns = rhs.outputcolumns;
    testaddress = rhs.testaddress;
    observedaddress = rhs.observedaddress;
    predictedaddress = rhs.predictedaddress;
    realization = rhs.realization;
    outputpath = rhs.outputpath;
    log_output = rhs.log_output;

    return *this;
}

bool CModelStructure::WriteToFile(const QString &filename)
{
    QFile file("output.txt");

        if (!file.open(QIODevice::Append | QIODevice::Text)) {
            return false;
        }

        QTextStream out(&file);
        out<<ParametersToString();

        file.close();
        return true;
}

QString CModelStructure::ParametersToString()
{
    QString out;
    out+="Number of hidden layers:" + QString::number(n_layers);
    out+=", Number of nodes: [";
    for (int i=0; i<n_nodes.size(); i++)
    {
        out+=QString::number(n_nodes[i]);
        if (i<n_nodes.size()-1)
        {
            out+= ",";
        }
    }
    out+="]";
    out+=", Columns: [";
    for (int i=0; i<inputcolumns.size(); i++)
    {
        out+=QString::number(inputcolumns[i]);
        if (i<inputcolumns.size()-1)
        {
            out+= ",";
        }
    }
    out+="]";
    out += ", Lags: [";
    for (unsigned int i=0; i<lags.size(); i++)
    {
        out += "[";
        for (unsigned int j=0; j<lags[i].size(); j++)
        {
            out+=QString::number(lags[i][j]);
            if (j<lags[i].size()-1)
            {
                out+= ",";
            }
        }
        out+="]";
        if (i<lags.size()-1)
        {
            out+= ",";
        }
    }
    out+="]";
    return out;
}

