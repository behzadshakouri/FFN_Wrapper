#ifndef MODELCREATOR_H
#define MODELCREATOR_H

#include <vector>
#include "ffnwrapper.h"

using namespace std;
class ModelCreator
{
public:
    ModelCreator();
    bool SetParameters(const vector<long int> &params)
    {
        parameters = params;
    }
    int ParametersSize();
    bool SetParameters(model_structure *modelstructure);
    bool CreateModel(model_structure *modelstructure);
    int total_number_of_columns = 0;
    int maximum_superficial_lag = 0;
    int lag_frequency = 0;
    int max_number_of_nodes_in_layers;
private:
    vector<long int> parameters;


};

std::vector<int> convertToBase(unsigned long int number, int base);

#endif // MODELCREATOR_H
