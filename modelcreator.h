#ifndef MODELCREATOR_H
#define MODELCREATOR_H

#include <vector>
#include "ffnwrapper.h"

using namespace std;
class ModelCreator
{
public:
    ModelCreator();
    bool SetParameters(const vector<int> &params)
    {
        parameters = params;
    }
    bool CreateModel(model_structure *modelstructure);
    int total_number_of_columns;
    int maximum_superficial_lag;
    int lag_frequency;
private:
    vector<int> parameters;


};

std::vector<int> convertToBase(int number, int base);

#endif // MODELCREATOR_H
