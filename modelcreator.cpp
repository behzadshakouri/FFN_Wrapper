#include "modelcreator.h"

ModelCreator::ModelCreator()
{

}

bool ModelCreator::CreateModel(model_structure *modelstructure)
{
    vector<int> columns = convertToBase(parameters[0],2);

    for (unsigned int i=0; i<columns.size(); i++)
    {
        if (columns[i]==1) modelstructure->inputcolumns.push_back(i);
    }
    int lag_multiplyer = parameters[1];
    for (int i=0; i<total_number_of_columns; i++)
    {
        vector<int> lags_onoff = convertToBase(parameters[i+2],lag_frequency);
        vector<int> lags;
        for (unsigned int j=0; j<lags_onoff.size(); j++)
        {
            if (lags_onoff[j]==0) lags.push_back(j*lag_multiplyer);
        }
        modelstructure->lags.push_back(lags);
    }
    return true;
}

std::vector<int> convertToBase(int number, int base) {
    std::vector<int> result;

    // Handle zero case
    if (number == 0) {
        result.push_back(0);
        return result;
    }

    // Convert the number to the specified base
    while (number > 0) {
        int remainder = number % base;
        result.push_back(remainder);
        number /= base;
    }

    return result;
}
