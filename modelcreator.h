#ifndef MODELCREATOR_H
#define MODELCREATOR_H

#include <vector>
#include "ffnwrapper.h"
#include <gsl/gsl_rng.h>

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
    bool CreateModel(model_structure *modelstructure) const;
    bool CreateRandomModelStructure(model_structure *modelstructure);
    bool AppendModelStructureToFile();
    int total_number_of_columns = 0;
    int maximum_superficial_lag = 0;
    int lag_frequency = 0;
    int max_number_of_nodes_in_layers = 10;
    int max_number_of_layers = 4;
    int max_lag_multiplier = 6;
    void clear(model_structure *modelstructure);
private:
    vector<long int> parameters;
    gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);


};

std::vector<int> convertToBase(unsigned long int number, int base);
bool operator==(const ModelCreator &m1, const ModelCreator &m2)
{
    model_structure modstruct1;
    model_structure modstruct2;
    m1.CreateModel(&modstruct1);
    m2.CreateModel(&modstruct2);
    return (modstruct1==modstruct2);
}
bool operator!=(const ModelCreator &m1, const ModelCreator &m2)
{
    return (!(m1==m2));
}

#endif // MODELCREATOR_H
