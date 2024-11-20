#ifndef MODELCREATOR_H
#define MODELCREATOR_H

#include <vector>
#include "ffnwrapper.h"
#include "ffnwrapper_multi.h"
#include <gsl/gsl_rng.h>
#include <QString>
#include <BTCSet.h>


using namespace std;
class ModelCreator
{
public:
    ModelCreator();
    ModelCreator(const ModelCreator &other);
    ModelCreator &operator=(const ModelCreator &other);
    bool SetParameters(const vector<long int> &params)
    {
        parameters = params;
    }
    int ParametersSize();
    bool SetParameters(CModelStructure *modelstructure);
    bool CreateModel(CModelStructure *modelstructure) const;
    bool CreateModel(CModelStructure_Multi *modelstructure) const;
    bool CreateRandomModelStructure(CModelStructure *modelstructure);
    bool CreateRandomModelStructure(CModelStructure_Multi *modelstructure);
    long unsigned int MaxParameter(int i);
    map<string, double> Fitness();
    void AssignParameters(const vector<long unsigned int> &x);
    bool CreateModel();
    FFNWrapper_Multi FFN;
    bool AppendModelStructureToFile();
    int total_number_of_columns = 0;
    bool initiated = false;
    int maximum_superficial_lag = 0;
    int lag_frequency = 0;
    int max_number_of_nodes_in_layers = 10;
    int max_number_of_layers = 4;
    int max_lag_multiplier = 6;
    void clear(CModelStructure *modelstructure);
    void clear(CModelStructure_Multi *modelstructure);
    bool operator==(const ModelCreator &m2)
    {
        CModelStructure modstruct1;
        CModelStructure modstruct2;
        CreateModel(&modstruct1);
        m2.CreateModel(&modstruct2);
        return (modstruct1==modstruct2);
    }
    bool operator!=(const ModelCreator &m2)
    {
        return (!operator==(m2));
    }


private:
    vector<long int> parameters;
    gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);


};

std::vector<int> convertToBase(unsigned long int number, int base);


#endif // MODELCREATOR_H
