#include "ga.h"

template<class T>
GeneticAlgorithm<T>::GeneticAlgorithm()
{

}

template<class T>
void GeneticAlgorithm<T>::Initialize()
{
    Individuals.resize(Settings.totalpopulation);
    models.resize(Settings.totalpopulation);
    for (int i=0; i<Individuals.size(); i++)
    {
        models[i] = model;
        Individuals[i].resize(model.ParametersSize());
        vector<int> splitlocations;
        for (int j=0; j<model.ParametersSize(); j++)
        {
            BinaryNumber B = BinaryNumber::randomBinary(model.MaxParameter(j));
            B.fixSize(BinaryNumber::decimalToBinary(model.MaxParameter(j)).numDigits());
            Individuals[i][j] = B;
            Individuals[i].splitlocations.push_back(BinaryNumber::decimalToBinary(model.MaxParameter(j)).numDigits());
        }
        Individuals[i].display();

    }

}

template<class T>
void GeneticAlgorithm<T>::AssignFitnesses()
{
    for (unsigned int i=0; i<models.size(); i++)
    {
        vector<unsigned long int> parameterset;
        for (unsigned int j=0; j<models[i].ParametersSize(); j++)
        {
            parameterset.push_back(Individuals[i][j].toDecimal());

        }
        models[i].AssignParameters(parameterset);
        models[i].CreateModel();
        cout<<models[i].FFN.ModelStructure.ParametersToString().toStdString()<<endl;
        if (models[i].FFN.ModelStructure.ValidLags())
        {
            Individuals[i].fitness_measures = models[i].Fitness();
            Individuals[i].fitness = Individuals[i].fitness_measures["MSE"];
        }
        else
        {
            Individuals[i].fitness_measures["MSE"]=1e12;
            Individuals[i].fitness_measures["R2"]=0;
            Individuals[i].fitness = Individuals[i].fitness_measures["MSE"];
        }
    }

}
