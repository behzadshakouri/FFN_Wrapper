#ifndef GeneticAlgorithm_H
#define GeneticAlgorithm_H

#include <vector>

#include "Binary.h"
#include "individual.h"

struct GeneticAlgorithmsettings
{
    unsigned int totalpopulation = 40;
    unsigned int generations = 100;
    double mutation_probability = 0.05;
};

using namespace std;

template<class T>
class GeneticAlgorithm
{
public:
    GeneticAlgorithm();
    T Optimize();
    void AssignFitnesses();
    void Initialize();
    vector<Individual> Individuals;
    T model;
    vector<T> models;
    GeneticAlgorithmsettings Settings;
    std::vector<int> getRanks();
    void CrossOver();
    const Individual& selectIndividualByRank();
private:
    unsigned int max_rank=0;

};

#include <ga.hpp>

#endif // GeneticAlgorithm_H
