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
#ifdef PowerEdge
    string outputpath = "/mnt/3rd900/Projects/FFN_Wrapper/ASM/Results/";
#elif Arash
    string outputpath = "/home/arash/Projects/FFNWrapper/ASM/Results/";
#elif
    string outputpath = "/home/behzad/Projects/FFNWrapper2/ASM/Results/";
#endif
    bool MSE_optimization = true; // true for MSE_Test minimization and false for (MSE_Test + MSE_Train) minimization
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
    void WriteToFile();
    vector<Individual> Individuals;
    T model;
    vector<T> models;
    GeneticAlgorithmsettings Settings;
    std::vector<int> getRanks();
    void CrossOver();
    const Individual& selectIndividualByRank();
private:
    unsigned int max_rank=0;
    std::ofstream file;
    unsigned int current_generation=0;

};


#include <ga.hpp>

#endif // GeneticAlgorithm_H
