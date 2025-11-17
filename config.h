#pragma once

#include <string>
#include "modelcreator.h"
#include "ffnwrapper_multi.h"

struct Config
{
    int Realization;

    double total_data_cols;
    double number_of_outputs;

    bool ASM;
    std::string data_name;
    bool log_output_d;

    double Seed_number;

    bool kfold;
    int kfold_num;
    int kfold_splitMode;

    bool GA_switch;
    double GA_Nsim;
    bool MSE_Test;
    bool optimized_structure;

    bool randommodelstructure;
    double Random_Nsim;

    std::string path;
    std::string path_ASM;
    std::string datapath;
    std::string datapath_ASM;

    ModelCreator modelCreator;
};

// Builds all address vectors inside ms (train/test/output)
void BuildAddresses(CModelStructure_Multi& ms, const Config& cfg);
