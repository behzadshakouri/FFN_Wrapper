#include <mlpack.hpp>
#include <iostream>

#include "config.h"
#include "modelbuilder.h"
#include "trainer.h"

int main()
{
    Config cfg;

    // =============================================================
    // FULL MANUAL CONTROL FROM MAIN
    // =============================================================
    cfg.ASM = true;
    cfg.data_name = "NO";

    cfg.log_output_d = false;
    cfg.Seed_number = 42;

    cfg.Realization = 1;

    if (cfg.ASM) {
        cfg.total_data_cols = 10;
        cfg.number_of_outputs = 1;
    } else {
        cfg.total_data_cols = 4;
        cfg.number_of_outputs = 2;
    }

    cfg.kfold = false;
    cfg.kfold_num = 10;
    cfg.kfold_splitMode = 2;

    cfg.GA_switch = false;
    cfg.GA_Nsim = 100;
    cfg.MSE_Test = true;
    cfg.optimized_structure = true;

    cfg.randommodelstructure = false;
    cfg.Random_Nsim = 1000;

#ifdef PowerEdge
    cfg.path = "/mnt/3rd900/Projects/FFN_Wrapper/";
    cfg.path_ASM = "/mnt/3rd900/Projects/FFN_Wrapper/ASM/";
#elif defined(Arash)
    cfg.path = "/home/arash/Projects/FFNWrapper/";
    cfg.path_ASM = "/home/arash/Projects/FFNWrapper/ASM/";
#else
    cfg.path = "/home/behzad/Projects/FFNWrapper2/";
    cfg.path_ASM = "/home/behzad/Projects/FFNWrapper2/ASM/";
#endif

    cfg.datapath = cfg.path;
    cfg.datapath_ASM = cfg.path_ASM;

    cfg.modelCreator.lag_frequency = 3;
    cfg.modelCreator.maximum_superficial_lag = 10;
    cfg.modelCreator.total_number_of_columns =
            cfg.total_data_cols - cfg.number_of_outputs;

    cfg.modelCreator.max_number_of_layers = 5;
    cfg.modelCreator.max_lag_multiplier = 10;
    cfg.modelCreator.max_number_of_nodes_in_layers = 40;

    // =============================================================
    // RUN PIPELINE
    // =============================================================
    CModelStructure_Multi ms;

    BuildModelStructure(ms, cfg);
    BuildAddresses(ms, cfg);

    if (cfg.GA_switch)
        RunGA(ms, cfg);
    else if (cfg.randommodelstructure)
        RunRandom(ms, cfg);
    else
        RunSingle(ms, cfg);

    return 0;
}
