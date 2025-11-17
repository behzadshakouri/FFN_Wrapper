/**
 * @file main.cpp
 * @brief Main entry point for the FFN Wrapper application.
 *
 * @details
 * This file contains the full runtime configuration for the FFN multi-output wrapper.
 * It defines all parameters directly inside `main()`, allowing complete control by
 * the user without modifying any other source file.
 *
 * ## Responsibilities
 * - Populate the @ref Config struct with all user-defined settings
 * - Configure:
 *   - Model type (ASM vs Settling)
 *   - Constituent (TKN, NH, NO, sCOD, VSS, ND)
 *   - Random seed, realizations
 *   - GA on/off and number of generations
 *   - K-fold settings
 *   - Random model structure search
 *   - All file paths
 *   - ModelCreator parameters
 *
 * - Build the model structure via BuildModelStructure()
 * - Build input/output file addresses via BuildAddresses()
 * - Execute training mode:
 *   - RunGA()
 *   - RunRandom()
 *   - RunSingle()
 *
 * ## Workflow Overview
 *
 * @code
 *         ┌──────────────────┐
 *         │   main.cpp       │
 *         └───────┬──────────┘
 *                 │ Config filled manually
 *                 ▼
 *      ┌─────────────────────────────┐
 *      │ BuildModelStructure()        │
 *      └─────────────────────────────┘
 *                 │ ms with NN architecture
 *                 ▼
 *      ┌─────────────────────────────┐
 *      │ BuildAddresses()            │
 *      └─────────────────────────────┘
 *                 │ ms with file paths
 *                 ▼
 *     ┌───────────────────────────────────┐
 *     │ Training Mode (GA / Random / Single)
 *     └───────────────────────────────────┘
 * @endcode
 *
 * This modular architecture makes main.cpp the **single source of truth** for all runtime behavior.
 *
 * @author
 *   Behzad Shakouri
 *   Arash Massoudieh
 */

#include <mlpack.hpp>
#include <iostream>

#include "config.h"
#include "modelbuilder.h"
#include "trainer.h"

int main()
{
    Config cfg;  ///< Global configuration object.

    // =====================================================================
    // 1. MODEL SELECTION & DATA SETTINGS
    // =====================================================================

    cfg.ASM          = true;        ///< true = ASM, false = simple settling.
    cfg.data_name    = "NO";       ///< Constituent ("NO","NH","sCOD","TKN","VSS","ND").
    cfg.log_output_d = false;       ///< Log-transform output?
    cfg.Seed_number  = 42;          ///< Random seed.
    cfg.Realization  = 1;           ///< Number of realizations.

    if (cfg.ASM)
    {
        cfg.total_data_cols   = 10; ///< Inputs + outputs for ASM.
        cfg.number_of_outputs = 1;  ///< ASM uses 1 output.
    }
    else
    {
        cfg.total_data_cols   = 4;  ///< Simple model.
        cfg.number_of_outputs = 2;  ///< 2 outputs.
    }

    // =====================================================================
    // 2. ARCHITECTURE SET SWITCH
    // =====================================================================
    /**
     * @brief Architecture set selector.
     *
     * 0 → original/old architectures
     * 1 → new architectures
     */
    cfg.architecture_set = 1;   // CONTROL THIS LINE TO SWITCH SET

    // =====================================================================
    // 3. K-FOLD SETTINGS
    // =====================================================================

    cfg.kfold          = false;
    cfg.kfold_num      = 10;
    cfg.kfold_splitMode = 2;

    // =====================================================================
    // 4. GENETIC ALGORITHM SETTINGS
    // =====================================================================

    cfg.GA_switch          = false;
    cfg.GA_Nsim            = 100;
    cfg.MSE_Test           = true;
    cfg.optimized_structure = true;

    // =====================================================================
    // 5. RANDOM MODEL STRUCTURE SEARCH
    // =====================================================================

    cfg.randommodelstructure = false;
    cfg.Random_Nsim          = 1000;

    // =====================================================================
    // 6. FILESYSTEM PATHS
    // =====================================================================

#ifdef PowerEdge
    cfg.path     = "/mnt/3rd900/Projects/FFN_Wrapper/";
    cfg.path_ASM = "/mnt/3rd900/Projects/FFN_Wrapper/ASM/";
#elif defined(Arash)
    cfg.path     = "/home/arash/Projects/FFNWrapper/";
    cfg.path_ASM = "/home/arash/Projects/FFNWrapper/ASM/";
#else
    cfg.path     = "/home/behzad/Projects/FFNWrapper2/";
    cfg.path_ASM = "/home/behzad/Projects/FFNWrapper2/ASM/";
#endif

    cfg.datapath     = cfg.path;
    cfg.datapath_ASM = cfg.path_ASM;

    // =====================================================================
    // 7. MODELCREATOR SETTINGS (for GA/RMS)
    // =====================================================================

    cfg.modelCreator.lag_frequency               = 3;
    cfg.modelCreator.maximum_superficial_lag      = 10;
    cfg.modelCreator.total_number_of_columns      =
        cfg.total_data_cols - cfg.number_of_outputs;

    cfg.modelCreator.max_number_of_layers         = 5;
    cfg.modelCreator.max_lag_multiplier           = 10;
    cfg.modelCreator.max_number_of_nodes_in_layers = 40;

    // =====================================================================
    // 8. BUILD MODEL STRUCTURE AND PATHS
    // =====================================================================

    CModelStructure_Multi ms;

    BuildModelStructure(ms, cfg);   ///< Build layers, nodes, lags, IO columns.
    BuildAddresses(ms, cfg);        ///< Build input/output file paths.

    // =====================================================================
    // 9. SELECT AND EXECUTE TRAINING MODE
    // =====================================================================

    if (cfg.GA_switch)
    {
        RunGA(ms, cfg);
    }
    else if (cfg.randommodelstructure)
    {
        RunRandom(ms, cfg);
    }
    else
    {
        RunSingle(ms, cfg);
    }

    return 0;
}
