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
    /**
     * @brief Global configuration object.
     * @details
     * All settings must be explicitly assigned below.
     */
    Config cfg;

    // =====================================================================
    // 1. MODEL SELECTION & DATA SETTINGS
    // =====================================================================

    /**
     * @brief Model selection.
     * @details
     * Set true for ASM (wastewater treatment),
     * or false for simple settling model.
     */
    cfg.ASM = true;

    /**
     * @brief Target constituent for ASM prediction.
     * @details
     * Allowed: "NO", "NH", "sCOD", "TKN", "VSS", "ND"
     */
    cfg.data_name = "NO";

    /**
     * @brief Apply log-transform to output.
     */
    cfg.log_output_d = false;

    /**
     * @brief Random seed for reproducibility.
     */
    cfg.Seed_number = 42;

    /**
     * @brief Number of model realizations to run.
     * @details Paths and outputs will be duplicated for each realization.
     */
    cfg.Realization = 1;


    // Set dataset input/output sizes based on model type
    if (cfg.ASM) {
        cfg.total_data_cols   = 10;   ///< Inputs + outputs for ASM
        cfg.number_of_outputs = 1;    ///< ASM uses 1 output
    }
    else {
        cfg.total_data_cols   = 4;    ///< Simple settling model
        cfg.number_of_outputs = 2;    ///< 2-output settling model
    }


    // =====================================================================
    // 2. K-FOLD SETTINGS
    // =====================================================================

    cfg.kfold          = false;  ///< Enable/disable K-fold
    cfg.kfold_num      = 10;     ///< K-fold value (e.g., 10 = 90/10 split)
    cfg.kfold_splitMode = 2;     ///< 0=random, 1=expanding, 2=fixed window


    // =====================================================================
    // 3. GENETIC ALGORITHM SETTINGS
    // =====================================================================

    cfg.GA_switch       = true; ///< true = enable GA optimization
    cfg.GA_Nsim         = 1000;    ///< Number of GA generations
    cfg.MSE_Test        = true;  ///< Optimize GA using only MSE-Test
    cfg.optimized_structure = true;


    // =====================================================================
    // 4. RANDOM MODEL STRUCTURE SEARCH
    // =====================================================================

    cfg.randommodelstructure = false; ///< true = random search
    cfg.Random_Nsim          = 1000;  ///< Number of random models


    // =====================================================================
    // 5. FILESYSTEM PATHS
    // =====================================================================

#ifdef PowerEdge
    cfg.path        = "/mnt/3rd900/Projects/FFN_Wrapper/";
    cfg.path_ASM    = "/mnt/3rd900/Projects/FFN_Wrapper/ASM/";
#elif defined(Arash)
    cfg.path        = "/home/arash/Projects/FFNWrapper/";
    cfg.path_ASM    = "/home/arash/Projects/FFNWrapper/ASM/";
#else
    cfg.path        = "/home/behzad/Projects/FFNWrapper2/";
    cfg.path_ASM    = "/home/behzad/Projects/FFNWrapper2/ASM/";
#endif

    cfg.datapath     = cfg.path;
    cfg.datapath_ASM = cfg.path_ASM;


    // =====================================================================
    // 6. MODELCREATOR SETTINGS (for Random or GA modes)
    // =====================================================================

    cfg.modelCreator.lag_frequency               = 3;
    cfg.modelCreator.maximum_superficial_lag      = 10;
    cfg.modelCreator.total_number_of_columns      =
        cfg.total_data_cols - cfg.number_of_outputs;

    cfg.modelCreator.max_number_of_layers         = 5;
    cfg.modelCreator.max_lag_multiplier           = 10;
    cfg.modelCreator.max_number_of_nodes_in_layers = 40;


    // =====================================================================
    // 7. CONSTRUCT MODEL STRUCTURE AND PATHS
    // =====================================================================

    CModelStructure_Multi ms;

    // Build NN structure (layers, nodes, inputs, outputs, lags)
    BuildModelStructure(ms, cfg);

    // Build dataset and output file paths
    BuildAddresses(ms, cfg);


    // =====================================================================
    // 8. SELECT AND EXECUTE TRAINING MODE
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
