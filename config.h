/**
 * @file config.h
 * @brief Defines the Config structure used to control all runtime parameters of the FFN Wrapper.
 *
 * @details
 * This header contains the central configuration structure for the entire FFN Wrapper pipeline
 * (data preparation, model structure generation, GA, K-fold training, and prediction).
 *
 * The philosophy:
 * - main.cpp controls all parameters
 * - Config is a plain container of values
 * - Other modules (modelbuilder, trainer) depend on Config but do not hide configuration
 *
 * The structure includes:
 * - Simulation settings
 * - Data/model settings
 * - GA & K-fold parameters
 * - Paths
 * - Architecture set selector
 * - A ModelCreator instance for random/GA-generated structures
 */

#pragma once

#include <string>
#include "modelcreator.h"
#include "ffnwrapper_multi.h"

/**
 * @struct Config
 * @brief Central configuration structure for all model settings and runtime parameters.
 *
 * @details
 * All parameters are manually assigned in main.cpp. This struct provides:
 * - Model type selection (ASM / Settling)
 * - Constituent selection (TKN, NH, NO, etc.)
 * - GA vs. Random vs. Manual mode switching
 * - K-fold settings
 * - Input/output dataset paths
 * - Random seed control
 * - Architecture set selection (old vs new combos)
 * - ModelCreator settings for lag/node/layer generation
 */
struct Config
{
    int Realization;              ///< Number of realizations to run.

    double total_data_cols;       ///< Total number of columns (inputs + outputs).
    double number_of_outputs;     ///< Number of output columns.

    bool ASM;                     ///< true = ASM model, false = simple settling model.
    std::string data_name;        ///< Constituent name ("TKN", "NH", "NO", "sCOD", "VSS", "ND", ...).
    bool log_output_d;            ///< Whether to log-transform output.

    double Seed_number;           ///< Random seed for reproducibility.

    bool kfold;                   ///< Whether to use K-fold training.
    int  kfold_num;               ///< Number of folds.
    int  kfold_splitMode;         ///< 0=random, 1=expanding, 2=fixed.

    bool   GA_switch;             ///< true = GA optimization enabled.
    double GA_Nsim;               ///< Number of GA generations.
    bool   MSE_Test;              ///< GA objective uses only MSE-Test.
    bool   optimized_structure;   ///< Whether to use GA-optimized structure.

    bool   randommodelstructure;  ///< true = random model structures (RMS mode).
    double Random_Nsim;           ///< Number of random structures.

    /**
     * @brief Architecture set selector.
     *
     * @details
     * - 0 → original/old architectures (default)
     * - 1 → new architectures (Behzad 2025 combos)
     *
     * You can extend with 2,3,... in the future for more sets.
     */
    int architecture_set = 0;

    std::string path;             ///< Root project path for non-ASM models.
    std::string path_ASM;         ///< Root project path for ASM models.
    std::string datapath;         ///< Data path for non-ASM datasets.
    std::string datapath_ASM;     ///< Data path for ASM datasets.

    ModelCreator modelCreator;    ///< ModelCreator instance for GA/RMS.
};

/**
 * @brief Build all input/output address paths for training and test data.
 *
 * @param ms  Model structure to update.
 * @param cfg Configuration parameters (read-only).
 *
 * @details
 * Populates:
 * - trainaddress
 * - testaddress
 * - outputpath
 * - trainobservedaddress
 * - trainpredictedaddress
 * - testobservedaddress
 * - testpredictedaddress
 */
void BuildAddresses(CModelStructure_Multi& ms, const Config& cfg);
