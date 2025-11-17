/**
 * @file config.h
 * @brief Defines the Config structure used to control all runtime parameters of the FFN Wrapper.
 *
 * @details
 * This header contains the central configuration structure for the entire FFN Wrapper pipeline
 * (data preparation, model structure generation, GA optimization, K-fold training, and prediction).
 *
 * The philosophy of the new architecture is:
 * - **main.cpp controls all parameters**
 * - The Config struct acts as a clean container for values
 * - Other modules (modelbuilder, trainer) use Config but do NOT modify its behavior
 *
 * The structure includes:
 * - Simulation settings
 * - Data/model settings
 * - GA & K-fold parameters
 * - Paths
 * - A ModelCreator instance for random/GA-generated structures
 *
 * Every field is documented so IDEs and Doxygen will show tooltips for all parameters.
 *
 * @author
 *   Behzad Shakouri
 *   Arash Massoudieh
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
 * All parameters are manually assigned in @c main.cpp.
 * This struct is intentionally kept plain (no methods) to maximize clarity and usability.
 *
 * It provides:
 * - Model type selection (ASM / Settling)
 * - Constituent selection (TKN, NH, NO, etc.)
 * - GA vs. Random vs. Manual mode switching
 * - K-fold settings
 * - Input/output dataset paths
 * - Random seed control
 * - ModelCreator settings for lag/node/layer generation
 *
 * This struct travels as a read/write reference to trainers and builders.
 */
struct Config
{
    /**
     * @brief Number of realizations to run.
     * @details
     * Typically set to 1.
     * If >1, BuildAddresses() will generate multiple output paths.
     */
    int Realization;

    /**
     * @brief Total number of columns in the input dataset (inputs + outputs).
     * @details
     * Determined automatically in main based on ASM or Settling model.
     */
    double total_data_cols;

    /**
     * @brief Number of output columns.
     * @details
     * 1 for ASM models, 2 for settling models.
     */
    double number_of_outputs;

    /**
     * @brief Whether ASM model architecture is used.
     * @details
     * - true  → ASM model
     * - false → Simple settling model
     */
    bool ASM;

    /**
     * @brief Constituent name for model structure (e.g., "TKN", "NH", "sCOD", etc.)
     */
    std::string data_name;

    /**
     * @brief Whether to log-transform output internally.
     */
    bool log_output_d;

    /**
     * @brief Random seed for reproducibility.
     */
    double Seed_number;

    /**
     * @brief Whether K-fold is used.
     */
    bool kfold;

    /**
     * @brief Number of folds (e.g., 10 for 90/10 split).
     */
    int kfold_num;

    /**
     * @brief Split mode for K-fold.
     * @details
     * - 0 = random
     * - 1 = expanding window
     * - 2 = fixed window
     */
    int kfold_splitMode;

    /**
     * @brief Whether Genetic Algorithm is used.
     */
    bool GA_switch;

    /**
     * @brief Number of GA generations.
     */
    double GA_Nsim;

    /**
     * @brief Whether GA should optimize based on MSE-Test alone.
     */
    bool MSE_Test;

    /**
     * @brief Whether to use GA-optimized network structure.
     */
    bool optimized_structure;

    /**
     * @brief Whether random model structures are generated (instead of GA/manual).
     */
    bool randommodelstructure;

    /**
     * @brief Number of random model structure simulations when Random mode is enabled.
     */
    double Random_Nsim;

    /**
     * @brief Root project path for non-ASM models.
     */
    std::string path;

    /**
     * @brief Root project path for ASM models.
     */
    std::string path_ASM;

    /**
     * @brief Input data path for non-ASM datasets.
     */
    std::string datapath;

    /**
     * @brief Input data path for ASM datasets.
     */
    std::string datapath_ASM;

    /**
     * @brief Instance of ModelCreator, used for GA and random structure generation.
     */
    ModelCreator modelCreator;
};

/**
 * @brief Build all input/output address paths for training and test data.
 *
 * @details
 * Populates the following inside @c CModelStructure_Multi:
 * - trainaddress
 * - testaddress
 * - trainobservedaddress
 * - testobservedaddress
 * - trainpredictedaddress
 * - testpredictedaddress
 * - outputpath
 *
 * Handles both ASM and non-ASM path layouts.
 *
 * @param ms  Reference to model structure to update.
 * @param cfg Configuration parameters (read-only).
 *
 * @note Called once in main.cpp after BuildModelStructure().
 */
void BuildAddresses(CModelStructure_Multi& ms, const Config& cfg);
