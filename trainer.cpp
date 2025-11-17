/**
 * @file trainer.cpp
 * @brief Implements all training modes for the FFN Wrapper:
 *        Genetic Algorithm (GA), Random Structure Search, and Single-Model Training.
 *
 * @details
 * This module contains the training logic for the three possible modes selected in main.cpp:
 *
 * 1. **RunGA()**
 *    - Performs Genetic Algorithm optimization using ModelCreator
 *    - Optimizes model structure based on MSE-Test (or other GA settings)
 *    - Saves results to disk
 *
 * 2. **RunRandom()**
 *    - Generates many random neural network structures
 *    - Trains and evaluates each one
 *    - Saves outputs and structure strings to RMS_Output.txt
 *
 * 3. **RunSingle()**
 *    - Runs a single, deterministic model structure
 *    - Performs training, testing, metrics, and plotting
 *
 * These functions keep the main pipeline simple and modular, while storing all
 * architecture logic in BuildModelStructure() and all path logic in BuildAddresses().
 *
 * @see trainer.h
 * @see BuildModelStructure()
 * @see BuildAddresses()
 */

#include "trainer.h"
#include "ga.h"

#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <fstream>

/**
 * @brief Execute Genetic Algorithm (GA) optimization for model structure.
 *
 * @details
 * Steps performed:
 *
 * 1. Initialize GA with:
 *    - Number of generations: @c cfg.GA_Nsim
 *    - Objective: MSE-Test (if cfg.MSE_Test = true)
 *    - ModelCreator instance from @c cfg.modelCreator
 *
 * 2. Run GA:
 *    - GA internally tests network structures using FFNWrapper_Multi
 *    - The best model structure is selected and returned
 *
 * 3. Save Outputs:
 *    - Train and Test predictions
 *    - GA results file: "GA_results.txt"
 *
 * @param ms   Model structure used by the GA and updated inside GA.Model.
 * @param cfg  Configuration (non-const because GA modifies ModelCreator state).
 *
 * @note
 * The ModelCreator in cfg stores the FFNWrapper instance and must remain mutable.
 */
void RunGA(CModelStructure_Multi& ms, Config& cfg)
{
    GeneticAlgorithm<ModelCreator> GA;

    // Configure GA settings
    GA.Settings.generations       = cfg.GA_Nsim;
    GA.Settings.MSE_optimization  = cfg.MSE_Test;
    GA.Settings.outputpath        = ms.outputpath;

    // Assign model creator
    GA.model = cfg.modelCreator;

    // GA needs full access to ModelStructure
    GA.model.FFN.ModelStructure = ms;

    // Run optimization
    auto OptimizedModel = GA.Optimize();
    OptimizedModel.FFN.silent = false;

    // Save results
    OptimizedModel.FFN.DataSave(datacategory::Train);
    OptimizedModel.FFN.DataSave(datacategory::Test);

    // Write GA output file
    QFile file(QString::fromStdString(ms.outputpath + "GA_results.txt"));
    if (file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QTextStream out(&file);
        out << "GA optimization completed successfully.\n";
        out << "Best structure: "
            << OptimizedModel.FFN.ModelStructure.ParametersToString();
    }
    else
    {
        qWarning() << "Could not open GA_results.txt for writing.";
    }
}

/**
 * @brief Perform random model structure search.
 *
 * @details
 * This function generates a large number of **random neural network architectures**
 * using @c cfg.modelCreator.CreateRandomModelStructure().
 *
 * For each random structure:
 * - Validate lag consistency
 * - Train FFNWrapper_Multi
 * - Perform Test()
 * - Compute performance metrics
 * - Save Train/Test predictions
 * - Append structure details to RMS_Output.txt
 *
 * ### Output File:
 * Written to:
 *   @c cfg.datapath_ASM + "Results/RMS_Output.txt"
 *
 * Each line contains the full parameter string of the model structure.
 *
 * @param ms   The model structure instance reused for each random trial.
 * @param cfg  Configuration (ModelCreator is mutated, thus passed non-const).
 *
 * @warning
 * Random search can be computationally expensive when Random_Nsim is large.
 */
void RunRandom(CModelStructure_Multi& ms, Config& cfg)
{
    std::string outpath = cfg.datapath_ASM + "Results/RMS_Output.txt";
    std::ofstream file(outpath);

    if (!file.is_open())
    {
        qWarning() << "Could not open RMS_Output.txt for writing.";
        return;
    }

    // Iterate through random simulations
    for (int i = 0; i < cfg.Random_Nsim; i++)
    {
        // Generate random architecture
        cfg.modelCreator.CreateRandomModelStructure(&ms);

        // Reject structure if lag structure is invalid
        if (!ms.ValidLags())
        {
            i--;
            continue;
        }

        // Prepare FFN wrapper
        FFNWrapper_Multi F;
        F.silent = false;
        F.ModelStructure = ms;
        F.Initiate();

        // Perform training
        if (!cfg.kfold)
            F.Train();
        else
            F.Train_kfold(cfg.kfold_num, cfg.kfold_splitMode);

        // Evaluate performance
        F.Test();
        F.PerformanceMetrics();

        // Save data
        F.DataSave(datacategory::Train);
        F.DataSave(datacategory::Test);

        // Write structure summary
        file << F.ModelStructure.ParametersToString().toStdString() << "\n";
    }
}

/**
 * @brief Train and evaluate a single model structure.
 *
 * @details
 * This is the most commonly used workflow when:
 * - GA is disabled (`cfg.GA_switch = false`)
 * - Random search is disabled (`cfg.randommodelstructure = false`)
 *
 * Steps:
 * 1. Initialize FFNWrapper_Multi with @c ms (already built by BuildModelStructure)
 * 2. Train using Train() or Train_kfold()
 * 3. Evaluate using Test()
 * 4. Compute metrics via PerformanceMetrics()
 * 5. Save train/test data
 * 6. Generate plots via Plotter()
 *
 * This mode is deterministic and ideal for production runs once architecture is fixed.
 *
 * @param ms   Fully built model structure.
 * @param cfg  Global configuration including K-fold flags and datapaths.
 */
void RunSingle(CModelStructure_Multi& ms, Config& cfg)
{
    FFNWrapper_Multi F;
    F.silent = false;
    F.ModelStructure = ms;
    F.Initiate();

    // Train using either normal or k-fold mode
    if (!cfg.kfold)
        F.Train();
    else
        F.Train_kfold(cfg.kfold_num, cfg.kfold_splitMode);

    // Evaluate
    F.Test();
    F.PerformanceMetrics();

    // Save train/test output
    F.DataSave(datacategory::Train);
    F.DataSave(datacategory::Test);

    // Create plots
    F.Plotter();
}
