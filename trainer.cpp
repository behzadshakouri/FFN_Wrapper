#include "trainer.h"
#include "ga.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>

void RunGA(CModelStructure_Multi& ms, Config& cfg)
{
    GeneticAlgorithm<ModelCreator> GA;
    GA.Settings.generations = cfg.GA_Nsim;
    GA.Settings.MSE_optimization = cfg.MSE_Test;
    GA.Settings.outputpath = ms.outputpath;
    GA.model = cfg.modelCreator;
    GA.model.FFN.ModelStructure = ms;

    auto Optimized = GA.Optimize();
    Optimized.FFN.silent = false;

    Optimized.FFN.DataSave(datacategory::Train);
    Optimized.FFN.DataSave(datacategory::Test);

    QFile f(QString::fromStdString(ms.outputpath + "GA_results.txt"));
    if (f.open(QIODevice::WriteOnly | QIODevice::Text))
        QTextStream(&f) << "GA completed\n";
}

void RunRandom(CModelStructure_Multi& ms, Config& cfg)
{
    std::ofstream file(cfg.datapath_ASM + "Results/RMS_Output.txt");

    for (int i = 0; i < cfg.Random_Nsim; i++)
    {
        cfg.modelCreator.CreateRandomModelStructure(&ms);

        if (!ms.ValidLags()) { i--; continue; }

        FFNWrapper_Multi F;
        F.silent = false;
        F.ModelStructure = ms;
        F.Initiate();

        if (!cfg.kfold)
            F.Train();
        else
            F.Train_kfold(cfg.kfold_num, cfg.kfold_splitMode);

        F.Test();
        F.PerformanceMetrics();

        F.DataSave(datacategory::Train);
        F.DataSave(datacategory::Test);

        file << F.ModelStructure.ParametersToString().toStdString() << "\n";
    }
}

void RunSingle(CModelStructure_Multi& ms, Config& cfg)
{
    FFNWrapper_Multi F;
    F.silent = false;
    F.ModelStructure = ms;
    F.Initiate();

    if (!cfg.kfold)
        F.Train();
    else
        F.Train_kfold(cfg.kfold_num, cfg.kfold_splitMode);

    F.Test();
    F.PerformanceMetrics();

    F.DataSave(datacategory::Train);
    F.DataSave(datacategory::Test);

    F.Plotter();
}
