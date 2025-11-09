TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += core

# ---------------- Precompiled Header ----------------
CONFIG += precompile_header
PRECOMPILED_HEADER = pch.h
precompile_header:!isEmpty(PRECOMPILED_HEADER) {
    DEFINES += USING_PCH
}

# ---------------- System & Build Configurations ----------------
DEFINES += GSL
CONFIG  += PowerEdge
DEFINES += PowerEdge

DEFINES += MLPACK_ENABLE_ANN_SERIALIZATION

# Uncomment for other systems:
# CONFIG += Behzad
# DEFINES += Behzad
# CONFIG += Arash
# DEFINES += Arash

# ---------------- Compiler & Linker Flags ----------------
QMAKE_CXXFLAGS += -fopenmp -O2
QMAKE_LFLAGS   += -fopenmp

# ---------------- Project Paths ----------------
PowerEdge {
    OHQPATH = /mnt/3rd900/Projects/Utilities
}

Behzad {
    OHQPATH = /home/behzad/Projects/Utilities
}

Arash {
    OHQPATH = /home/arash/Projects/Utilities
}

# ---------------- Include Paths ----------------
INCLUDEPATH += $$OHQPATH
INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/include
INCLUDEPATH += $$HOME/Libraries/ensmallen/include
INCLUDEPATH += $$HOME/Libraries/mlpack/include
INCLUDEPATH += $$HOME/Libraries/armadillo/include
INCLUDEPATH += $$HOME/Libraries/boost/include

# ---------------- Libraries ----------------
LIBS += -lmlpack -larmadillo -llapack -lblas -lgsl \
        -lboost_filesystem -lboost_system -lboost_iostreams \
        -lgomp -lpthread

# ---------------- Numerical / VTK Support ----------------
DEFINES += ARMA_USE_LAPACK ARMA_USE_BLAS _ARMA
DEFINES += use_VTK ARMA_USE_SUPERLU
CONFIG  += use_VTK

# ---------------- Source Files ----------------
SOURCES += \
    $$OHQPATH/Utilities.cpp \
    cmodelstructure.cpp \
    cmodelstructure_multi.cpp \
    ffnwrapper.cpp \
    ffnwrapper_multi.cpp \
    modelcreator.cpp \
    main.cpp

# ---------------- Header Files ----------------
HEADERS += \
    ../Utilities/BTC.h \
    ../Utilities/BTC.hpp \
    ../Utilities/BTCSet.h \
    ../Utilities/BTCSet.hpp \
    Binary.h \
    CTransformation.h \
    ga.h \
    ga.hpp \
    individual.h \
    cmodelstructure.h \
    cmodelstructure_multi.h \
    ffnwrapper.h \
    ffnwrapper_multi.h \
    modelcreator.h \
    pch.h

# ---------------- Build Notes ----------------
# To build under Qt Creator:
# 1. Make sure libmlpack-dev, libarmadillo-dev, libboost-all-dev, libgsl-dev are installed.
# 2. On PowerEdge, ensure /mnt/3rd900/Projects/Utilities exists and contains Utilities.cpp/h.
# 3. For release build, enable optimization:
# CONFIG += release
