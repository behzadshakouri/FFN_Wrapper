TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

#DEFINES += GSL

CONFIG += Behzad
DEFINES += Behzad

SOURCES += \
        ffnwrapper.cpp \
        main.cpp

DEFINES += ARMA_USE_LAPACK ARMA_USE_BLAS

Behzad {
    OHQPATH = /home/behzad/Projects/OpenHydroQual/aquifolium

}



DEFINES += use_VTK ARMA_USE_SUPERLU
CONFIG += use_VTK

INCLUDEPATH += $$OHQPATH/include
INCLUDEPATH += $$OHQPATH/include/GA
INCLUDEPATH += $$OHQPATH/src

LIBS += -larmadillo -llapack -lblas -lgsl

HEADERS += \
    ffnwrapper.h
