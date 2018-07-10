AMREX_HOME = ../amrex

DIM = 3

USE_MPI = TRUE

DEBUG = TRUE

COMP = gcc

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/Extern/amrdata/Make.package

INCLUDE_LOCATIONS += $(AMREX_HOME)/Src/Extern/amrdata
VPATH_LOCATIONS   += $(AMREX_HOME)/Src/Extern/amrdata

include $(AMREX_HOME)/Src/Extern/SWFFT/Make.package
INCLUDE_LOCATIONS	+= $(AMREX_HOME)/Src/Extern/SWFFT
VPATH_LOCATIONS		+= $(AMREX_HOME)/Src/Extern/SWFFT

LIBRARIES += -L$(FFTW_DIR) -lfftw3_mpi -lfftw3_omp -lfftw3
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
