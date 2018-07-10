#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>
#include <math.h>
#include <new>
#include <unistd.h>

#include <AMReX_Box.H>
#include <AMReX_DataServices.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <AMReX_MultiFabUtil.H>

// These are for SWFFT
#include <Distribution.H>
#include <AlignedAllocator.h>
#include <Dfft.H>

#define ALIGN 16
#define XDIR 0
#define YDIR 1
#define ZDIR 2
#define DIM  3

using namespace amrex;

int  nProcs;
int  myProc;
int  IOProc;

int  verbose;

int  nVars;

int  transpose_dp;

int  BLix, BLjx, BLkx;
int  FTix, FTjx, FTkx;
int  FTis, FTjs, FTks;
int  FThkxpo;
int  wavenumbers;

int  Qix, Qjx, Qkx;
Vector<Real*> spectrum;
Vector<Real*> spectrumS;
Vector<Real*> spectrumC;
Vector<Real*> Qx, Qy, Qz;
Real *sum, *sum2;

// Box probDomain;
Vector<Real> probLo;
Vector<Real> probHi;
Real Lx, Ly, Lz;
Real dx, dy, dz;

std::string infile;
Vector<std::string> whichVar;
Real Time;
int  timeSteps;

void Spectra(MultiFab &mf, Vector<Geometry> &geoms, BoxArray& ba);


//
// Usage note:
//
// FFTW uses a slab decomposition and decides how to arrange the data
// Suppose there are N cells in the k direction and P processors
// The domain is decomposed into P slabs with N/P cells
// It is therefore sensible to choose P such that P divides N
// P cannot exceed N, and my experience suggests P=N/4 is a good choice
//
// Takes an input file to define the following variables:
//
// verbose = 0 or 1
// infile = [list of input plotfiles to use]
// vars = [list of variables to load]
// transpose_dp = 0 or 1
// finestLevel = finest level to use
//

int main (int argc, char* argv[])
{
    //
    // Initialize
    //

    amrex::Initialize(argc,argv);

    nProcs = ParallelDescriptor::NProcs();
    myProc = ParallelDescriptor::MyProc();
    IOProc = ParallelDescriptor::IOProcessorNumber();

    //
    // Read input data
    //
    ParmParse pp;

    if (ParallelDescriptor::IOProcessor()) 
        std::cout << "getting started" << std::endl;

    verbose=0;
    pp.query("verbose",verbose);
    if (ParallelDescriptor::IOProcessor()) 
        std::cout << "setting verbose = " << verbose << std::endl;

    int nPlotFiles(pp.countval("infile"));
    if(nPlotFiles <= 0) {
        std::cerr << "Bad nPlotFiles:  " << nPlotFiles << std::endl;
        std::cerr << "Exiting." << std::endl;
        DataServices::Dispatch(DataServices::ExitRequest, NULL);
    }

    // Make an array of srings containing paths of input plot files
    Vector<std::string> plotFileNames(nPlotFiles);
    for(int iPlot = 0; iPlot < nPlotFiles; ++iPlot) {
        pp.get("infile", plotFileNames[iPlot], iPlot);
    }
    if (ParallelDescriptor::IOProcessor()){ 
        std::cout << "number of plotfiles  = " << nPlotFiles << std::endl;
        std::cout << "first infile = " << plotFileNames[0] << std::endl;
    }

    nVars=pp.countval("vars");
    if (nVars==0)
        amrex::Abort("Must specify vars to load");
    if (ParallelDescriptor::IOProcessor())
        std::cout << "nVars = " << nVars << std::endl;

    whichVar.resize(nVars);
    if (ParallelDescriptor::IOProcessor())
        std::cout << "vars = ";

    for (int i=0; i<nVars; i++) {
        pp.get("vars",whichVar[i],i);
        if (ParallelDescriptor::IOProcessor())
            std::cout << " " << whichVar[i];
    }

    if (ParallelDescriptor::IOProcessor())
        std::cout << std::endl;

    Vector<int> destFills(nVars);
    for (int c=0; c<nVars; c++) destFills[c] = c;

    //size arrays for holding data
    sum  = (Real*) malloc(sizeof(Real)*nVars);
    sum2 = (Real*) malloc(sizeof(Real)*nVars);

    spectrum.resize(nVars+1);

    Qx.resize(nVars);
    Qy.resize(nVars);
    Qz.resize(nVars);

    if (ParallelDescriptor::IOProcessor())
        std::cout << std::endl;

    //
    // Read plot file info
    //
    DataServices::SetBatchMode();
    Amrvis::FileType fileType(Amrvis::NEWPLT);

    for (int iPlot=0; iPlot<nPlotFiles; iPlot++) {
        // initialize sum to zero
        for (int iVar=0; iVar<nVars; iVar++)
            sum[iVar]=sum2[iVar]=0.;

        infile=plotFileNames[iPlot];
        if (ParallelDescriptor::IOProcessor())
            std::cout << "working on " << plotFileNames[iPlot] << std::endl;

        DataServices *dataServices = new DataServices(infile, fileType);
        if( ! dataServices->AmrDataOk())
            DataServices::Dispatch(DataServices::ExitRequest, NULL);

        AmrData amrData(dataServices->AmrDataRef());

        Time      = amrData.Time();
        timeSteps = amrData.LevelSteps()[0];

        int finestLevel = amrData.FinestLevel();
        int finestLevelIn(-1);
        pp.query("finestLevel",finestLevelIn);
        if (finestLevelIn>=0 && finestLevelIn<finestLevel) {
            finestLevel=finestLevelIn;
        }

        if (ParallelDescriptor::IOProcessor())
            std::cout << "Using finestLevel = " << finestLevel << std::endl;

        Box probDomain(amrData.ProbDomain()[finestLevel]);

        // Set AMReX and FourierTranform array sizes
        // Note this defaults to a transposition
        BLix = FTkx = probDomain.length(XDIR);
        BLjx = FTjx = probDomain.length(YDIR);
        BLkx = FTix = probDomain.length(ZDIR);

        // Half kx+1 - accounts for the fftw padding
        FThkxpo = FTkx/2+1;

        // Number of wavenumbers in the spectra
        wavenumbers = FTix/2;
        if (FTjx/2 < wavenumbers) wavenumbers = FTjx/2;
        if (FTkx/2 < wavenumbers) wavenumbers = FTkx/2;

        // Size of correlation functions
        Qix = BLix/2;
        Qjx = BLjx/2;
        Qkx = BLkx/2;

        // Declare memory for spectra (plus one for counting hits)
        for (int iVar=0; iVar<=nVars; iVar++) {
            spectrum[iVar]=(Real*)malloc(sizeof(Real)*wavenumbers);
            for (int wn=0; wn<wavenumbers; wn++)
                spectrum[iVar][wn] = 0.0;
        }

        // Declare memory for correlation functions
        // Qx, Qy and Qz are the correlations in the three directions
        for (int iVar=0; iVar<nVars; iVar++) {
            Qx[iVar]=(Real*)malloc(sizeof(Real)*Qix);
            for (int i=0; i<Qix; i++)
                Qx[iVar][i] = 0.0;
            Qy[iVar]=(Real*)malloc(sizeof(Real)*Qjx);
            for (int j=0; j<Qjx; j++)
                Qy[iVar][j] = 0.0;
            Qz[iVar]=(Real*)malloc(sizeof(Real)*Qkx);
            for (int k=0; k<Qkx; k++)    Qz[iVar][k] = 0.0;
        }

        // Other AMReX stuff
        probLo=amrData.ProbLo();
        probHi=amrData.ProbHi();

        Lx = probHi[0]-probLo[0];
        Ly = probHi[0]-probLo[0];
        Lz = probHi[0]-probLo[0];

        dx = Lx/(Real)BLix;
        dy = Ly/(Real)BLjx;
        dz = Lz/(Real)BLkx;

        Real dxyz=dx*dy*dz;

        //
        // Load plot file into data structure
        // 

        int ngrow = 0;
        BoxArray ba = amrData.boxArray(finestLevel);
        // Max size?
        DistributionMapping dm {ba};

        MultiFab mf;
        mf.define(ba, dm, nVars, ngrow, MFInfo().SetAlloc(true));

        Vector<Geometry> geoms(finestLevel + 1);
        {
            Vector<Box> domains = amrData.ProbDomain();
            Vector<Real> probSizes = amrData.ProbSize();
            Vector<Real> ProbLo = amrData.ProbLo();
            Vector<Real> ProbHi = amrData.ProbHi();
            auto rbox = RealBox(
                    {AMREX_D_DECL(ProbLo[0], ProbLo[1], ProbLo[2])},
                    {AMREX_D_DECL(ProbHi[0], ProbHi[1], ProbHi[2])}
                    );
            std::array<int,AMREX_SPACEDIM> is_periodic {AMREX_D_DECL(1, 1, 1)};
            for (int iLevel = 0; iLevel <= finestLevel; ++iLevel)
                geoms[iLevel] = Geometry(
                        domains[iLevel],
                        &rbox,
                        0,  // cartesian coords
                        is_periodic.data()
                        );
        }

        amrData.FillVar(mf, finestLevel, whichVar, destFills);

        //
        // Evaluate fft
        //
        Spectra(mf, geoms, ba);

        if (ParallelDescriptor::IOProcessor()) {
            std::string suffix;
            suffix = "";

            std::cout << "Outputting to file..." << std::endl;
            for (int iVar=0; iVar<=nVars; iVar++) {
                std::string outfile;
                if (iVar==nVars)
                    outfile = infile + "/spectrum_count" + suffix;
                else
                    outfile = infile + "/" + whichVar[iVar] + "_spectrum" + suffix;
                FILE* file=fopen(outfile.c_str(),"w");
                for (int wn=0; wn<wavenumbers; wn++)
                    fprintf(file,"%i %e\n",wn,spectrum[iVar][wn]);
                fclose(file);
            }
            for (int iVar=0; iVar<nVars; iVar++) {
                std::string outfile;
                FILE* file;

                // Integrals
                outfile = infile + "/" + whichVar[iVar] + "_Int" + suffix;
                file=fopen(outfile.c_str(),"w");
                fprintf(file,"%e %e %e\n",Time,sum[iVar]*dxyz,sum2[iVar]*dxyz);
                fclose(file);

                // Qx
                outfile = infile + "/" + whichVar[iVar] + "_Qx" + suffix;
                file=fopen(outfile.c_str(),"w");
                for (int i=0; i<Qix; i++)
                    fprintf(file,"%e %e\n",dx*(0.5+(Real)i),Qx[iVar][i]);
                fclose(file);

                // Qy
                outfile = infile + "/" + whichVar[iVar] + "_Qy" + suffix;
                file=fopen(outfile.c_str(),"w");
                for (int i=0; i<Qjx; i++)
                    fprintf(file,"%e %e\n",dy*(0.5+(Real)i),Qy[iVar][i]);
                fclose(file);

                // Qz
                outfile = infile + "/" + whichVar[iVar] + "_Qz" + suffix;
                file=fopen(outfile.c_str(),"w");
                for (int i=0; i<Qkx; i++)
                    fprintf(file,"%e %e\n",dz*(0.5+(Real)i),Qz[iVar][i]);
                fclose(file);
            }
            std::cout << "   ...done." << std::endl;
        }

        for (int iVar=0; iVar<nVars; iVar++) {
            //free(local_data[iVar]);
            //free(Qx[iVar]);
            //free(Qy[iVar]);
            //free(Qz[iVar]);
            //free(spectrum[iVar]);
        }
    } // iPlot

    amrex::Finalize();
}





void Spectra(MultiFab &mf, Vector<Geometry>& geoms, BoxArray& ba)
{  
    //
    // Populate fft data
    //
    if (ParallelDescriptor::IOProcessor())
        std::cout << "Populating fft data..." << std::endl;

    const DistributionMapping& dm = mf.DistributionMap();

    if (mf.nGrow() != 0) 
        amrex::Error("Current implementation requires that mf has no ghost cells");

    // We assume that all grids have the same size hence 
    // we have the same mfix,mfjx,mfkx on all ranks
    int mfix = ba[0].size()[0];
    int mfjx = ba[0].size()[1];
    int mfkx = ba[0].size()[2];

    Box domain(geoms[0].Domain());

    int nbx = domain.length(0) / mfix;
    int nby = domain.length(1) / mfjx;
    int nbz = domain.length(2) / mfkx;
    int nboxes = nbx * nby * nbz;
    if (nboxes != ba.size()) 
        amrex::Error("NBOXES NOT COMPUTED CORRECTLY");
    amrex::Print() << "Number of boxes:\t" << nboxes << std::endl;


    Vector<int> rank_mapping;
    rank_mapping.resize(nboxes);

    for (int ib = 0; ib < nboxes; ++ib)
    {
        int i = ba[ib].smallEnd(0) / mfix;
        int j = ba[ib].smallEnd(1) / mfjx;
        int k = ba[ib].smallEnd(2) / mfkx;

        // This would be the "correct" local index if the data wasn't being transformed
        // int local_index = k*nbx*nby + j*nbx + i;

        // This is what we pass to dfft to compensate for the Fortran ordering
        //      of amrex data in MultiFabs.
        int local_index = i*nby*nbz + j*nbz + k;

        rank_mapping[local_index] = dm[ib];
        if (verbose)
            amrex::Print() << "LOADING RANK NUMBER " << dm[ib] << " FOR GRID NUMBER " << ib 
                << " WHICH IS LOCAL NUMBER " << local_index << std::endl;
    }

    Real h = geoms[0].CellSize(0);
    Real hsq = h*h;

    // Assume for now that mfix = mfjx = mfkx
    int Ndims[3] = { nbz, nby, nbx };
    int     n[3] = {domain.length(2), domain.length(1), domain.length(0)};
    hacc::Distribution d(MPI_COMM_WORLD,n,Ndims,&rank_mapping[0]);
    hacc::Dfft dfft(d);







    for (MFIter mfi(mf,false); mfi.isValid(); ++mfi)
    {
        int gid = mfi.index();

        size_t local_size  = dfft.local_size();

        std::vector<complex_t, hacc::AlignedAllocator<complex_t, ALIGN> > a;
        std::vector<complex_t, hacc::AlignedAllocator<complex_t, ALIGN> > b;

        std::cout << mfix << std::endl;
        std::cout << mfjx << std::endl;
        std::cout << mfkx << std::endl;
        a.resize(mfix*mfjx*mfkx);
        b.resize(mfix*mfjx*mfkx);

        if (ParallelDescriptor::IOProcessor())
            std::cout << "Making SWFFT plan..." << std::endl;
        dfft.makePlans(&a[0],&b[0],&a[0],&b[0]);
        ParallelDescriptor::Barrier();
        if (ParallelDescriptor::IOProcessor())
            std::cout << "SWFFT plan done." << std::endl;

        // *******************************************
        // Copy real data from Rhs into real part of a -- no ghost cells and
        // put into C++ ordering (not Fortran)
        // *******************************************
        if (ParallelDescriptor::IOProcessor())
            std::cout << "Copying data to C++ ordering..." << std::endl;
        complex_t zero(0.0, 0.0);
        size_t local_indx = 0;
        for(size_t k=0; k<(size_t)mfkx; k++) {
            for(size_t j=0; j<(size_t)mfjx; j++) {
                for(size_t i=0; i<(size_t)mfix; i++) {

                    complex_t temp(mf[mfi].dataPtr()[local_indx],0.);
                    a[local_indx] = temp;
                    local_indx++;

                }
            }
        }
        ParallelDescriptor::Barrier();
        if (ParallelDescriptor::IOProcessor())
            std::cout << "Done copying." << std::endl;

        //  *******************************************
        //  Compute the forward transform
        //  *******************************************
        if (ParallelDescriptor::IOProcessor())
            std::cout << "Calling FFTs..." << std::endl;
        dfft.forward(&a[0]);
        ParallelDescriptor::Barrier();
        if (ParallelDescriptor::IOProcessor())
            std::cout << "Done with FFT." << std::endl;

        //  *******************************************
        //  Normalize and bin values
        //  *******************************************
        if (ParallelDescriptor::IOProcessor())
            std::cout << "Normalizing and binning spectrum." << std::endl;

        // Divisor to normalise transform
        Real div = ((Real)FTix)*((Real)FTjx)*((Real)FTkx);

        size_t global_size  = dfft.global_size();
        double fac = hsq / global_size;

        local_indx = 0;
        for(size_t k=0; k<(size_t)mfkx; k++) {
            for(size_t j=0; j<(size_t)mfjx; j++) {
                for(size_t i=0; i<(size_t)mfix; i++) {

                    // Divide by 2 pi N
                    mf[mfi].dataPtr()[local_indx] = fac * std::real(a[local_indx]);
                    local_indx++;

                    // for binning
                    int wn = (int) (0.5+sqrt((Real)(i*i+j*j+k*k)));

                    int ccell = (j * FTix + i) * FThkxpo + k;

                    for (int iVar=0; iVar<nVars; iVar++) {
                        Real re = a[local_indx].real() / div;
                        Real im = a[local_indx].imag() / div;
                        Real sq = re*re + im*im;
                        if (wn<wavenumbers)
                            spectrum[iVar][wn] += 0.5 * sq;
                    }
                    // Let's count the number of hits
                    if (wn<wavenumbers)
                        spectrum[nVars][wn] += 1;
                }
            }
        }
        ParallelDescriptor::Barrier();
        if (ParallelDescriptor::IOProcessor())
            std::cout << "Done binning." << std::endl;

        for (int iVar=0; iVar<=nVars; iVar++)
            ParallelDescriptor::ReduceRealSum(spectrum[iVar],wavenumbers,IOProc);
    }
}



