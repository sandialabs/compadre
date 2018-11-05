\section installing_sec Installing the Compadre Toolkit
 
\subsection installing_subsec Compadre Toolkit Installation Instructions

  1.) Copy one of the examples bash script files from ./scripts to the folder where you would like to build the project.
      Ideally, this should not be at the root of the repository as it is never suggested to build in-source.
  
  example:

\verbatim
  >> mkdir build
  >> cp ./scripts/script_you_choose.sh build
\endverbatim
  
  2.) Edit the script file that you copied to your soon-to-be build folder.
      Make changes to these files to suit your needs (Trilinos location, etc...)
      Then, run the modified bash script file.
  
  (assumes you moved the script to ./build as in #1 )
  
\verbatim
  >> cd ./build
  >> vi script_you_choose.sh
\endverbatim
  (make any changes and save)
  
\verbatim
  >> ./script_you_choose.sh
\endverbatim
      
  3.) Build the project.
  
\verbatim
  >> make -j4                      # if you want to build using 4 processors
  >> make install
\endverbatim
  
  4.) Test the built project by exercising the suite of tests.
  
\verbatim
  >> ctest
\endverbatim

  5.) Build doxygen documentation for the project by executing

\verbatim
  >> make Doxygen
  >> [your favorite internet browser executable] doc/output/html/index.html
\endverbatim

   
  If some tests fail, be sure to check the error as it is possible that you have not configured CMake
  as to where it should locate libraries like Netcdf, VTK, Trilinos, etc...
  If a library is missing but not turned on in the CMake options, then the test will simply fail.


\subsection importing_eclipse_subsec Importing Project Into Eclipse

__From https://stackoverflow.com/questions/11645575/importing-a-cmake-project-into-eclipse-cdt,
the instructions for importing from CMake into eclipse are as follows:

\verbatim
 First, choose a directory for the CMake files. I prefer to keep my Eclipse workspaces in 
 ~/workspaces and the source code in ~/src. Data which I need to build or test the project 
 goes in subdirs of the project's workspace dir, so I suggest doing the same for CMake.
 
 Assuming both your workspace and source folders are named someproject, do:
 
 cd ~/workspaces/someproject
 mkdir cmake
 cd cmake
 cmake -G "Eclipse CDT4 - Unix Makefiles" ~/src/someproject
 
 Then, in your Eclipse workspace, do:
 
 File > Import... > General > Existing Projects into Workspace
 
 Check Select root directory and choose ~/workspaces/someproject/cmake. Make sure Copy projects into workspace is NOT checked.
 
 Click Finish and you have a CMake project in your workspace.
 
 Two things to note:
 
   I used cmake for the workspace subdir, but you can use a name of your choice.
   If you make any changes to your build configuration (such as editing Makefile.am), you will need to re-run the 
   last command in order for Eclipse to pick up the changes.

\endverbatim


\subsection extra_examples_subsec Extra examples demonstrating linking to the Compadre Toolkit library externally

The folder ./extern_example has a build directory with a my-do-configure-cpu.sh script already created inside.
This example demonstrates how one can go about linking to the installed Compadre Toolkit from another application.


\mainpage Compadre Toolkit

\section About

The Compadre Toolkit provides a framework for meshless remap and PDE solution. It allows users to harness meshless discretizations such as Generalized Moving Least Squares (GMLS), while executing these parallel communication-sparse, computationally-dense kernels on modern architectures. 

\subsection Generalized Moving Least Squares (GMLS)

A GMLS problem requires the specification of a target functional \f$\tau\f$ (Compadre::TargetOperation), a reconstruction space \f$V\f$ (Compadre::ReconstructionSpace), and a sampling functional \f$\lambda\f$ (Compadre::SamplingFunctional).

The Compadre Toolkit is designed to efficiently assemble, factorize, and solve large batches of minimization problems having the form:

<center>
\f$p^{*} = \underset{p \in V}{\text{arg min}}\;\frac{1}{2}\sum_{j=1}^N (\lambda_j(u)-\lambda_j(p))^{2}\omega(\tau;\lambda_j).\f$

\f$\tau(u) \approx \tau(p^{*})\f$
</center>

\section Recent Changes

<a href="md__home_pakuber_Desktop_ProjectStubs_Particles_compadre_repo_Changelog.html"> See recent changes</a>.

\section Toolkit Installation

<a href="md__home_pakuber_Desktop_ProjectStubs_Particles_compadre_repo_README.html"> See installation instructions</a>.

