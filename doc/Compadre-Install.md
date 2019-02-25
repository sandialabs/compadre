# Installing the Compadre Toolkit
 
## Compadre Toolkit Installation Instructions

  1.) Copy one of the examples bash script files from ./scripts to the folder where you would like to build the project.
      Ideally, this should not be at the root of the repository as it is never suggested to build in-source.
  
  example:

```
  >> mkdir build
  >> cp ./scripts/script_you_choose.sh build
```
  
  2.) Edit the script file that you copied to your soon-to-be build folder.
      Make changes to these files to suit your needs (KokkosCore_PREFIX, etc...)
      Then, run the modified bash script file.
  
  (assumes you moved the script to ./build as in #1 )
  
```
  >> cd ./build
  >> vi script_you_choose.sh
```
  (make any changes and save, [Python Interface and Matlab examples](Python-Interface-Install.md))
  
```
  >> ./script_you_choose.sh
```
      
  3.) Build the project.
  
```
  >> make -j4                      # if you want to build using 4 processors
  >> make install
```
  
  4.) Test the built project by exercising the suite of tests.
  
```
  >> ctest
```

  5.) Build doxygen documentation for the project by executing

```
  >> make Doxygen
  >> [your favorite internet browser executable] doc/output/html/index.html
```

   
  If some tests fail, be sure to check the error as it is possible that you have not configured CMake
  as to where it should locate libraries like Kokkos, Python, etc...
  If a library is missing but not turned on in the CMake options, then the test will simply fail.
  
## Importing Project Into Eclipse

From https://stackoverflow.com/questions/11645575/importing-a-cmake-project-into-eclipse-cdt,
the instructions for importing from CMake into eclipse are as follows:

> First, choose a directory for the CMake files. I prefer to keep my Eclipse workspaces in 
> ~/workspaces and the source code in ~/src. Data which I need to build or test the project 
> goes in subdirs of the project's workspace dir, so I suggest doing the same for CMake.
> 
> Assuming both your workspace and source folders are named someproject, do:
> 
> cd ~/workspaces/someproject
> mkdir cmake
> cd cmake
> cmake -G "Eclipse CDT4 - Unix Makefiles" ~/src/someproject
> 
> Then, in your Eclipse workspace, do:
> 
> File > Import... > General > Existing Projects into Workspace
> 
> Check Select root directory and choose ~/workspaces/someproject/cmake. Make sure Copy projects into workspace is NOT checked.
> 
> Click Finish and you have a CMake project in your workspace.
> 
> Two things to note:
> 
>   I used cmake for the workspace subdir, but you can use a name of your choice.
>   If you make any changes to your build configuration (such as editing Makefile.am), you will need to re-run the 
>   last command in order for Eclipse to pick up the changes.

