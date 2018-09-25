cp scripts/kuberry/White/script.lsf build-kepler35/f.lsf
cp scripts/kuberry/White/script.lsf build-kepler35/t.lsf
cp scripts/kuberry/White/script.lsf build-pascal60/g.lsf
cp scripts/kuberry/White/script.lsf build-cpu/cf.lsf
cp scripts/kuberry/White/script.lsf build-cpu/ct.lsf
cp scripts/kuberry/White/script.lsf build-cpu/cg.lsf
cd build-kepler35
qsub -q rhel7F f.lsf
qsub -q rhel7T t.lsf
cd ../build-pascal60
qsub -q rhel7G g.lsf
cd ../build-cpu
qsub -q rhel7F cf.lsf
qsub -q rhel7T ct.lsf
qsub -q rhel7G cg.lsf
cd ..
qstat

