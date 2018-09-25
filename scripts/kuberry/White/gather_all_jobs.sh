cat build-kepler35/f.lsf.o > output.txt
rm -rf build-kepler35/f.lsf.o
cat build-kepler35/t.lsf.o >> output.txt
rm -rf build-kepler35/t.lsf.o
cat build-pascal60/g.lsf.o >> output.txt
rm -rf build-pascal60/g.lsf.o

cat build-cpu/cf.lsf.o >> output.txt
rm -rf build-cpu/cf.lsf.o
cat build-cpu/cg.lsf.o >> output.txt
rm -rf build-cpu/cg.lsf.o
cat build-cpu/ct.lsf.o >> output.txt
rm -rf build-cpu/ct.lsf.o
cat output.txt

