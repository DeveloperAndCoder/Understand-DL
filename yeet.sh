set -e
[ -z $1 ] && echo "Runnum missing" && exit 1
R=$1
echo $R
cd checkpoint/$R/combined
echo "Keeping frequency 5 only"
mkdir temp
mv *0_*.hdf5 temp/
mv *5_*.hdf5 temp/ #comment this line for 10 frequency
rm -f *.hdf5
mv temp/* ./
rmdir temp
ls
echo "Done"
