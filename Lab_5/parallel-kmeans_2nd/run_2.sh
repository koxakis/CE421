#!/bin/bash
#Repeat for 12 times and store results in a .txt file
for ((i=12; i>0; i--))
do
echo s
./seq_main -o -b -n 4 -i Image_data/color17695.bin >> outfile_2th.txt
echo i4
./seq_main -o -b -n 10 -i Image_data/color17695.bin >> outfile_2th.txt
echo i10
./seq_main -o -b -n 100 -i Image_data/color17695.bin >> outfile_2th.txt
echo i100
./seq_main -o -b -n 1000 -i Image_data/color17695.bin >> outfile_2th.txt
echo i1000
./seq_main -o -b -n 10000 -i Image_data/color17695.bin >> outfile_2th.txt
echo i10000
./seq_main -o -b -n 15000 -i Image_data/color17695.bin >> outfile_2th.txt
echo i15000
./seq_main -o -b -n 4 -i Image_data/texture17695.bin >> outfile_2th.txt
echo i4
./seq_main -o -b -n 10 -i Image_data/texture17695.bin >> outfile_2th.txt
echo i10
./seq_main -o -b -n 100 -i Image_data/texture17695.bin >> outfile_2th.txt
echo i100
./seq_main -o -b -n 1000 -i Image_data/texture17695.bin >> outfile_2th.txt
echo i1000
./seq_main -o -b -n 10000 -i Image_data/texture17695.bin >> outfile_2th.txt
echo i10000
./seq_main -o -b -n 15000 -i Image_data/texture17695.bin >> outfile_2th.txt
echo i15000
./seq_main -o -b -n 4 -i Image_data/edge17695.bin >> outfile_2th.txt
echo i4
./seq_main -o -b -n 10 -i Image_data/edge17695.bin >> outfile_2th.txt
echo i10
./seq_main -o -b -n 100 -i Image_data/edge17695.bin >> outfile_2th.txt
echo i100
./seq_main -o -b -n 1000 -i Image_data/edge17695.bin >> outfile_2th.txt
echo i1000
./seq_main -o -b -n 10000 -i Image_data/edge17695.bin >> outfile_2th.txt
echo i10000
./seq_main -o -b -n 15000 -i Image_data/edge17695.bin >> outfile_2th.txt
echo e15000
done

exit 0
