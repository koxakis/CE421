#!/bin/bash
#Repeat for 12 times and store results in a .txt file
for ((i=12; i>0; i--))
do
./sobel_orig_2nd_test_set >> outfile.txt
done

exit 0
