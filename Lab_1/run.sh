#!/bin/bash

for ((i=12; i>0; i--))
do
./sobel_orig >> outfile.txt
done

exit 0
