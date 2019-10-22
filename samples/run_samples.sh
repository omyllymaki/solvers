#!/bin/sh

echo "Running samples"
for sample in "ls_solver" "gn_solver" "gd_solver" "ea_solver" "find_lr"
do
   echo "\n" 
   echo "SAMPLE: " $sample 
   ./$sample | grep "Function: int main"
done
echo "\n"
echo "Running samples finished"

