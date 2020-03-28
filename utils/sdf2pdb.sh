#!/bin/sh

for filename in $1/*; do
    obabel $filename -O$2/"$(basename -- $filename)".pdb
done
