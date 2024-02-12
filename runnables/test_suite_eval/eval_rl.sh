#!/bin/bash

for level in {5..50}
do
  python pid.py --discr_level $level
done