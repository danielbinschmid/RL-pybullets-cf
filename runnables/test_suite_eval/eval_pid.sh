#!/bin/bash

for level in {5..50}
do
  python pid.py --discr_level $level --eval_set "--eval_set "./test_suite_eval/eval-v0_n-ctrl-points-3_n-tracks-1000_2024-02-12_10:59:21_d689e67e-4179-4764-a474-e5f3237a239d"
done