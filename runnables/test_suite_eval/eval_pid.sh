#!/bin/bash

for level in {5..50}
do
  python pid.py --discr_level $level --output_folder "/shared/d/master/ws23/UAV-lab/git_repos/RL-pybullets-cf/checkpointed_models"
done