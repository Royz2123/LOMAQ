#!/usr/bin/env bash

# https://wiki.cs.huji.ac.il/wiki/Connecting_Remotely

# river
ssh roy_zohar%river@gw.cs.huji.ac.il

# general (phoenix)
ssh roy_zohar%phoenix@gw.cs.huji.ac.il

# Make script executable
chmod +x scripts/linux/run_multi_cart.sh

# Run script interactively
srun --mem=400m -c4 --time=2:0:0 "scripts/linux/run_multi_cart.sh"

# Run script regularly
sbatch --mem=400m -c4 --time=2:0:0 "scripts/linux/run_multi_cart.sh"
