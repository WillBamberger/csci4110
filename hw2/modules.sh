module purge
module unload intel
module load gcc/10.2.0
module load openmpi/4.0.5-gcc10.2.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/packages/openmpi/4.0.5-gcc10.2.0/lib

