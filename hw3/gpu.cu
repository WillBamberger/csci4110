#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256
#define MAX_PARTICLES_PER_BIN 32

extern double size;
//
//  benchmarking program
//

__device__ __forceinline__ void apply_force_gpu(particle_t &particle, const particle_t &neighbor)  // CHANGE: pass by reference
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if (r2 > cutoff * cutoff)
      return;
   
  r2 = fmax(r2, min_r * min_r);
  double r = sqrt(r2);
  //
  //  very simple short-range repulsive force
  //
  double coef = (1 - cutoff / r) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* __restrict__ particles, int n, int* __restrict__ bins, int* __restrict__ bin_counts, double bin_size, int bins_per_row)
{
  // Get thread (particle) ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= n) return;

  particle_t &p = particles[tid];
  p.ax = p.ay = 0;

  int bin_x = p.x / bin_size;
  int bin_y = p.y / bin_size;
  if (bin_x < 0 || bin_x >= bins_per_row || bin_y < 0 || bin_y >= bins_per_row) return;

    int bin_index = bin_y * bins_per_row + bin_x;

    int idx = atomicAdd(&bin_counts[bin_index], 1);
    if (idx < MAX_PARTICLES_PER_BIN)
        bins[bin_index * MAX_PARTICLES_PER_BIN + idx] = tid;



  for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            int nbx = bin_x + dx;
            int nby = bin_y + dy;
            if (nbx < 0 || nbx >= bins_per_row || nby < 0 || nby >= bins_per_row)
                continue;

            int neighbor_bin = nby * bins_per_row + nbx;
            int count = bin_counts[neighbor_bin];

            for (int j = 0; j < count; j++)
            {
                int neighbor_idx = bins[neighbor_bin * MAX_PARTICLES_PER_BIN + j];
                if (neighbor_idx == tid) continue;
                apply_force_gpu(p, particles[neighbor_idx]);
            }
        }
    }
 
}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}

__global__ void reset_bin_counts(int* d_bin_counts, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_bins)
        d_bin_counts[tid] = 0;
}

int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaDeviceSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaDeviceSynchronize();
    double copy_time = read_timer( );
    
    double bin_size = cutoff;
    int bins_per_row = ceil(size / bin_size);
    int num_bins = bins_per_row * bins_per_row;

    int *d_bins;
    int *d_bin_counts;
    cudaMalloc(&d_bins, num_bins * MAX_PARTICLES_PER_BIN * sizeof(int));
    cudaMalloc(&d_bin_counts, num_bins * sizeof(int));

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    int bin_blks = (num_bins + NUM_THREADS - 1) / NUM_THREADS;

    cudaDeviceSynchronize();
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    cudaDeviceSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //
        reset_bin_counts<<<bin_blks, NUM_THREADS>>>(d_bin_counts, num_bins);
        compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n, d_bins, d_bin_counts, bin_size, bins_per_row);
        
        //
        //  move particles
        //
        move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
            // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
        }
    }
    cudaDeviceSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    cudaFree(d_bins);
    cudaFree(d_bin_counts);
    if( fsave )
        fclose( fsave );
    
    return 0;
}

