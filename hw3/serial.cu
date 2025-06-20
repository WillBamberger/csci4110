#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"

#define MAX_PARTICLES_PER_BIN 64
extern double size;

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
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
    set_size( n );
    init_particles( n, particles );
    
    double bin_size = cutoff;
    int bins_per_row = ceil(size / bin_size);
    int num_bins = bins_per_row * bins_per_row;

    particle_t **bins = (particle_t**) malloc(num_bins * sizeof(particle_t*));
    int *bin_counts = (int*) malloc(num_bins * sizeof(int));
    for (int i = 0; i < num_bins; i++) {
        bins[i] = (particle_t*) malloc(MAX_PARTICLES_PER_BIN * sizeof(particle_t));
        bin_counts[i] = 0;
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        for (int i = 0; i < num_bins; i++) {
           bin_counts[i] = 0;
        }

        for (int i = 0; i < n; i++) {
            int bin_x = particles[i].x / bin_size;
            int bin_y = particles[i].y / bin_size;
            int bin_index = bin_y * bins_per_row + bin_x;

            if (bin_index >= 0 && bin_index < num_bins && bin_counts[bin_index] < MAX_PARTICLES_PER_BIN) {
                bins[bin_index][bin_counts[bin_index]++] = particles[i];
            }
        }

        //
        //  compute forces
        //
        for( int i = 0; i < n; i++ )
        {
            particles[i].ax = particles[i].ay = 0;
            
            int bin_x = particles[i].x / bin_size;
            int bin_y = particles[i].y / bin_size;

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int nbx = bin_x + dx;
                    int nby = bin_y + dy;

                    if (nbx < 0 || nbx >= bins_per_row || nby < 0 || nby >= bins_per_row) continue;

                    int neighbor_bin = nby * bins_per_row + nbx;
                    for (int j = 0; j < bin_counts[neighbor_bin]; j++) {
                        apply_force(particles[i], bins[neighbor_bin][j]);
                    }
                }
            }
        }
        
        //
        //  move particles
        //
        for( int i = 0; i < n; i++ ) 
            move( particles[i] );
        
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    for (int i = 0; i < num_bins; i++) {
        free(bins[i]);
    }
    free(bins);
    free(bin_counts);
    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}

