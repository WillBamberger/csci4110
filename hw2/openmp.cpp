#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"
#include "omp.h"

#define BIN_SIZE 0.01

extern double size;

int get_bin_index(int bins_per_row, double x, double y)
{
    int row = y / BIN_SIZE;
    int col = x / BIN_SIZE;
    return row * bins_per_row + col;
}

//
//  benchmarking program
//
int main(int argc, char **argv)
{   
    int navg, nabsavg = 0, numthreads; 
    double dmin, absmin = 1.0, davg, absavg = 0.0;

    if (find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n"); 
        printf("-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen(sumname, "a") : NULL;      

    particle_t *particles = (particle_t *)malloc(n * sizeof(particle_t));
    set_size(n);
    init_particles(n, particles);

    int bins_per_row = ceil(size / BIN_SIZE);
    int num_bins = bins_per_row * bins_per_row;

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();

    #pragma omp parallel
    {
        numthreads = omp_get_num_threads();
        std::vector<std::vector<particle_t*>> local_bins(num_bins);

        for (int step = 0; step < 1000; step++)
        {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;
	//
        //  compute all forces
        //
            
            for (int i = 0; i < num_bins; i++)
                local_bins[i].clear();

                #pragma omp for nowait
		for (int i = 0; i < n; i++)
		{
                	particles[i].ax = particles[i].ay = 0;
                	int bin_index = get_bin_index(bins_per_row, particles[i].x, particles[i].y);
                	local_bins[bin_index].push_back(&particles[i]);
		}

	 	#pragma omp for reduction(+:navg, davg)
        	for (int bin_idx = 0; bin_idx < bins_per_row * bins_per_row; bin_idx++) {
			int row = bin_idx / bins_per_row;
			int col = bin_idx % bins_per_row;

			 for (auto* p : local_bins[bin_idx]) {
        			for (int dy = -1; dy <= 1; dy++) {
            				int nr = row + dy;
            				if (nr < 0 || nr >= bins_per_row) continue;

            				for (int dx = -1; dx <= 1; dx++) {
                				int nc = col + dx;
                				if (nc < 0 || nc >= bins_per_row) continue;

                				int neighbor_idx = nr * bins_per_row + nc;
                				for (auto* q : local_bins[neighbor_idx])
                    					apply_force(*p, *q, &dmin, &davg, &navg);
            				}
        			}
    			}
		}


	//
        //  move particles
        //
            #pragma omp for simd
            for (int i = 0; i < n; i++)
                move(particles[i]);

            if (find_option(argc, argv, "-no") == -1)
            {
              //
              //  compute statistical data
              //
              #pragma omp critical
                {
                    if (navg) {
                        absavg += davg / navg;
                        nabsavg++;
                    }
                    if (dmin < absmin) absmin = dmin;
                }
	  	//
          	//  save if necessary
          	//
                #pragma omp master
                if (fsave && (step % SAVEFREQ) == 0)
                    save(fsave, n, particles);
            }
        }
    }

    simulation_time = read_timer() - simulation_time;

    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -the minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");
    
    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
