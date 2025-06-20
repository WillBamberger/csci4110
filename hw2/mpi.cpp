#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "common.h"

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
    int navg, nabsavg = 0;
    double dmin, absmin = 1.0, davg, absavg = 0.0;
    double rdavg, rdmin;
    int rnavg;
    //
    //  process command line parameters
    //
    if (find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen(sumname, "a") : NULL;

    MPI_Datatype PARTICLE;
    MPI_Type_contiguous(6, MPI_DOUBLE, &PARTICLE);
    MPI_Type_commit(&PARTICLE);

    particle_t *particles = (particle_t *)malloc(n * sizeof(particle_t));

    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size(n);
    if (rank == 0)
        init_particles(n, particles);

    int *partition_offsets = (int *)malloc((n_proc + 1) * sizeof(int));
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    for (int i = 0; i <= n_proc; i++)
        partition_offsets[i] = i * particle_per_proc > n ? n : i * particle_per_proc;

    int *partition_sizes = (int *)malloc(n_proc * sizeof(int));
    for (int i = 0; i < n_proc; i++)
        partition_sizes[i] = partition_offsets[i + 1] - partition_offsets[i];

    int nlocal = partition_sizes[rank];
    particle_t *local = (particle_t *)malloc(nlocal * sizeof(particle_t));
    std::vector<particle_t> ghosts;

    MPI_Scatterv(particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD);

    int bins_per_row = ceil(size / BIN_SIZE);
    int num_bins = bins_per_row * bins_per_row;
    std::vector<std::vector<particle_t *>> bins(num_bins);

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
    for (int step = 0; step < NSTEPS; step++)
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        for (auto &bin : bins) bin.clear();
        ghosts.clear();

        for (int i = 0; i < nlocal; i++)
        {
            local[i].ax = local[i].ay = 0;
            int bin_index = get_bin_index(bins_per_row, local[i].x, local[i].y);
            bins[bin_index].push_back(&local[i]);
        }

     for (int dir = -1; dir <= 1; dir += 2)
        {
            int neighbor = rank + dir;
            if (neighbor >= 0 && neighbor < n_proc)
            {
                int send_count = 0;
                for (int i = 0; i < nlocal; i++)
                {
                    double stripe_width = size / n_proc;
                    double edge = (dir == -1) ? (rank * stripe_width) : ((rank + 1) * stripe_width);
                    if (fabs(local[i].x - edge) < 2 * BIN_SIZE)
                        send_count++;
                }
                std::vector<particle_t> sendbuf(send_count);
                int idx = 0;
                for (int i = 0; i < nlocal; i++)
                {
                    double stripe_width = size / n_proc;
                    double edge = (dir == -1) ? (rank * stripe_width) : ((rank + 1) * stripe_width);
                    if (fabs(local[i].x - edge) < 2 * BIN_SIZE)
                        sendbuf[idx++] = local[i];
                }
                int recv_count;
                MPI_Sendrecv(&send_count, 1, MPI_INT, neighbor, 0,
                             &recv_count, 1, MPI_INT, neighbor, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<particle_t> recvbuf(recv_count);
                MPI_Sendrecv(sendbuf.data(), send_count, PARTICLE, neighbor, 1,
                             recvbuf.data(), recv_count, PARTICLE, neighbor, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                ghosts.insert(ghosts.end(), recvbuf.begin(), recvbuf.end());
            }
        }

        for (auto &g : ghosts)
        {
            int bin_index = get_bin_index(bins_per_row, g.x, g.y);
            bins[bin_index].push_back(&g);
        }

        for (int row = 0; row < bins_per_row; row++)
        {
            for (int col = 0; col < bins_per_row; col++)
            {
                int bin_idx = row * bins_per_row + col;
                for (auto *p : bins[bin_idx])
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            int nr = row + dy, nc = col + dx;
                            if (nr >= 0 && nr < bins_per_row && nc >= 0 && nc < bins_per_row)
                            {
                                int neighbor_idx = nr * bins_per_row + nc;
                                for (auto *q : bins[neighbor_idx])
                                    apply_force(*p, *q, &dmin, &davg, &navg);
                            }
                        }
                    }
                }
            }
        }
	//
        //  move particles
        //
        for (int i = 0; i < nlocal; i++)
            move(local[i]);

        if (find_option(argc, argv, "-no") == -1)
        {
            MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

            if (rank == 0){
	      //
              // Computing statistical data
              //
              if (rnavg){
                absavg += rdavg / rnavg;
                nabsavg++;
              }
              if (rdmin < absmin) absmin = rdmin;
            }
        }

        if (find_option(argc, argv, "-no") == -1 && fsave && (step % SAVEFREQ) == 0 && rank == 0)
        {
            MPI_Gatherv(local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, 0, MPI_COMM_WORLD);
            save(fsave, n, particles);
        }
    }
    simulation_time = read_timer() - simulation_time;
    
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

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
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
