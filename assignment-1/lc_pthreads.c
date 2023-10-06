#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

/* Gives us high-resolution timers. */
#undef _POSIX_C_SOURCE // This prevents the compiler displaying a warning
#define _POSIX_C_SOURCE 199309L
#include <time.h>

/**
 * @brief Return the number of seconds since an unspecified time (e.g., Unix
 *        epoch). This is accomplished with a high-resolution monotonic timer,
 *        suitable for performance timing.
 *
 * @return The number of seconds.
 */
static inline double monotonic_seconds()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
} 

/**
 * @brief Output the seconds elapsed while execution.
 *
 * @param seconds Seconds spent on execution, excluding IO.
 */
static void print_time(double const seconds)
{
  printf("Execution time: %0.04fs\n", seconds);
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("Usage: %s <points> <labels> <iters> <threads>\n", argv[0]);
        return -1;
    }

    int num_points;
    int dim_points;
    double* points; // points[m*dim_points + n] is the nth coord of the mth point
    double* labels;

    int result = load_data(argv[1], argv[2],
                           &num_points, &dim_points,
                           &points, &labels);
    if (result) {
        return -1;
    }

    int iterations = atoi(argv[3]);
    if (iterations == 0) {
        perror("atoi");
        free(points);
        free(labels);
        return -1;
    }

    int threads = atoi(argv[4]);
    if (threads == 0) {
        perror("atoi");
        free(points);
        free(labels);
        return -1;
    }

    double start_time = monotonic_seconds();

    // Put algorithm code here
    double* w = malloc(dim_points * sizeof(int));
    if (w == NULL) {
        perror("malloc");
        free(points);
        free(labels);
        return -1;
    }

    double* w_new = malloc(dim_points * sizeof(int));
    if (w_new == NULL) {
        perror("malloc");
        free(points);
        free(labels);
        return -1;
    }

    double end_time = monotonic_seconds();
    print_time(end_time - start_time);

    free(points);
    free(labels);
    return 0;
}
