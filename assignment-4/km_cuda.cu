#include <stdlib.h>
#include <stdio.h>

#define MAX_ITERS 20

/* Gives us high-resolution timers. */
#undef _POSIX_C_SOURCE // This prevents the compiler displaying a warning
#define _POSIX_C_SOURCE 199309L
#include <time.h>

/* OSX timer includes */
#ifdef __MACH__
  #include <mach/mach.h>
  #include <mach/mach_time.h>
#endif

/**
 * @brief Return the number of seconds since an unspecified time (e.g., Unix
 *        epoch). This is accomplished with a high-resolution monotonic timer,
 *        suitable for performance timing.
 *
 * @return The number of seconds.
 */
static inline double monotonic_seconds()
{
#ifdef __MACH__
  /* OSX */
  static mach_timebase_info_data_t info;
  static double seconds_per_unit;
  if(seconds_per_unit == 0) {
    mach_timebase_info(&info);
    seconds_per_unit = (info.numer / info.denom) / 1e9;
  }
  return seconds_per_unit * mach_absolute_time();
#else
  /* Linux systems */
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/**
 * @brief Output the seconds elapsed while clustering.
 *
 * @param seconds Seconds spent on k-means clustering, excluding IO.
 */
static void print_time(double const seconds)
{
  printf("k-means clustering time: %0.04fs\n", seconds);
}

typedef struct {
    int num; // The number of points
    int dim; // The number of dimensions
    double* coords; // The jth coord of the ith points is coords[i*dim+j]
} points_t;

void load_points(
    char* filename,
    points_t* points)
{
    FILE *fh = fopen(filename, "r");
    fscanf(fh, "%d %d", &points->num, &points->dim);

    int size = points->num * points->dim;
    points->coords = (double*) malloc(size * sizeof(double));

    for (int i = 0; i < size; i++) {
        fscanf(fh, "%lf", points->coords + i);
    }

    fclose(fh);
}

__global__ void average(points_t points, double out)
{
    int size = points.num * points.dim;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("Usage: %s <input> <clusters> <blocks> <threads>\n", argv[0]);
        return -1;
    }

    // Load point data
    points_t points;
    load_points(argv[1], &points);
    int size = points.num * points.dim;

    // Copy points to device memory
    points_t d_points;
    d_points.num = points.num;
    d_points.dim = points.dim;
    cudaMalloc((void**) &d_points.coords, size * sizeof(double));

    // // Copy point data to device memory
    // cudaMemcpy(&d_points->num, &points->num, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(&d_points->dim, &points->dim, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_points->coords, points->coords, size * sizeof(double), cudaMemcpyHostToDevice);

    int clusters = atoi(argv[2]);
    int blocks = atoi(argv[3]);
    int threads = atoi(argv[4]);

    double total = 0;
    for (int i = 0; i < size; i++)
        total += points.coords[i];
    printf("avg: %lf\n", total / size);

    free(points.coords);

    return 0;
}
