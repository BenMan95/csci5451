#include <stdlib.h>
#include <stdio.h>

/* Gives us high-resolution timers. */
#undef _POSIX_C_SOURCE // This prevents the compiler displaying a warning
#define _POSIX_C_SOURCE 199309L
#include <time.h>

// /* OSX timer includes */
// #ifdef __MACH__
//   #include <mach/mach.h>
//   #include <mach/mach_time.h>
// #endif

// /**
//  * @brief Return the number of seconds since an unspecified time (e.g., Unix
//  *        epoch). This is accomplished with a high-resolution monotonic timer,
//  *        suitable for performance timing.
//  *
//  * @return The number of seconds.
//  */
// static inline double monotonic_seconds()
// {
// #ifdef __MACH__
//   /* OSX */
//   static mach_timebase_info_data_t info;
//   static double seconds_per_unit;
//   if(seconds_per_unit == 0) {
//     mach_timebase_info(&info);
//     seconds_per_unit = (info.numer / info.denom) / 1e9;
//   }
//   return seconds_per_unit * mach_absolute_time();
// #else
//   /* Linux systems */
//   struct timespec ts;
//   clock_gettime(CLOCK_MONOTONIC, &ts);
//   return ts.tv_sec + ts.tv_nsec * 1e-9;
// #endif
// }

// /**
//  * @brief Output the seconds elapsed while clustering.
//  *
//  * @param seconds Seconds spent on k-means clustering, excluding IO.
//  */
// static void print_time(double const seconds)
// {
//   printf("k-means clustering time: %0.04fs\n", seconds);
// }

typedef struct {
    int num; // The number of points
    int dim; // The number of dimensions
    int* coords; // The jth coord of the ith points is coords[i*dim+j]
} points_t;

void load_points(
    char* filename,
    points_t* points)
{
    FILE *fh = fopen(filename, "r");
    fscanf(fh, "%d %d", &points->num, &points->dim);

    int size = points->num * points->dim;
    points->coords = (int*) malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        fscanf(fh, "%d", points->coords + i);
    }

    fclose(fh);
}

__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("Usage: %s <input> <clusters> <blocks> <threads>\n", argv[0]);
        return -1;
    }

    int clusters = atoi(argv[2]);
    int blocks = atoi(argv[3]);
    int threads = atoi(argv[4]);

    points_t* points = (points_t*) malloc(sizeof(points_t));
    load_points(argv[1], points);

    // TEST
    n = 5;
    int a, b, c;
    int *d_a, *d_b, *d_c;
    size_t size = 

    cudaMalloc((void**) &d_a, sizeof(int));
    cudaMalloc((void**) &d_b, sizeof(int));
    cudaMalloc((void**) &d_c, sizeof(int));

    a = 2;
    b = 7;

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    add<<<2,2>>>(d_a, d_b, d_c); 
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("c=%d\n", c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // TEST

    free(points->coords);
    free(points);
    return 0;
}
