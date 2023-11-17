#include "utils.h"

#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Write a vector of labels to a file.
 *
 * @param filename The name of the file to write to.
 * @param labels The array of labels.
 * @param nlabels How many labels to write.
 */
static void print_labels(
    char const * const filename,
    unsigned const * const labels,
    size_t const nlabels)
{
  size_t i;
  FILE * fout;

  /* open file */
  if((fout = fopen(filename, "w")) == NULL) {
    fprintf(stderr, "error opening '%s'\n", filename);
    abort();
  }

  /* write labels to fout */
  for(i = 0; i < nlabels; ++i) {
    fprintf(fout, "%u\n", labels[i]);
  }

  fclose(fout);
}

/**
 * @brief Output the seconds elapsed steps 2-5. This excludes input and
 *        output time. This should be wallclock time, not CPU time.
 *
 * @param seconds Seconds spent sorting.
 */
static void print_time25(
    double const seconds)
{
  printf("2-5 Time: %0.04fs\n", seconds);
}

/**
 * @brief Output the seconds elapsed for step 5. This excludes input and
 *        output time. This should be wallclock time, not CPU time.
 *
 * @param seconds Seconds spent sorting.
 */
static void print_time5(
    double const seconds)
{
  printf("5 Time: %0.04fs\n", seconds);
}