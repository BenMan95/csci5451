#include <stdlib.h>

#ifndef UTILS_H
#define UTILS_H

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
    size_t const nlabels);

/**
 * @brief Output the seconds elapsed steps 2-5. This excludes input and
 *        output time. This should be wallclock time, not CPU time.
 *
 * @param seconds Seconds spent sorting.
 */
static void print_time25(
    double const seconds);

/**
 * @brief Output the seconds elapsed for step 5. This excludes input and
 *        output time. This should be wallclock time, not CPU time.
 *
 * @param seconds Seconds spent sorting.
 */
static void print_time5(
    double const seconds);

#endif
