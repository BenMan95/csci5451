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
void print_labels(
    char const * const filename,
    unsigned const * const labels,
    size_t const nlabels);

/**
 * @brief Output the seconds elapsed steps 2-5. This excludes input and
 *        output time. This should be wallclock time, not CPU time.
 *
 * @param seconds Seconds spent sorting.
 */
void print_time25(
    double const seconds);

/**
 * @brief Output the seconds elapsed for step 5. This excludes input and
 *        output time. This should be wallclock time, not CPU time.
 *
 * @param seconds Seconds spent sorting.
 */
void print_time5(
    double const seconds);

typedef struct {
    int num_nodes, num_edges; // How many nodes and edges there are in the graph
                              // Note: Each edge is counted twice, once for each direction
    int *counts; // How many edges each node has
    int *offsets; // Offsets for each node's edges
    int *edges; // The underlying data array for edges
                    // edges[offsets[i]+j] is the jth edge of node i
                    // If graph data was set by load_graph
} graph_t;

/**
 * @brief Load point data from a file. This assumes the file contains points in both directions.
 *
 * @param file_name The name of the file to load data from
 * @param graph The graph struct to load data into
 */
void load_graph(
    char *file_name,
    graph_t *graph);

/**
 * @brief Free the arrays of a graph
 *        Does not free the graph struct itself
 *
 * @param graph The graph whose data to free
 */
void free_graph_arrays(graph_t *graph);

#endif
