// #include <stdio.h>
// #include <stdlib.h>
#include <mpi.h>

#include "utils.c"

int main(int argc, char** argv)
{
  graph_t graph;
  int *labels;

  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    load_graph(argv[1], &graph);
  }

  MPI_Bcast(&graph.num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&graph.num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    alloc_graph(&graph);
  }

  free_graph(&graph);

  MPI_Finalize();
  return 0;
}
