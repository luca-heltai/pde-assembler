#include "pidomus.h"
#include "pidomus_macros.h"

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

using namespace dealii;
using namespace deal2lkit;


// This file contains the implementation of the functions
// required to refine the mesh and transfer the solutions
// to the newly created mesh.

#define INSTANTIATE(dim,spacedim,LAC) \
  template class piDoMUS<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)
