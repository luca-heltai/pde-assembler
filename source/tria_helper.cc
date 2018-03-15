#include "pidomus_macros.h"
#include <tria_helper.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

template <int dim, int spacedim, typename LAC>
TriaHelper<dim,spacedim,LAC>::TriaHelper(const MPI_Comm &_comm):
  comm(_comm),
  pgg("Domain"),
  p_serial(nullptr),
  p_parallel(nullptr)
{}

template <int dim, int spacedim, typename LAC>
void TriaHelper<dim,spacedim,LAC>::make_grid()
{
  if (LAC::triatype == TriaType::serial)
    p_serial = shared_ptr<Triangulation<dim,spacedim> >(pgg.serial());
#ifdef DEAL_II_WITH_MPI
  else
    p_parallel = shared_ptr<parallel::distributed::Triangulation<dim,spacedim> >(pgg.distributed(comm));
#endif
}

template <int dim, int spacedim, typename LAC>
shared_ptr<Triangulation<dim,spacedim> >
TriaHelper<dim,spacedim,LAC>::get_tria() const
{
  if (LAC::triatype == TriaType::serial)
    return p_serial;
#ifdef DEAL_II_WITH_MPI
  return p_parallel;
#endif
}

#define INSTANTIATE(dim,spacedim,LAC) \
  template class TriaHelper<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)
