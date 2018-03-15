#include "pde_handler.h"
#include "pidomus_macros.h"

template <int dim, int spacedim, typename LAC>
PDEAssemblerAcces<dim,spacedim,LAC>::PDEAssemblerAcces ()
{}


template <int dim, int spacedim, typename LAC>
PDEAssemblerAcces<dim,spacedim,LAC>::
PDEAssemblerAcces (const PDEHandler<dim,spacedim,LAC> &simulator_object)
  :
  simulator (&simulator_object)
{}


template <int dim, int spacedim, typename LAC>
PDEAssemblerAcces<dim,spacedim,LAC>::~PDEAssemblerAcces ()
{}



template <int dim, int spacedim, typename LAC>
void
PDEAssemblerAcces<dim,spacedim,LAC>::
initialize_simulator (const PDEHandler<dim,spacedim,LAC> &simulator_object) const
{
  simulator = &simulator_object;
}



template <int dim, int spacedim, typename LAC>
const PDEHandler<dim,spacedim,LAC> &
PDEAssemblerAcces<dim,spacedim,LAC>::get_simulator() const
{
  return *simulator;
}



template <int dim, int spacedim, typename LAC>
Signals<dim,spacedim,LAC> &
PDEAssemblerAcces<dim,spacedim,LAC>::get_signals() const
{
  // we need to connect to the signals so a const_cast is required
  return const_cast<Signals<dim,spacedim,LAC>&>(simulator->signals);
}


#ifdef DEAL_II_WITH_MPI
template <int dim, int spacedim, typename LAC>
const MPI_Comm &PDEAssemblerAcces<dim, spacedim, LAC>::get_mpi_communicator() const
{
  return simulator->comm;
}
#endif


template <int dim, int spacedim, typename LAC>
const ConditionalOStream &
PDEAssemblerAcces<dim,spacedim,LAC>::get_pcout () const
{
  return simulator->pcout;
}


// template <int dim, int spacedim, typename LAC>
// unsigned int PDEAssemblerAcces<dim,spacedim,LAC>::get_timestep_number () const
// {
//   return simulator->timestep_number;
// }


template <int dim, int spacedim, typename LAC>
const Triangulation<dim,spacedim> &
PDEAssemblerAcces<dim,spacedim,LAC>::get_triangulation () const
{
  return *simulator->triangulation;
}


template <int dim, int spacedim, typename LAC>
const std::vector<shared_ptr<typename LAC::VectorType>> &
PDEAssemblerAcces<dim,spacedim,LAC>::get_solutions () const
{
  return simulator->solutions;
}



template <int dim, int spacedim, typename LAC>
const std::vector<shared_ptr<typename LAC::VectorType>> &
PDEAssemblerAcces<dim,spacedim,LAC>::get_locally_relevant_solutions () const
{
  return simulator->locally_relevant_solutions;
}


template <int dim, int spacedim, typename LAC>
const DoFHandler<dim,spacedim> &
PDEAssemblerAcces<dim,spacedim,LAC>::get_dof_handler () const
{
  return *simulator->dof_handler;
}


template <int dim, int spacedim, typename LAC>
const FiniteElement<dim,spacedim> &
PDEAssemblerAcces<dim,spacedim,LAC>::get_fe () const
{
  Assert (simulator->dof_handler->n_locally_owned_dofs() != 0,
          ExcMessage("You are trying to access the FiniteElement before the DOFs have been "
                     "initialized. This may happen when accessing the Simulator from a plugin "
                     "that gets executed early in some cases (like material models) or from "
                     "an early point in the core code."));
  return simulator->dof_handler->get_fe();
}

template <int dim, int spacedim, typename LAC>
const ParsedDirichletBCs<dim, spacedim> &
PDEAssemblerAcces<dim,spacedim,LAC>::get_dirichlet_bcs() const
{
  return simulator->dirichlet_bcs;
}

#define INSTANTIATE(dim,spacedim,LA) \
  template class PDEAssemblerAcces<dim,spacedim,LA>;

PIDOMUS_INSTANTIATE(INSTANTIATE)
