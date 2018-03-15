#ifndef __pidomus_tria_helper_h
#define __pidomus_tria_helper_h



#include <lac/lac_type.h>
#include <memory>
#include <deal2lkit/parsed_grid_generator.h>
#include <deal.II/base/mpi.h>

using namespace dealii;
using namespace deal2lkit;

/**
  * this class helps pidomus in creating the appropriate triangulation
  * i.e. serial or parallel::distributed
  */
template <int dim, int spacedim, typename LAC>
class TriaHelper
{
public:

  /**
    * constructor
    */
  TriaHelper(const MPI_Comm &comm=MPI_COMM_WORLD);

  /**
   * @brief Generate the triangulation according to the LAC::TriaType
   * and set the corresponding shared_ptr
   */
  void make_grid();

  /**
   * @brief Return a pointer to the triangulation
   */
  std::shared_ptr<Triangulation<dim,spacedim> > get_tria() const;

private:

  const MPI_Comm &comm;
  ParsedGridGenerator<dim, spacedim>   pgg;
  std::shared_ptr<Triangulation<dim,spacedim> > p_serial;

#ifdef DEAL_II_WITH_MPI
  std::shared_ptr<parallel::distributed::Triangulation<dim,spacedim> > p_parallel;
#endif
};

#endif // __pidomus_tria_helper_h
