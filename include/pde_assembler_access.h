#ifndef pde_assembler_acess_h
#define pde_assembler_acess_h

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>

using namespace dealii;
//using namespace deal2lkit;


// forward declaration

template <int dim, int spacedim, typename LAC> class PDEHandler;
template <int dim, int spacedim, typename LAC> struct Signals;

template <int dim, int spacedim, typename LAC>
class PDEAssemblerAcces
{
public:

  /**
   * Default constructor. Initialize the PDEAssemblerAcces object without
   * a reference to a particular PDEAssembler object. You will later have
   * to call initialize() to provide this reference to the PDEAssembler
   * object.
   */
  PDEAssemblerAcces ();

  /**
   * Create a PDEAssemblerAcces object that is already initialized for
   * a particular PDEAssembler.
   */
  PDEAssemblerAcces (const PDEHandler<dim,spacedim,LAC> &simulator_object);

  /**
   * Destructor. Does nothing but is virtual so that derived classes
   * destructors are also virtual.
   */
  virtual
  ~PDEAssemblerAcces ();

  /**
   * Initialize this class for a given simulator. This function is marked
   * as virtual so that derived classes can do something upon
   * initialization as well, for example look up and cache data; derived
   * classes should call this function from the base class as well,
   * however.
   *
   * @param simulator_object A reference to the main simulator object.
   */
  virtual void initialize_simulator (const PDEHandler<dim,spacedim,LAC> &simulator_object) const;

  /** @name Accessing variables that identify overall properties of the simulator */
  /** @{ */

  /**
   * Return a reference to the PDEAssembler itself. Note that you can not
   * access any members or functions of the PDEAssembler. This function
   * exists so that any class with PDEAssemblerAcces can create other
   * objects with PDEAssemblerAcces (because initializing them requires a
   * reference to the PDEAssembler).
   */
  const PDEHandler<dim,spacedim,LAC> &
  get_simulator () const;

  const std::vector<shared_ptr<typename LAC::VectorType>> &
                                                       get_solutions() const;

  const std::vector<shared_ptr<typename LAC::VectorType>> &
                                                       get_locally_relevant_solutions() const;

  /**
   * Get access to the structure containing the signals of PDEAssembler
   */
  Signals<dim,spacedim,LAC> &
  get_signals() const;

  /**
   * Return the MPI communicator for this simulation.
   */
  const MPI_Comm &
  get_mpi_communicator () const;

  /**
   * Return a reference to the stream object that only outputs something
   * on one processor in a parallel program and simply ignores output put
   * into it on all other processors.
   */
  const ConditionalOStream &
  get_pcout () const;

  /**
   * Return a reference to the triangulation in use by the simulator
   * object.
   */
  const Triangulation<dim,spacedim> &
  get_triangulation () const;
  /** @} */

  /**
   * Return a reference to the DoFHandler that is used to
   * discretize the variables at the current time step.
   */
  const DoFHandler<dim,spacedim> &
  get_dof_handler () const;

  /**
   * Return a reference to the finite element that the DoFHandler
   * that is used to discretize the variables at the current time
   * step is built on.
   */
  const FiniteElement<dim,spacedim> &
  get_fe () const;

  /**
   * Return a reference to the ParsedDirichletBCs that stores
   * the Dirichlet boundary conditions set in the parameter file.
   */
  const ParsedDirichletBCs<dim,spacedim> &
  get_dirichlet_bcs () const;

  /** @} */

private:

  /**
   * A pointer to the simulator object to which we want to get
   * access.
   */
  mutable const PDEHandler<dim,spacedim,LAC> *simulator;
};



#endif
