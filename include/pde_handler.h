#ifndef pde_handler_h
#define pde_handler_h


#include <deal.II/base/timer.h>
// #include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/linear_operator.h>


#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/base/mpi.h>


// #include <deal.II/lac/precondition.h>


#include "pde_base_interface.h"
#include "pde_handler_access.h"
#include "pidomus_signals.h"

#include <deal2lkit/parsed_grid_generator.h>
#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/parsed_grid_refinement.h>
#include <deal2lkit/error_handler.h>
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parameter_acceptor.h>
#include <deal2lkit/parsed_zero_average_constraints.h>
#include <deal2lkit/parsed_mapped_functions.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>

#include <deal2lkit/any_data.h>
#include <deal2lkit/fe_values_cache.h>

#include "lac/lac_type.h"
#include "lac/lac_initializer.h"
#include "tria_helper.h"

using namespace dealii;
using namespace deal2lkit;
using namespace pidomus;

template<int dim> using dim_is_one = std::integral_constant<bool, (dim==1)>;

template <int dim, int spacedim = dim, typename LAC = LATrilinos>
class PDEHandler : public deal2lkit::ParameterAcceptor
{
  // This is a class required to make tests
  template<int fdim, int fspacedim, typename fn_LAC>
  friend void test(PDEHandler<fdim, fspacedim, fn_LAC> &);

public:

  /**
   * Assembler method for a general (possibly constrained) energy.
   *
   * @param name
   * @param energy
   * @param comm
   */
  PDEHandler (const std::string &name,
              const PDEBaseInterface<dim, spacedim, LAC> &pde,
              const MPI_Comm &comm = MPI_COMM_WORLD);

  /**
   * Computes the residual and the Jacobian associated with the given energy of
   * the problem.
   *
   * The assumption is that there exists a scalar function \f$E(x, p)\f$ and a
   * vector function \f$R_0(x, p)\f$ such that you can write a residual
   * function \f$R_i(x, p) = \sum_j (\partial E(x, p)/\partial x_j, v_i)+
   * (R_0(x, p), v_i)\f$, where both $x$ and $p$ are arbitrary length vectors
   * of finite element solutions and double parameters.
   *
   * The Jacobian of the system is computed as
   * \f[
   * J_{ij} := \sum_k c_k (\partial R_j(x, p)/\partial x_k, v_i)
   * \f]
   * where the coefficients $c_k$ are given as an input parameter to the
   * residual function, and the Jacobian is computed only if the parameter
   * @p setup_jacobian is set to true.
   *
   * @param parameters Arbitrary coefficients (like time, stabilization, etc.)
   * @param coefficients Coefficients used to scale the derivative of the
   *        Jacobian w.r.t. to the ith input vector.
   * @param input_vectors All input vectors used to compute the residual and the
   *        energy.
   * @param dst The actual residual.
   * @param setup_jacobian If this flag is true, then the jacobian is updated
   *        internally.
   */
  void residual(const std::vector<double>                                     &coefficients,
                const std::vector<std::shared_ptr<typename LAC::VectorType> > &input_vectors,
                typename LAC::VectorType                                      &residual_vector);

  /**
   * Same as above, but use all internally stored variables.
   *
   * @param[out] residual The actually computed residual.
   */
  void residual(typename LAC::VectorType &residual_vector);

  /**
   * Set time to @p t for forcing terms and boundary conditions
   */
  void update_functions_and_constraints(const double &t);


  /**
   * Apply Dirichlet boundary conditions.
   * It takes as argument a DoF handler @p dof_handler, a
   * ParsedDirichletBCs and a constraint matrix @p constraints.
   *
   */
  void apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                            const ParsedDirichletBCs<dim,spacedim> &bc,
                            ConstraintMatrix &constraints) const;

  /**
   * Apply Neumann boundary conditions.
   *
   */
  void apply_neumann_bcs (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                          FEValuesCache<dim,spacedim> &scratch,
                          std::vector<double> &local_residual) const;


  /**
   * Applies CONSERVATIVE forcing terms, which can be defined by
   * expressions in the parameter file.
   *
   * If the problem involves NON-conservative loads, they must be
   * included in the residual formulation.
   *
   */
  void apply_forcing_terms (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                            FEValuesCache<dim,spacedim> &scratch,
                            std::vector<double> &local_residual) const;



  void make_grid_fe();
  void setup_dofs (const bool &first_run = true);

  void output_solution(const std::string &suffix="") const;

  void assemble_matrices (const std::vector<double> &coefficients,
                          const std::vector<std::shared_ptr<typename LAC::VectorType>>  &input_vectors);

  /**
   * Same as above, using already stored parameters and coefficients.
   */
  void assemble_matrices ();


  typename LAC::VectorType &v(const std::string &vec_name) const
  {
    Assert(pde.solution_index.find(vec_name) != pde.solution_index.end(),
           ExcMessage("Vector not found"));
    return *solutions[pde.solution_index.at(vec_name)];
  }

  typename LAC::BlockMatrix &m(const std::string &matrix_name) const
  {
    Assert(pde.matrix_index.find(matrix_name) != pde.matrix_index.end(),
           ExcMessage("Matrix not found"));
    return *matrices[pde.matrix_index.at(matrix_name)];
  }

  template<int odim=dim>
  typename std::enable_if<(odim==1)>::type refine_mesh ();

  template<int odim=dim>
  typename std::enable_if<(odim>1)>::type refine_mesh ();

  template<int odim=dim>
  typename std::enable_if<(odim>1)>::type refine_and_transfer_solutions (
    std::vector<std::shared_ptr<LADealII::VectorType>> &solutions,
    bool adaptive_refinement);

  template<int odim=dim>
  typename std::enable_if<(odim>1)>::type refine_and_transfer_solutions (
    std::vector<std::shared_ptr<typename LATrilinos::VectorType>> &solutions,
    bool adaptive_refinement);

  template<int odim=dim>
  typename std::enable_if<(odim==1)>::type
  refine_and_transfer_solutions (std::vector<std::shared_ptr<typename LAC::VectorType>> &,
                                 bool);


  void set_constrained_dofs_to_zero(typename LAC::VectorType &v) const;

  void interpolate_or_project(const Function<spacedim> &f, typename LAC::VectorType &v);

  const MPI_Comm &comm;

  const PDEBaseInterface<dim, spacedim, LAC>    &pde;

  ConditionalOStream        pcout;

  TriaHelper<dim,spacedim,LAC> tria_helper;
  std::shared_ptr<Triangulation<dim,spacedim> >   triangulation;
  ParsedGridRefinement  pgr;

  ParsedFiniteElement<dim,spacedim>                    pfe;
  std::shared_ptr<FiniteElement<dim, spacedim> >       fe;
  std::shared_ptr<DoFHandler<dim, spacedim> >          dof_handler;

  std::vector<shared_ptr<ConstraintMatrix> >      constraints;

  std::vector<LinearOperator<typename LAC::VectorType>> operators;

  mutable ParsedDataOut<dim, spacedim> data_out;
  //// current time
public:
  /**
   * Vector of solutions.
   */
  mutable std::vector<shared_ptr<typename LAC::VectorType> > solutions;

  /**
   * Same as above, with read access only on relevant dofs.
   */
  std::vector<shared_ptr<typename LAC::VectorType> > locally_relevant_solutions;

  /**
   * Current parameters.
   */
  std::map<std::string, double> parameters;

  std::vector<double> energy_coefficients;

  unsigned int initial_global_refinement;

  /**
   * Teucos timer file
   */
  mutable TimeMonitor       computing_timer;

  double current_time = 0;

  const unsigned int n_matrices;
  std::vector<shared_ptr<typename LAC::BlockSparsityPattern> > matrix_sparsities;
  std::vector<shared_ptr<typename LAC::BlockMatrix> >  matrices;

  ParsedMappedFunctions<spacedim>  forcing_terms; // on the volume
  ParsedMappedFunctions<spacedim>  neumann_bcs;
  ParsedDirichletBCs<dim,spacedim> dirichlet_bcs;

  ParsedZeroAverageConstraints<dim,spacedim> zero_average;

  std::vector<types::global_dof_index> dofs_per_block;
  IndexSet global_partitioning;
  std::vector<IndexSet> partitioning;
  std::vector<IndexSet> relevant_partitioning;

  bool adaptive_refinement;
  const bool we_are_parallel;
  bool use_direct_solver;

  /**
   * Print all avaible information about processes.
   */
  bool verbose;

  /**
   * Struct containing the signals
   */
  Signals<dim,spacedim,LAC>    signals;

  /**
   * PDEAssemblerAccess accesses to all internal variables and returns a
   * const reference to them through functions named get_variable()
   */
  friend class PDEHandlerAccess<dim,spacedim,LAC>;

  virtual void declare_parameters(ParameterHandler &prm);

  virtual void parse_parameters_call_back();
public:

  /**
   * call this function after ParameterAcceptor::initialize and before
   * any other function of the solver
   */
  void init();
};

// Template and inline functions.


template <int dim, int spacedim, typename LAC>
template<int odim>
typename std::enable_if<(odim==1)>::type
PDEHandler<dim, spacedim, LAC>::refine_mesh ()
{
  Assert(false, ExcImpossibleInDim(dim));
}



template <int dim, int spacedim, typename LAC>
template <int odim>
typename std::enable_if<(odim>1)>::type
PDEHandler<dim, spacedim, LAC>::refine_mesh()
{
  auto _timer = computing_timer.scoped_timer ("Mesh refinement");

  signals.begin_refine_mesh();

  if (adaptive_refinement)
    {
      Vector<float> estimated_error_per_cell (triangulation->n_active_cells());

      pde.estimate_error_per_cell(estimated_error_per_cell);

      pgr.mark_cells(estimated_error_per_cell, *triangulation);
    }

  refine_and_transfer_solutions(solutions,
                                adaptive_refinement);

  signals.end_refine_mesh();
}


template <int dim, int spacedim, typename LAC>
template <int odim>
typename std::enable_if<(odim>1)>::type
PDEHandler<dim, spacedim, LAC>::
refine_and_transfer_solutions(std::vector<shared_ptr<typename LATrilinos::VectorType>> &solutions,
                              bool adaptive_refinement)
{
  signals.begin_refine_and_transfer_solutions();
  AssertDimension(solutions.size(), locally_relevant_solutions.size());
  std::vector<const LATrilinos::VectorType *> old_sols (solutions.size());

  for (unsigned int i=0; i<solutions.size(); ++i)
    {
      locally_relevant_solutions[i] = solutions[i];
      old_sols[i] = locally_relevant_solutions[i].get();
    }

  parallel::distributed::SolutionTransfer<dim, LATrilinos::VectorType, DoFHandler<dim,spacedim> > sol_tr(*dof_handler);

  triangulation->prepare_coarsening_and_refinement();
  sol_tr.prepare_for_coarsening_and_refinement (old_sols);

  if (adaptive_refinement)
    triangulation->execute_coarsening_and_refinement ();
  else
    triangulation->refine_global (1);

  setup_dofs(false);

  std::vector<std::shared_ptr<typename LATrilinos::VectorType>> new_solutions
                                                             (solutions.size(), SP(new LATrilinos::VectorType(*solutions[0])));

  std::vector<LATrilinos::VectorType *> new_sols (solutions.size());

  for (unsigned int i=0; i<solutions.size(); ++i)
    new_sols[i] = new_solutions[i].get();

  sol_tr.interpolate (new_sols);

  for (unsigned int i=0; i<solutions.size(); ++i)
    solutions[i] = new_solutions[i];

  constraints[0]->distribute(*solutions[0]);

  for (unsigned int i=0; i<solutions.size(); ++i)
    locally_relevant_solutions[i] = solutions[i];

  signals.end_refine_and_transfer_solutions();
}


template <int dim, int spacedim, typename LAC>
template <int odim>
typename std::enable_if<(odim>1)>::type
PDEHandler<dim, spacedim, LAC>::
refine_and_transfer_solutions(std::vector<shared_ptr<LADealII::VectorType>> &y,
                              bool adaptive_refinement)
{
  signals.begin_refine_and_transfer_solutions();
  SolutionTransfer<dim, LADealII::VectorType, DoFHandler<dim,spacedim> > sol_tr(*dof_handler);

  std::vector<LADealII::VectorType> old_sols(pde.n_vectors);
  for (unsigned int i=0; i<pde.n_vectors; ++i)
    old_sols[i] = *y[i];

  triangulation->prepare_coarsening_and_refinement();
  sol_tr.prepare_for_coarsening_and_refinement (old_sols);

  if (adaptive_refinement)
    triangulation->execute_coarsening_and_refinement ();
  else
    triangulation->refine_global (1);

  setup_dofs(false);

  std::vector<LADealII::VectorType> new_sols(pde.n_vectors);
  for (unsigned int i=0; i<pde.n_vectors; ++i)
    {
      new_sols[i].reinit(*solutions[i], true);
      y[i]->reinit(*solutions[i],true);
    }

  sol_tr.interpolate (old_sols, new_sols);

  for (unsigned int i=0; i<pde.n_vectors; ++i)
    {
      *y[i] = new_sols[i];
    }

  signals.end_refine_and_transfer_solutions();
}

template <int dim, int spacedim, typename LAC>
template <int odim>
typename std::enable_if<(odim==1)>::type
PDEHandler<dim, spacedim, LAC>::refine_and_transfer_solutions (std::vector<shared_ptr<typename LAC::VectorType>> &,
    bool)
{
  return;
}


#endif
