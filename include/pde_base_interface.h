#ifndef pde_base_interface_h_
#define pde_base_interface_h_

#include "copy_data.h"
#include "pde_handler_access.h"
#include "lac/lac_type.h"

#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/parsed_data_out.h>
#include <deal2lkit/fe_values_cache.h>
#include <deal2lkit/parsed_dirichlet_bcs.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/numerics/error_estimator.h>

//forward declaration
template <int dim, int spacedim, typename LAC> struct Signals;

using namespace pidomus;
using namespace deal2lkit;
/**
 * PDE Base Interface.
 *
 * Provides an unified interface to fill the local (cell-wise) contributions of
 * all matrices and residuals required for the definition of the problem.
 *
 * The underlying PDEHandler driver uses TBB and MPI to assemble matrices and
 * vectors, and calls the virtual methods assemble_local_matrices() and
 * assemble_local_residuals() of this class. If the user wishes to maximise
 * efficiency, then these methods can be directly overloaded by a user class,
 * that manually assembles the local matrices.
 *
 * The default implementation exploit the Sacado package of the Trilinos
 * library to automatically compute the local matrices taking the derivative of
 * residuals and the hessian of the energies supplied by the
 * assemble_energies_and_residuals() method. This method is overloaded for
 * different types, and since these types cannot be inherited by derived
 * classes using template arguments, the Curiously recurring template pattern
 * strategy (CRTP) or F-bound polymorphism
 * (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) is used
 * to allow the implementation in user derived classes of a single templated
 * function energies_and_residuals() that is statically linked and called
 * inside the method PDESystemInterface::assemble_energies_and_residuals().
 *
 * The user can directly overload the methods of this class (which
 * cannot be templated), or derive their classes from PDESystemInterface
 * instead, using CRTP, as in the following example:
 *
 * \code
 * template<int dim, int spacedim, typename LAC>
 * MyInterface : public PDESystemInterface<dim,spacedim, MyInterface<dim,spacedim,LAC>, LAC> {
 * public:
 *  template<typename Number>
 *  void energies_and_residual(...);
 * }
 * \endcode
 *
 * The class PDESystemInterface is derived from PDEBaseInterface, and implements
 * CRTP.
 *
 * PDEBaseInterface derives from PDEHandlerAccess class, which stores a
 * reference to the PDEHandler. Each variable inside the simulator can be
 * accessed via getter functions of the PDEHandlerAccess helper.
 *
 * @authors Luca Heltai, Alberto Sartori, 2016
 */
template <int dim,int spacedim=dim, typename LAC=LATrilinos>
class PDEBaseInterface : public deal2lkit::ParameterAcceptor, public PDEHandlerAccess<dim,spacedim,LAC>
{

public:

  /**
   * Virtual destructor. Does nothing, but guarantees that no leak is left over.
   */
  virtual ~PDEBaseInterface() {}

  /** @name Function needed in order to construct the interface */
  /** @{ */

  /**
   * Constructor. It takes the name of the subsection within the parameter
   * file, the number of components, the number of matrices, the finite element
   * used to discretize the system, the name of the components and a string
   * were the block of differential and algebraic components are specified.
   */
  PDEBaseInterface(const std::string &pde_name="",
                   const std::vector<std::string> &component_names= {"u"},
                   const std::vector<std::string> &matrices_names= {"system"},
                   const std::vector<std::string> &solution_names= {"solution"},
                   const std::string &default_fe="FE_Q(1)");

  /**
   * Set the dimension of coupling consistenly with the number of
   * components and matrices set in the constructor.  This function
   * must be called inside the constructor of the derived class.
   */
  void init ();
  /** @} */

  /**
   * Override this function in your derived interface in order to
   * connect to the signals, which are defined in the struct Signals.
   *
   * Example of implementation:
   * @code
   * auto &signals = this->get_signals();
   * signals.fix_initial_conditions.connect(
   *         [this](VEC &y, VEC &y_dot)
   *          {
   *            y=1;
   *            y_dot=0;
   *          });
   * @endcode
   */
  virtual void connect_to_signals() const;

  /**
   * Solution preprocessing. This function can be used to store abitrary
   * variables needed during the assembling of energies and residuals, that
   * cannot be computed there (e.g., GLOBAL variables). The variables must be
   * stored inside the AnyData of the passed FEValuescache.
   *
   * You may want to use a WorkStream inside this function.
   */
  virtual void solution_preprocessing (FEValuesCache<dim,spacedim> &scratch) const;

  /**
   * Update the internally stored coefficients for the computation of the Jacobian.
   *
   * This class assumes that the Jacobian is the sum of the Hessian of the
   * energy, w.r.t. the solution vectors, each scaled with a coefficient.
   *
   * The coefficients are used to distinguish between active and passive
   * vectors. If a coefficient is zero, then the Jacobian does not depend on
   * that component (i.e., that component is an *explicit* component).
   *
   * @author Luca Heltai, 2018.
   */
  virtual void set_jacobian_coefficients(const std::vector<double> &coefficients) const;


  /** @name Functions dedicated to assemble system and preconditioners */
  /** @{ */

  /**
   * Assemble energies and residuals. To be used when computing only residual
   * quantities, i.e., the energy here is a Sacado double, while the residual
   * is a pure double.
   */
  virtual void assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                               FEValuesCache<dim,spacedim> &,
                                               std::vector<Sdouble> &energies,
                                               std::vector<std::vector<double> > &local_residuals,
                                               bool compute_only_system_terms) const;


  /**
   * Compute linear operators needed by the problem: - @p system_op represents
   * the system matrix associated to the Newton's iterations; - @p prec_op
   * represents the preconditioner; - @p prec_op_finer represents a finer
   * preconditioner that can be used in the case the problem does not converge
   * using @p prec_op .
   *
   * To clarify the difference between @p prec_op and @p prec_op_finer consider
   * the case where you have a matrix A and you want to provide an 'inverse'
   * using AMG. A possible strategy consist in using the linear operator
   * associated to AMG and a further strategy is to invert A using AMG. In
   * detail, @p prec_op should be \code{.cpp} auto A_inv = linear_operator(A,
   * AMG) \endcode while @p prec_op_finer \code{.cpp} auto A_inv =
   * inverse_operator(A, solver, AMG) \endcode
   *
   * In the .prm file it is possible to specify the maximum
   * number of iterations allowed for the solver in the case we are
   * using @p prec_op or @p prec_op_finer.
   *
   * To enable the finer preconditioner it is sufficient to set
   * "Enable finer preconditioner" equals to true.
   */
  virtual void compute_system_operators(const std::vector<shared_ptr<typename LATrilinos::BlockMatrix> >,
                                        LinearOperator<LATrilinos::VectorType> &system_op,
                                        LinearOperator<LATrilinos::VectorType> &prec_op,
                                        LinearOperator<LATrilinos::VectorType> &prec_op_finer) const;


  /**
   * Compute linear operators needed by the problem. When using
   * deal.II vector and matrix types, this function is empty, since a
   * direct solver is used by default.
   */
  virtual void compute_system_operators(const std::vector<shared_ptr<typename LADealII::BlockMatrix> >,
                                        LinearOperator<typename LADealII::VectorType> &,
                                        LinearOperator<typename LADealII::VectorType> &,
                                        LinearOperator<typename LADealII::VectorType> &) const;


  /**
   * Assemble energies and residuals. To be used when computing energetical
   * quantities, i.e., the energy here is a SacadoSacado double, while the residual
   * is a Sacado double.
   */
  virtual void assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                               FEValuesCache<dim,spacedim> &,
                                               std::vector<SSdouble> &energies,
                                               std::vector<std::vector<Sdouble> > &local_residuals,
                                               bool compute_only_system_terms) const;

  /**
   * Assemble local matrices. The default implementation calls
   * PDEBaseInterface::assemble_energies_and_residuals and creates the
   * local matrices by performing automatic differentiation on the
   * results.
   */
  virtual void assemble_local_matrices (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                        FEValuesCache<dim,spacedim> &scratch,
                                        CopyData &data) const;

  /**
   * Assemble the local system residual associated to the given cell.
   * This function is called to evaluate the local system residual at each
   * Newton iteration.
   */
  virtual void assemble_local_system_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                               FEValuesCache<dim,spacedim> &scratch,
                                               CopyData &data) const;

  /**
   * Call the reinit method of the dealii::FEValues with the given cell, and
   * cache all solution vectors, while properly initializing the independent
   * degrees of freedom to work with Sacado.
   */
  template<typename Number>
  void reinit(const Number &alpha,
              const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
              FEValuesCache<dim,spacedim> &fe_cache) const;

  /**
   * Call the reinit method of the dealii::FEFaceValues with the given cell,
   * and cache all solution vectors, while properly initializing the
   * independent degrees of freedom to work with Sacado.
   */
  template<typename Number>
  void reinit(const Number &alpha,
              const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
              const unsigned int &face_no,
              FEValuesCache<dim,spacedim> &fe_cache) const;


  /**
   * Call the reinit method of the dealii::FESubFaceValues with the given cell,
   * and cache all solution vectors, while properly initializing the
   * independent degrees of freedom to work with Sacado.
   */
  template<typename Number>
  void reinit(const Number &alpha,
              const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
              const unsigned int &face_no,
              const unsigned int &subface_no,
              FEValuesCache<dim,spacedim> &fe_cache) const;
  /** @} */


  /** @name Functions dedicated to set properties of the interface */
  /** @{ */

  /**
   * Return the mapping used when no different mapping has been
   *  specified.
   * Return dealii::StaticMappingQ1<dim,spacedim>::mapping by default.
   */
  virtual const Mapping<dim,spacedim> &get_default_mapping() const;

  /**
   * Return the mapping to use with the output.
   */
  virtual const Mapping<dim,spacedim> &get_output_mapping() const;

  /**
   * Return the mapping to use with FEValues.
   */
  virtual const Mapping<dim,spacedim> &get_fe_mapping() const;

  /**
   * Return the mapping to use with boundary conditions.
   */
  virtual const Mapping<dim,spacedim> &get_bc_mapping() const;

  /**
   * Return the mapping to use with the class
   *  dealii::KellyErrorEstimator< dim, spacedim >.
   */
  virtual const Mapping<dim,spacedim> &get_kelly_mapping() const;

  /**
   * Return the mapping to use with errors and convergence
   * rates.
   */
  virtual const Mapping<dim,spacedim> &get_error_mapping() const;

  /**
   * Return the mapping to use with interpolation of functions..
   */
  virtual const Mapping<dim,spacedim> &get_interpolation_mapping() const;

  /**
   * Return the mapping to use with projection of functions...
   */
  virtual const Mapping<dim,spacedim> &get_projection_mapping() const;

  /**
   * This function is called in order to know what are the update flags
   * on the face cell. By default it returns
   * (update_values         | update_quadrature_points  |
   *  update_normal_vectors | update_JxW_values);
   * If you want to use different update flags you need to overwrite
   * this function.
   */
  virtual UpdateFlags get_face_update_flags() const;

  /**
   * This function is called in order to know what are the update flags
   * on the cell. By default it returns
   * (update_values         | update_quadrature_points  |
   *  update_gradients      | update_JxW_values);
   * If you want to use different update flags you need to overwrite
   * this function.
   */
  virtual UpdateFlags get_cell_update_flags() const;
  /** @} */

  /**
   * This function is called to get the coupling of the @p i-th matrix
   */
  const Table<2,DoFTools::Coupling> &get_matrix_coupling(const unsigned int &i) const;

  /**
   * Return the component names.
   */
  std::string get_component_names() const;

  virtual void estimate_error_per_cell(Vector<float> &estimated_error) const;


  /**
   * The names of the components for this pde.
   */
  const std::vector<std::string> component_names;

  /**
   * Names of independent vectors needed for residuals and energies.
   */
  const std::vector<std::string> matrices_names;

  /**
   * Names of independent vectors needed for residuals and energies.
   */
  const std::vector<std::string> solution_names;

  /**
   * Number of components
   */
  const unsigned int n_components;

  /**
   * Number of matrices to be assembled
   */
  const unsigned int n_matrices;

  /**
   * Number of independent vectors needed for residuals and energies.
   */
  const unsigned int n_vectors;

  /**
   * What to use as a default finite element.
   */
  const std::string default_fe_name;

  /**
   * Number of vectors that only appear explicitly;
   */
  mutable unsigned int n_passive_vectors;

  /**
   * Number of vectors that appear as independent variables, for which we need
   * to compute matrices and/or derivatives.
   */
  mutable unsigned int n_active_vectors;

  /**
   * Names of passive vectors.
   */
  mutable std::vector<std::string> passive_vector_names;

  /**
   * Names of active vectors.
   */
  mutable std::vector<std::string> active_vector_names;

  /**
   * Pointers to vectors that need differentiation.
   */
  mutable std::vector<typename LAC::VectorType *> active_vectors;

  /**
   * Pointers to vectors that do not need differentiation.
   */
  mutable std::vector<typename LAC::VectorType *> passive_vectors;

  /**
   * The coefficients to use when computing derivatives.
   */
  mutable std::vector<double> current_coefficients;

  /**
   * Same as above, but only the ones that are different from zero.
   */
  mutable std::vector<double> active_coefficients;

  /**
   * Given the name of a solution vector, get the index in the vector list.
   */
  std::map<std::string, unsigned int> matrix_index;

  /**
   * Given the name of a solution vector, get the index in the vector list.
   */
  std::map<std::string, unsigned int> solution_index;

protected:

  void build_couplings();

  /**
   * Define the coupling of each matrix among its blocks.  By default it
   * sets a full coupling among each block.  If you want to specify the
   * coupling you need to override this function and implement it
   * according to the following example
   * @code
   * void set_matrix_couplings(std::vector<std::string> &couplings) const
   * {
   *   // suppose we have 2 matrices (system and preconditioner)
   *   // we are solving incompressible Stokes equations
   *   couplings[0] = "1,1;1,0"; // first block is the velocity, the second is the pressure
   *   couplings[1] = "1,0;0,1";
   * }
   * @endcode
   */
  virtual void set_matrix_couplings(std::vector<std::string> &couplings) const;

  void convert_string_to_int(const std::string &str_coupling,
                             std::vector<std::vector<unsigned int> > &int_coupling) const;

  /**
   * Convert integer table into a coupling table.
   */
  Table<2, DoFTools::Coupling> to_coupling(const std::vector<std::vector<unsigned int> > &table) const;

  std::vector<Table<2,DoFTools::Coupling> > matrix_couplings;
};



// Template and inline functions

template <int dim, int spacedim, typename LAC>
template<typename Number>
void
PDEBaseInterface<dim,spacedim,LAC>::reinit(const Number &,
                                           const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                           FEValuesCache<dim,spacedim> &fe_cache) const
{
  Number dummy = 0;
  double double_dummy = 0;
  fe_cache.reinit(cell);
  fe_cache.cache_local_solution_vectors(passive_vector_names, passive_vectors, double_dummy);
  fe_cache.cache_local_solution_vectors(active_vector_names, active_vectors, dummy);
}



template <int dim, int spacedim, typename LAC>
template<typename Number>
void
PDEBaseInterface<dim,spacedim,LAC>::reinit(const Number &,
                                           const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                           const unsigned int &face_no,
                                           FEValuesCache<dim,spacedim> &fe_cache) const
{
  Number dummy = 0;
  double double_dummy = 0;
  fe_cache.reinit(cell, face_no);
  fe_cache.cache_local_solution_vectors(passive_vector_names, passive_vectors, double_dummy);
  fe_cache.cache_local_solution_vectors(active_vector_names, active_vectors, dummy);
}



template <int dim, int spacedim, typename LAC>
template<typename Number>
void
PDEBaseInterface<dim,spacedim,LAC>::reinit(const Number &,
                                           const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                           const unsigned int &face_no,
                                           const unsigned int &subface_no,
                                           FEValuesCache<dim,spacedim> &fe_cache) const
{
  Number dummy = 0;
  double double_dummy = 0;
  fe_cache.reinit(cell, face_no, subface_no);
  fe_cache.cache_local_solution_vectors(passive_vector_names, passive_vectors, double_dummy);
  fe_cache.cache_local_solution_vectors(active_vector_names, active_vectors, dummy);
}

#endif
