#ifndef pde_system_interface_h
#define pde_system_interface_h

#include "pde_base_interface.h"

using namespace deal2lkit;
using namespace pidomus;

/**
 * This is the class that users should derive from. This class implements the
 * Curiously Recursive Template algorithm (CRTP) or F-bound polymorphism
 * (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) to
 * allow users to define non virtual templated functions to fill local energy
 * densities and residuals.
 *
 * User derived classes need only to implement the function
 * PDESystemInterface::energies_and_residuals(), which can be a templated
 * function on the type of the energy and of the residual.
 *
 * This allows one to compute Jacobian matrices, residual vectors and scalar
 * energies all in the same place, by simply switching the type from double (to
 * compute only an energy), to Sacado double (to allow automatic differentiation
 * on the energy to extract a residual) or a Sacado Sacado double (to allow the
 * automatic construction of the Hessian of a matrix starting from a single
 * scalar energy filled using a Sacado Sacado double).
 *
 * The use of the Curiously Recursive Template algorithm is necessary because
 * the standard does not allow the definition of templated virtual functions,
 * which is what would really be required here. This requirement is circumvented
 * by implementing a user interface (here called Implementation) with template
 * functions (*not* virtual), and passing such interface as a template argument
 * to this class, which is derived by the BaseInterface class.
 *
 * Whenever pi-DoMUS requires an energy or a residual of some type, it will call
 * the BaseInterface class method (where all possible combinations are
 * implemented). This class will defer its call to the Implementation class,
 * which will be well defined at compile time, thus making the compiler happy.
 *
 * The public interface of this class is almost empty. It only scope is to
 * delegate the various function calls to appropriate user classes. To see
 * the full interface used by pi-DoMUS, refer to the documentation of the
 * BaseInterface class.
 */
template<int dim, int spacedim, class Implementation,  typename LAC=LATrilinos>
class PDESystemInterface : public PDEBaseInterface<dim,spacedim,LAC>
{
public:

  virtual ~PDESystemInterface() {}

  /**
   * Pass initializers to the base class constructor.
   */
  PDESystemInterface(const std::string &pde_name="",
                     const std::vector<std::string> &component_names=std::vector<std::string>({"u"}),
                     const std::vector<std::string> &matrix_names=std::vector<std::string>({"system"}),
                     const std::vector<std::string> &solution_names=std::vector<std::string>({"solution"}),
                     const std::string &default_fe="FE_Q(1)") :
    PDEBaseInterface<dim,spacedim,LAC>(pde_name,
                                       component_names,
                                       matrix_names,
                                       solution_names,
                                       default_fe)
  {
    static_cast<Implementation *>(this)->init();
  }


  virtual void assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                               FEValuesCache<dim,spacedim> &scratch,
                                               std::vector<Sdouble> &energies,
                                               std::vector<std::vector<double> > &local_residuals,
                                               bool compute_only_system_terms) const
  {
    static_cast<const Implementation *>(this)->energies_and_residuals(cell,
        scratch,
        energies,
        local_residuals,
        compute_only_system_terms);
  }


  virtual void assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                               FEValuesCache<dim,spacedim> &scratch,
                                               std::vector<SSdouble> &energies,
                                               std::vector<std::vector<Sdouble> > &local_residuals,
                                               bool compute_only_system_terms) const
  {
    static_cast<const Implementation *>(this)->energies_and_residuals(cell,
        scratch,
        energies,
        local_residuals,
        compute_only_system_terms);
  }

};

#endif
