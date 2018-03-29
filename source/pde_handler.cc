#include "pde_handler.h"

#include "pidomus_macros.h"
#include "pidomus_signals.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal2lkit/utilities.h>
#include "lac/lac_initializer.h"

#include <typeinfo>
#include <limits>
#include <numeric>

#include "tria_helper.h"


using namespace dealii;
using namespace deal2lkit;

template <int dim, int spacedim, typename LAC>
PDEHandler<dim, spacedim, LAC>::PDEHandler (const std::string &name,
                                            const PDEBaseInterface<dim, spacedim, LAC> &pde,
                                            const MPI_Comm &communicator)
  :
  ParameterAcceptor(name),
  comm(communicator),
  interface(pde),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(comm)
            == 0)),

    tria_helper(comm),

    pgr("Refinement"),

    pfe("Finite element space", pde.default_fe_name,
        pde.get_component_names(), pde.n_components),

    constraints(pde.n_matrices),

    data_out("Output Parameters", "none", 1, pde.get_component_names(),
             "solution", "", comm),

    n_matrices(pde.n_matrices),

    iterative_solver("Solvers -- Coarse solver", "cg", 100, 1e-8),

    finer_iterative_solver("Solvers -- Fine solver", "cg", 1000, 1e-7),

    forcing_terms("Problem data -- Forcing terms",
                  pde.n_components,
                  pde.get_component_names(), ""),
    neumann_bcs("Problem data -- Neumann boundary conditions",
                pde.n_components,
                pde.get_component_names(), ""),
    dirichlet_bcs("Problem data -- Dirichlet boundary conditions",
                  pde.n_components,
                  pde.get_component_names(), "0=ALL"),

    zero_average("Problem data -- Zero average constraints",
                 pde.n_components,
                 pde.get_component_names() ),

    we_are_parallel(Utilities::MPI::n_mpi_processes(comm) > 1)
{

  pde.initialize_simulator (*this);

  constraints[0] = SP(new ConstraintMatrix());

  for (unsigned int i=1; i<n_matrices; ++i)
    constraints[i] = constraints[0];

  for (unsigned int i=0; i<n_matrices; ++i)
    {
      matrices.push_back( SP( new typename LAC::BlockMatrix() ) );
      matrix_sparsities.push_back( SP( new typename LAC::BlockSparsityPattern() ) );
    }

  for (unsigned int i=0; i<pde.n_vectors; ++i)
    {
      solutions.push_back(SP(new typename LAC::VectorType()));
      locally_relevant_solutions.push_back(SP(new typename LAC::VectorType()));
    }
}

template<int dim, int spacedim, typename LAC>
void PDEHandler<dim,spacedim,LAC>::init()
{
  interface.connect_to_signals();
  make_grid_fe();
  setup_dofs(true);
}

template <int dim, int spacedim, typename LAC>
void PDEHandler<dim, spacedim, LAC>::make_grid_fe()
{
  auto _timer = computing_timer.scoped_timer("Make grid and finite element");
  signals.begin_make_grid_fe();
  tria_helper.make_grid();
  triangulation = tria_helper.get_tria();
  dof_handler = SP(new DoFHandler<dim, spacedim>(*triangulation));
  signals.postprocess_newly_created_triangulation(triangulation.get());
  fe = pfe();
  triangulation->refine_global (initial_global_refinement);
  signals.end_make_grid_fe();
}



template <int dim, int spacedim, typename LAC>
void PDEHandler<dim, spacedim, LAC>::setup_dofs (const bool &first_run)
{
  auto _timer = computing_timer.scoped_timer("Setup dof systems");
  signals.begin_setup_dofs();
  std::vector<unsigned int> sub_blocks = pfe.get_component_blocks();
  dof_handler->distribute_dofs (*fe);
  DoFRenumbering::component_wise (*dof_handler, sub_blocks);

  dofs_per_block.clear();
  dofs_per_block.resize(pfe.n_blocks());

  DoFTools::count_dofs_per_block (*dof_handler, dofs_per_block,
                                  sub_blocks);

  std::locale s = pcout.get_stream().getloc();
  pcout.get_stream().imbue(std::locale(""));
  pcout << "Number of active cells: "
        << triangulation->n_global_active_cells()
        << " (on "
        << triangulation->n_levels()
        << " levels)"
        << std::endl
        << "Number of degrees of freedom: "
        << dof_handler->n_dofs()
        << "(" << print(dofs_per_block, "+") << ")"
        << std::endl
        << std::endl;
  pcout.get_stream().imbue(s);


  partitioning.resize(0);
  relevant_partitioning.resize(0);

  IndexSet relevant_set;
  {
    global_partitioning = dof_handler->locally_owned_dofs();
    for (unsigned int i = 0; i < pfe.n_blocks(); ++i)
      partitioning.push_back(global_partitioning.get_view( std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i, 0),
                                                           std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i + 1, 0)));

    DoFTools::extract_locally_relevant_dofs (*dof_handler,
                                             relevant_set);

    for (unsigned int i = 0; i < pfe.n_blocks(); ++i)
      relevant_partitioning.push_back(relevant_set.get_view(std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i, 0),
                                                            std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i + 1, 0)));
  }

  update_functions_and_constraints(current_time,current_time_step);

  ScopedLACInitializer initializer(dofs_per_block,
                                   partitioning,
                                   relevant_partitioning,
                                   comm);

  AssertDimension(solutions.size(), locally_relevant_solutions.size());
  AssertDimension(solutions.size(), interface.n_vectors);
  for (unsigned int i=0; i<solutions.size(); ++i)
    {
      initializer(*solutions[i]);
      initializer.ghosted(*locally_relevant_solutions[i]);
    }

  for (unsigned int i=0; i < n_matrices; ++i)
    {
      matrices[i]->clear();
      initializer(*matrix_sparsities[i],
                  *dof_handler,
                  *constraints[i],
                  interface.get_matrix_coupling(i));
      matrices[i]->reinit(*matrix_sparsities[i]);
    }

  if (first_run)
    {
//      if (fe->has_support_points())
//        {
//          VectorTools::interpolate(interface.get_interpolate_mapping(), *dof_handler, initial_solution, solution);
//          VectorTools::interpolate(interface.get_interpolate_mapping(), *dof_handler, initial_solution_dot, solution_dot);
//        }
//      else if (!we_are_parallel)
//        {
//          const QGauss<dim> quadrature_formula(fe->degree + 1);
//          VectorTools::project(interface.get_project_mapping(), *dof_handler, C(0), quadrature_formula, initial_solution, solution);
//          VectorTools::project(interface.get_project_mapping(), *dof_handler, C(0), quadrature_formula, initial_solution_dot, solution_dot);
//        }
//      else
//        {
//          Point<spacedim> p;
//          Vector<double> vals(interface.n_components);
//          Vector<double> vals_dot(interface.n_components);
//          initial_solution.vector_value(p, vals);
//          initial_solution_dot.vector_value(p, vals_dot);

//          unsigned int comp = 0;
//          for (unsigned int b=0; b<solution.n_blocks(); ++b)
//            {
//              solution.block(b) = vals[comp];
//              solution_dot.block(b) = vals_dot[comp];
//              comp += fe->element_multiplicity(b);
//            }
//        }

//      signals.fix_initial_conditions(solution, solution_dot);
//      locally_relevant_explicit_solution = solution;

    }
  signals.end_setup_dofs();
}



template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim,spacedim,LAC>::
apply_neumann_bcs (
  const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
  FEValuesCache<dim,spacedim> &scratch,
  std::vector<double> &local_residual) const
{

  double dummy = 0.0;

  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      unsigned int face_id = cell->face(face)->boundary_id();
      if (cell->face(face)->at_boundary() && neumann_bcs.acts_on_id(face_id))
        {
          interface.reinit(dummy, cell, face, scratch);

          auto &fev = scratch.get_current_fe_values();
          auto &q_points = scratch.get_quadrature_points();
          auto &JxW = scratch.get_JxW_values();

          for (unsigned int q=0; q<q_points.size(); ++q)
            {
              Vector<double> T(interface.n_components);
              neumann_bcs.get_mapped_function(face_id)->vector_value(q_points[q], T);

              for (unsigned int i=0; i<local_residual.size(); ++i)
                for (unsigned int c=0; c<interface.n_components; ++c)
                  local_residual[i] -= T[c]*fev.shape_value_component(i,q,c)*JxW[q];

            }// end loop over quadrature points

          break;

        } // endif face->at_boundary

    }// end loop over faces

}// end function definition



template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim,spacedim,LAC>::
apply_forcing_terms (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                     FEValuesCache<dim,spacedim> &scratch,
                     std::vector<double> &local_residual) const
{
  unsigned cell_id = cell->material_id();
  if (forcing_terms.acts_on_id(cell_id))
    {
      double dummy = 0.0;
      interface.reinit(dummy, cell, scratch);

      auto &fev = scratch.get_current_fe_values();
      auto &q_points = scratch.get_quadrature_points();
      auto &JxW = scratch.get_JxW_values();
      for (unsigned int q=0; q<q_points.size(); ++q)
        for (unsigned int i=0; i<local_residual.size(); ++i)
          for (unsigned int c=0; c<interface.n_components; ++c)
            {
              double B = forcing_terms.get_mapped_function(cell_id)->value(q_points[q],c);
              local_residual[i] -= B*fev.shape_value_component(i,q,c)*JxW[q];
            }
    }
}


template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim,spacedim,LAC>::
apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                     const ParsedDirichletBCs<dim,spacedim> &bc,
                     ConstraintMatrix &constraints) const
{
  try
    {
      bc.interpolate_boundary_values(interface.get_bc_mapping(),dof_handler,constraints);
    }
  catch (...)
    {
      AssertThrow(!we_are_parallel,
                  ExcMessage("You called VectorTools::project_boundary_values(), which is not \n"
                             "currently supported on deal.II in parallel settings.\n"
                             "Feel free to submit a patch :)"));
      const QGauss<dim-1> quad(fe->degree+1);
      bc.project_boundary_values(interface.get_bc_mapping(),dof_handler,quad,constraints);
    }
  // unsigned int codim = spacedim - dim;
  // if (codim == 0)
  //   bc.compute_nonzero_normal_flux_constraints(dof_handler,interface.get_bc_mapping(),constraints);
}



template <int dim, int spacedim, typename LAC>
void PDEHandler<dim, spacedim, LAC>::assemble_matrices
(const std::vector<double>                   &coefficients,
 const std::vector<std::shared_ptr<typename LAC::VectorType>>  &input_vectors)
{
  solutions = input_vectors;
  interface.set_jacobian_coefficients(coefficients);

  assemble_matrices();
}

template <int dim, int spacedim, typename LAC>
void PDEHandler<dim, spacedim, LAC>::assemble_matrices()
{
  auto _timer = computing_timer.scoped_timer ("Assemble matrices");

  signals.begin_assemble_matrices();

  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);

  FEValuesCache<dim,spacedim> fev_cache(interface.get_fe_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());

  interface.solution_preprocessing(fev_cache);

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;


  for (unsigned int i=0; i<n_matrices; ++i)
    *(matrices[i]) = 0;

  auto local_copy = [this]
                    (const pidomus::CopyData & data)
  {

    for (unsigned int i=0; i<n_matrices; ++i)
      this->constraints[i]->distribute_local_to_global (data.local_matrices[i],
                                                        data.local_dof_indices,
                                                        *(this->matrices[i]));
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    this->interface.assemble_local_matrices(cell, scratch, data);
  };



  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));

  for (unsigned int i=0; i<n_matrices; ++i)
    matrices[i]->compress(VectorOperation::add);
}



template<int dim, int spacedim, typename LAC>
typename LAC::VectorType &PDEHandler<dim,spacedim,LAC>::v(const std::string &vec_name) const
{
  Assert(interface.solution_index.find(vec_name) != interface.solution_index.end(),
         ExcMessage("Vector not found"));
  return *solutions[interface.solution_index.at(vec_name)];
}



template<int dim, int spacedim, typename LAC>
typename  LAC::BlockMatrix &PDEHandler<dim,spacedim,LAC>::m(const std::string &matrix_name) const
{
  Assert(interface.matrix_index.find(matrix_name) != interface.matrix_index.end(),
         ExcMessage("Matrix not found"));
  return *matrices[interface.matrix_index.at(matrix_name)];
}



template<int dim, int spacedim, typename LAC>
typename LAC::VectorType &PDEHandler<dim,spacedim,LAC>::v(const unsigned int vec_index) const
{
  Assert(vec_index < interface.n_vectors,
         ExcIndexRange(vec_index, 0, interface.n_vectors));
  return *solutions[vec_index];
}



template<int dim, int spacedim, typename LAC>
typename LAC::BlockMatrix &PDEHandler<dim,spacedim,LAC>::m(const unsigned int matrix_index) const
{
  Assert(matrix_index < interface.n_vectors,
         ExcIndexRange(matrix_index, 0, interface.n_vectors));
  return *matrices[matrix_index];
}


template<int dim, int spacedim, typename LAC>
ConstraintMatrix &PDEHandler<dim,spacedim,LAC>::C(const unsigned int constraint_index) const
{
  Assert(constraint_index < interface.n_matrices,
         ExcIndexRange(constraint_index, 0, interface.n_matrices));
  return *constraints[constraint_index];
}



template<int dim, int spacedim, typename LAC>
void PDEHandler<dim,spacedim,LAC>::interpolate_or_project(const Function<spacedim> &f, typename LAC::VectorType &v)
{
  if (fe->has_support_points())
    {
      VectorTools::interpolate(interface.get_default_mapping(), *dof_handler, f, v);
    }
  else if (!we_are_parallel)
    {
      const QGauss<dim> quadrature_formula(fe->degree + 1);
      VectorTools::project(interface.get_default_mapping(), *dof_handler, C(0),
                           quadrature_formula, f, v);
    }
  else
    {
      Point<spacedim> p;
      Vector<double> vals(interface.n_components);
      f.vector_value(p, vals);

      unsigned int comp = 0;
      for (unsigned int b=0; b<v.n_blocks(); ++b)
        {
          v.block(b) = vals[comp];
          comp += fe->element_multiplicity(b);
        }
    }
}



template<int dim, int spacedim, typename LAC>
void PDEHandler<dim,spacedim,LAC>::solve(typename LAC::VectorType &dst,
                                         const typename LAC::VectorType &src) const
{
  bool done = false;
  if (use_direct_solver) // try direct solver if we were asked so
    done = solve_direct(dst, src);

  if (done == false) // revert to iterative solver otherwise
    {
      try
        {
          iterative_solver.vmult(dst, src);
        }
      catch (const std::exception &e)
        {
          if (enable_finer_preconditioner)
            {
              finer_iterative_solver.vmult(dst,src);
            }
          else
            {
              AssertThrow(false,ExcMessage(e.what()));
            }

        }
    }
}


template<int dim, int spacedim, typename LAC>
bool PDEHandler<dim, spacedim, LAC>::solve_direct(typename LADealII::VectorType &dst,
                                                  const typename LADealII::VectorType &src) const
{
  if (we_are_parallel)
    return false;
  else
    {
      direct_umfpack.vmult(dst,src);
      return true;
    }
}


template<int dim, int spacedim, typename LAC>
bool PDEHandler<dim, spacedim, LAC>::solve_direct(typename LATrilinos::VectorType &dst,
                                                  const typename LATrilinos::VectorType &src) const
{
  if (interface.n_components == 1)
    {
      Assert(direct_trilinos, ExcInternalError());
      direct_trilinos->solve(dst.block(0),src.block(0));
      return true;
    }
  else
    return false;
}


template<int dim, int spacedim, typename LAC>
void PDEHandler<dim, spacedim, LAC>::initialize_solver()
{
  bool done = false;
  if (use_direct_solver)
    done = factorize_matrix(m(0));

  if (done == false)
    {
      interface.compute_system_operators(matrices, iterative_solver.op, iterative_solver.prec,
                                         finer_iterative_solver.prec);

      finer_iterative_solver.op = iterative_solver.op;

      iterative_solver.parse_parameters_call_back();
      finer_iterative_solver.parse_parameters_call_back();
    }
}



template<int dim, int spacedim, typename LAC>
bool PDEHandler<dim, spacedim, LAC>::factorize_matrix(LADealII::BlockMatrix &A)
{
  direct_umfpack.initialize(A);
  return true;
}


template<int dim, int spacedim, typename LAC>
bool PDEHandler<dim, spacedim, LAC>::factorize_matrix(LATrilinos::BlockMatrix &A)
{
  if (interface.n_components == 1)
    {
      direct_solver_control = UP(new SolverControl());
      direct_trilinos = UP(new TrilinosWrappers::SolverDirect(*direct_solver_control,
                                                              TrilinosWrappers::SolverDirect::AdditionalData(false,
                                                                  direct_solver_type)));
      direct_trilinos->initialize(A.block(0,0));
      return true;
    }
  else
    return false;
}



template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim, spacedim, LAC>::residual(const std::vector<double>                                     &coefficients,
                                         const std::vector<std::shared_ptr<typename LAC::VectorType> > &input_vectors,
                                         typename LAC::VectorType                                      &residual_vector)
{
  auto _timer = computing_timer.scoped_timer ("Residual");

  signals.begin_residual();

  // syncronize(t,solution,solution_dot);
  solutions = input_vectors;
  interface.set_jacobian_coefficients(coefficients);

  residual(residual_vector);
}


template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim, spacedim, LAC>::residual(typename LAC::VectorType &residual_vector)
{
  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);


  FEValuesCache<dim,spacedim> fev_cache(interface.get_fe_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());

  interface.solution_preprocessing(fev_cache);

  residual_vector = 0;

  auto local_copy = [&residual_vector, this] (const pidomus::CopyData & data)
  {

    this->constraints[0]->distribute_local_to_global (data.local_residual,
                                                      data.local_dof_indices,
                                                      residual_vector);
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    this->interface.assemble_local_system_residual(cell,scratch,data);
    // apply conservative loads
    this->apply_forcing_terms(cell, scratch, data.local_residual);

    if (cell->at_boundary())
      this->apply_neumann_bcs(cell, scratch, data.local_residual);
  };

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));


  residual_vector.compress(VectorOperation::add);

//  auto id = solutions[0]->locally_owned_elements();
//  for (unsigned int i = 0; i < id.n_elements(); ++i)
//    {
//      auto j = id.nth_index_in_set(i);
//      if (constraints[0]->is_constrained(j))
//        residual_vector[j] = (*solutions[0])(j) - (*locally_relevant_solutions[0])(j);
//    }

//  residual_vector.compress(VectorOperation::insert);

  signals.end_residual();
}


template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim, spacedim, LAC>::
declare_parameters (ParameterHandler &prm)
{
  add_parameter(  prm,
                  &initial_global_refinement,
                  "Initial global refinement",
                  "1",
                  Patterns::Integer (0));

  prm.enter_subsection("Solvers -- Direct solver");
  add_parameter(  prm,
                  &use_direct_solver,
                  "Use direct solver if available",
                  "true",
                  Patterns::Bool());

  add_parameter( prm,
                 &direct_solver_type,
                 "Direct solver type",
                 "umfpack",
                 Patterns::Selection(std::is_same<LAC,LADealII>::value ?
                                     "umfpack" : "Amesos_Lapack|Amesos_Scalapack|"
                                     "Amesos_Klu|Amesos_Umfpack|Amesos_Pardiso|"
                                     "Amesos_Taucs|Amesos_Superlu|Amesos_Superludist|"
                                     "Amesos_Dscpack|Amesos_Mumps"));

  add_parameter(  prm,
                  &enable_finer_preconditioner,
                  "Enable finer preconditioner",
                  "false",
                  Patterns::Bool());

  prm.leave_subsection();

  add_parameter(  prm,
                  &adaptive_refinement,
                  "Adaptive refinement",
                  "true",
                  Patterns::Bool());

  add_parameter(  prm,
                  &verbose,
                  "Print some useful informations about processes",
                  "true",
                  Patterns::Bool());
}


template <int dim, int spacedim, typename LAC>
void PDEHandler<dim, spacedim, LAC>::update_functions_and_constraints (const double &t, const double &time_step)
{
  auto _timer = computing_timer.scoped_timer ("Update functions and constraints");
  current_time = t;
  current_time_step = time_step;

  if (!std::isnan(t))
    {
      dirichlet_bcs.set_time(t);
      forcing_terms.set_time(t);
      neumann_bcs.set_time(t);
    }

  // clear previously stored constraints
  for (unsigned int i=0; i<n_matrices; ++i)
    constraints[i]->clear();

  // compute hanging nodes
  DoFTools::make_hanging_node_constraints (*dof_handler,
                                           C(0));

  zero_average.apply_zero_average_constraints(*dof_handler, C(0));

  // compute boundary values for the system matrix
  apply_dirichlet_bcs(*dof_handler, dirichlet_bcs, C(0));

  // apply zero average constraints to the system matrix
  zero_average.apply_zero_average_constraints(*dof_handler, C(0));

  // add user-supplied constraints
  // signals.update_constraint_matrices(constraints);

  // close the constraints
  for (unsigned int i=0; i<n_matrices; ++i)
    constraints[i]->close();
}


template<int dim, int spacedim, typename LAC>
void
PDEHandler<dim,spacedim,LAC>::
output_solution (const std::string &suffix) const
{
  data_out.prepare_data_output( *dof_handler,
                                suffix);
  auto cnames = interface.component_names;
  for (unsigned int i=0; i<solutions.size(); ++i)
    *locally_relevant_solutions[i] = *solutions[i];

  for (unsigned int i=0; i<solutions.size(); ++i)
    {
      std::vector<std::string> names(cnames.size(), interface.solution_names[i]);
      for (unsigned int j=0; j<cnames.size(); ++j)
        names[j] += "_" + cnames[j];
      data_out.add_data_vector (*locally_relevant_solutions[i], print(names));
    }

  data_out.write_data_and_clear(interface.get_output_mapping());
}


#define INSTANTIATE(dim,spacedim,LAC) \
  template class PDEHandler<dim,spacedim,LAC>;




PIDOMUS_INSTANTIATE(INSTANTIATE)


