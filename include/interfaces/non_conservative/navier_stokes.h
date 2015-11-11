#ifndef _navier_stokes_h_
#define _navier_stokes_h_

/**
 *  This interface solves a Navier Stokes Equation:
 *  \f[
 *     \partial_t u + (u\cdot\nabla)u - \nu\textrm{div} \epsilon(u)
 *     + \frac{1}{\rho}\grad p = f
 *  \f]
 *  where \f$ \epsilon(u) = \frac{\nabla u + [\nabla u]^t}{2}. \f$
 */

#include "interfaces/non_conservative.h"
#include <deal2lkit/parsed_function.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal2lkit/fe_values_cache.h>
#include <deal2lkit/parsed_function.h>

template <int dim>
class NavierStokes : public NonConservativeInterface<dim,dim,dim+1, NavierStokes<dim> >
{
public:
  typedef FEValuesCache<dim,dim> Scratch;
  typedef Assembly::CopyData::piDoMUSPreconditioner<dim,dim> CopyPreconditioner;
  typedef Assembly::CopyData::piDoMUSSystem<dim,dim> CopySystem;
  typedef TrilinosWrappers::MPI::BlockVector VEC;

  /* specific and useful functions for this very problem */
  NavierStokes(std::string prec="default");

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  template<typename Number>
  void preconditioner_residual(const typename DoFHandler<dim>::active_cell_iterator &,
                               Scratch &,
                               CopyPreconditioner &,
                               std::vector<Number> &local_residual) const;

  template<typename Number>
  void system_residual(const typename DoFHandler<dim>::active_cell_iterator &,
                       Scratch &,
                       CopySystem &,
                       std::vector<Number> &local_residual) const;

  virtual void compute_system_operators(const DoFHandler<dim> &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        const TrilinosWrappers::BlockSparseMatrix &,
                                        LinearOperator<VEC> &,
                                        LinearOperator<VEC> &) const;

private:

  double rho;
  double nu;
  double gamma;

  std::string prec_name;

  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> Mp_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionAMG>    Kp_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> T_preconditioner;

};

template<int dim>
NavierStokes<dim>::NavierStokes(std::string prec) :
  NonConservativeInterface<dim,dim,dim+1,NavierStokes<dim> >("Navier Stokes",
                                                             "FESystem[FE_Q(2)^d-FE_Q(1)]",
                                                             "u,u,p", "1,1; 1,0", "1,0; 0,1","1,0"),
  prec_name(prec)
{};


template <int dim>
template<typename Number>
void NavierStokes<dim>::preconditioner_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                Scratch &fe_cache,
                                                CopyPreconditioner &data,
                                                std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution",      *this->solution,      alpha);
  fe_cache.cache_local_solution_vector("solution_dot",  *this->solution_dot,  alpha);

  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);

  auto &us          = fe_cache.get_values(
                        "solution",     "u",      velocity,       alpha);
  auto &div_us      = fe_cache.get_divergences(           "solution",     "div_u",  velocity,       alpha);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients(
                        "solution",     "sym_grad_u", velocity,       alpha);
  auto &grad_us = fe_cache.get_gradients(
                    "solution",     "grad_u", velocity,       alpha);
  auto &us_dot      = fe_cache.get_values(
                        "solution_dot", "u_dot",  velocity,       alpha);
  auto &ps          = fe_cache.get_values(
                        "solution",     "p",      pressure,       alpha);
  auto &grad_ps = fe_cache.get_gradients(
                    "solution",     "grad_p", pressure,       alpha);

  const unsigned int n_q_points = ps.size();

  // Jacobian:
  auto &JxW = fe_cache.get_JxW_values();

  // Init residual to 0
  for (unsigned int i=0; i<local_residual.size(); ++i)
    local_residual[i] = 0;

  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      // variables:
      const Tensor<1, dim, Number> &u           = us[q];
      const Tensor<1, dim, Number> &u_dot       = us_dot[q];
      const Number                 &div_u      = div_us[q];
      const Tensor<2, dim, Number> &sym_grad_u  = sym_grad_us[q];
      const Tensor<2, dim, Number> &grad_u      = grad_us[q];
      const Tensor<1, dim, Number> &grad_p      = grad_ps[q];
      const Number                 &p           = ps[q];

      for (unsigned int i=0; i<local_residual.size(); ++i)
        {
          // test functions:
          auto v      = fev[velocity     ].value(i,q);
          auto div_v  = fev[velocity     ].divergence(i,q);
          auto sym_grad_v = fev[velocity ].symmetric_gradient(i,q);
          auto grad_v = fev[velocity     ].gradient(i,q);
          auto m      = fev[pressure     ].value(i,q);
          auto grad_q = fev[pressure     ].gradient(i,q);

          // compute residual:
          if (prec_name=="default" || prec_name=="diag")
            {
              local_residual[i] +=  (
                                      u_dot * v +
                                      gamma * div_u * div_v +
                                      scalar_product(transpose(grad_u)*u, v) +
                                      nu * scalar_product(sym_grad_u,sym_grad_v) +
                                      (1./rho)*p*m
                                    )*JxW[q];
            }
          else if (prec_name=="linear" || prec_name=="linear_diag")
            {
              local_residual[i] +=  (
                                      u_dot * v +
                                      gamma * div_u * div_v  +
                                      scalar_product(transpose(grad_u)*u, v) +
                                      nu * scalar_product(sym_grad_u,sym_grad_v) +
                                      (1./rho)*p*m
                                    )*JxW[q];

            }
          else if ( prec_name=="BFBt_identity"  ||
                    prec_name=="BFBt_diagA"     ||
                    prec_name=="cahouet-chabard"||
                    prec_name=="diag_cahouet-chabard")
            {
              local_residual[i] +=  (
                                      u_dot * v +
                                      gamma * div_u * div_v +
                                      scalar_product(transpose(grad_u)*u, v) +
                                      nu * scalar_product(sym_grad_u,sym_grad_v) +
                                      (1./rho)*grad_p*grad_q
                                    )*JxW[q];
            }
        }

    }

}


template <int dim>
template<typename Number>
void
NavierStokes<dim>::
system_residual(const typename DoFHandler<dim>::active_cell_iterator &cell,
                Scratch &fe_cache,
                CopySystem &data,
                std::vector<Number> &local_residual) const
{
  Number alpha = this->alpha;
  fe_cache.reinit(cell);

  fe_cache.cache_local_solution_vector("solution",      *this->solution,      alpha);
  fe_cache.cache_local_solution_vector("solution_dot",  *this->solution_dot,  alpha);

  this->fix_solution_dot_derivative(fe_cache, alpha);

  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);


  // velocity:
  auto &us          = fe_cache.get_values(                "solution",     "u",      velocity,       alpha);
  auto &div_us      = fe_cache.get_divergences(           "solution",     "div_u",  velocity,       alpha);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients(   "solution",     "sym_grad_u",      velocity,       alpha);
  auto &grad_us     = fe_cache.get_gradients(   "solution",     "grad_u",          velocity,       alpha);
  auto &us_dot      = fe_cache.get_values(                "solution_dot", "u_dot",  velocity,       alpha);

  // pressure:
  auto &ps          = fe_cache.get_values(                "solution",     "p",      pressure,       alpha);

  // Jacobian:
  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_q_points = ps.size();

  // Init residual to 0
  for (unsigned int i=0; i<local_residual.size(); ++i)
    local_residual[i] = 0;

  auto &fev = fe_cache.get_current_fe_values();
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      // variables:
      //    velocity:
      const Tensor<1, dim, Number>  &u          = us[q];
      const Number                  &div_u      = div_us[q];
      const Tensor<1, dim, Number>  &u_dot      = us_dot[q];
      const Tensor <2, dim, Number> &sym_grad_u = sym_grad_us[q];
      const Tensor <2, dim, Number> &grad_u     = grad_us[q];

      //    pressure:
      const Number                  &p          = ps[q];

      for (unsigned int i=0; i<local_residual.size(); ++i)
        {
          // test functions:
          //    velocity:
          auto v      = fev[velocity    ].value(i,q);
          auto grad_v = fev[velocity    ].gradient(i,q);
          auto sym_grad_v = fev[velocity ].symmetric_gradient(i,q);
          auto div_v  = fev[velocity    ].divergence(i,q);

          //    pressure:
          auto m      = fev[pressure    ].value(i,q);

          // compute residual:
          local_residual[i] += (
                                 u_dot * v +
                                 scalar_product(transpose(grad_u)*u, v) +
                                 gamma * div_u * div_v +
                                 nu * scalar_product(sym_grad_u,sym_grad_v) +
                                 (1./rho)*p*div_v
                               )*JxW[q];
        }
    }
}




template <int dim>
void
NavierStokes<dim>::
declare_parameters (ParameterHandler &prm)
{
  NonConservativeInterface<dim,dim,dim+1, NavierStokes<dim> >::declare_parameters(prm);
  this->add_parameter(prm, &rho,         "rho [kg m^3]",  "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &nu,          "nu [Pa s]",     "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &gamma,       "grad-div stabilization",     "1.0", Patterns::Double(0.0));
  this->add_parameter(prm, &prec_name,   "Preconditioner","default",
                      Patterns::Selection("default|linear|diag|linear_diag|BFBt_identity|BFBt_diagA|cahouet-chabard|diag_cahouet-chabard"));
}

template <int dim>
void
NavierStokes<dim>::
parse_parameters_call_back ()
{
  NonConservativeInterface<dim,dim,dim+1, NavierStokes<dim> >::parse_parameters_call_back();
}


template <int dim>
void
NavierStokes<dim>::compute_system_operators(const DoFHandler<dim> &dh,
                                            const TrilinosWrappers::BlockSparseMatrix &matrix,
                                            const TrilinosWrappers::BlockSparseMatrix &preconditioner_matrix,
                                            LinearOperator<VEC> &system_op,
                                            LinearOperator<VEC> &prec_op) const
{
  auto alpha = this->alpha;

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector velocity_components(0);
  DoFTools::extract_constant_modes (dh, dh.get_fe().component_mask(velocity_components),
                                    constant_modes);

  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,0) );
  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,1) );
  //  auto B =  transpose_operator(Bt);
  auto B  = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,0) );
  auto ZeroP = 0*linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(1,1) );



  // ASSEMBLE THE PROBLEM:
  Assert(prec_name != "", ExcNotInitialized());
  if (prec_name=="default" || prec_name == "linear")
    {
      Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
      Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
      Amg_data.constant_modes = constant_modes;
      Amg_data.elliptic = true;
      Amg_data.higher_order_elements = true;
      Amg_data.smoother_sweeps = 2;
      Amg_data.aggregation_threshold = 0.02;

      Mp_preconditioner->initialize (preconditioner_matrix.block(1,1));
      Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                      Amg_data);

      auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >
                   ( preconditioner_matrix.block(1,1) );

      static ReductionControl solver_control_pre(5000, 1e-8);
      static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
      auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
      auto Schur_inv = inverse_operator( Mp, solver_CG, *Mp_preconditioner);

      auto P00 = A_inv;
      auto P01 = null_operator(Bt);
      auto P10 = Schur_inv * B * A_inv;
      auto P11 = -1 * Schur_inv;

      prec_op = block_operator<2, 2, VEC >({{
          {{ P00, P01 }} ,
          {{ P10, P11 }}
        }
      });

    }
  else if (prec_name=="diag" || prec_name == "linear_diag")
    {
      Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
      Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
      Amg_data.constant_modes = constant_modes;
      Amg_data.elliptic = true;
      Amg_data.higher_order_elements = true;
      Amg_data.smoother_sweeps = 2;
      Amg_data.aggregation_threshold = 0.02;

      Mp_preconditioner->initialize (preconditioner_matrix.block(1,1));
      Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                      Amg_data);

      auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >
                   ( preconditioner_matrix.block(1,1) );

      static ReductionControl solver_control_pre(5000, 1e-8);
      static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
      auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
      auto Schur_inv = inverse_operator( Mp, solver_CG, *Mp_preconditioner);

      auto P00 = A_inv;
      auto P01 = null_operator(Bt);
      auto P10 = null_operator(B);
      auto P11 = -1 * Schur_inv;

      prec_op = block_operator<2, 2, VEC >({{
          {{ P00, P01 }} ,
          {{ P10, P11 }}
        }
      });

    }
  else if (prec_name=="cahouet-chabard")
    {
      Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
      Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
      Kp_preconditioner.reset  (new TrilinosWrappers::PreconditionAMG());

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
      Amg_data.constant_modes = constant_modes;
      Amg_data.elliptic = true;
      Amg_data.higher_order_elements = true;
      Amg_data.smoother_sweeps = 2;
      Amg_data.aggregation_threshold = 0.02;

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_p;

      Amg_data_p.elliptic = true;
      Amg_data_p.higher_order_elements = true;
      Amg_data_p.smoother_sweeps = 2;
      Amg_data_p.aggregation_threshold = 0.02;

      Kp_preconditioner->initialize (preconditioner_matrix.block(1,1),
                                     Amg_data_p);
      Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                      Amg_data);
      Mp_preconditioner->initialize (matrix.block(1,1));

      auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >
                   (matrix.block(1,1) );
      auto Kp    = linear_operator< TrilinosWrappers::MPI::Vector >
                   (preconditioner_matrix.block(1,1) );

      static ReductionControl solver_control_pre(5000, 1e-8);
      static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
      auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
      auto Mp_inv    = inverse_operator( Mp, solver_CG, *Mp_preconditioner);
      auto Kp_inv    = inverse_operator( Kp, solver_CG, *Kp_preconditioner);
      auto Schur_inv = nu * Mp_inv + alpha * Kp_inv;

      auto P00 = A_inv;
      auto P01 = null_operator(Bt);
      auto P10 = Schur_inv * B * A_inv;
      auto P11 = -1 * Schur_inv;

      prec_op = block_operator<2, 2, VEC >({{
          {{ P00, P01 }} ,
          {{ P10, P11 }}
        }
      });
    }
  else if (prec_name=="diag_cahouet-chabard")
    {
      Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
      Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
      Kp_preconditioner.reset  (new TrilinosWrappers::PreconditionAMG());

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
      Amg_data.constant_modes = constant_modes;
      Amg_data.elliptic = true;
      Amg_data.higher_order_elements = true;
      Amg_data.smoother_sweeps = 2;
      Amg_data.aggregation_threshold = 0.02;

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_p;
      Amg_data_p.elliptic = true;
      Amg_data_p.higher_order_elements = true;
      Amg_data_p.smoother_sweeps = 2;
      Amg_data_p.aggregation_threshold = 0.02;

      Kp_preconditioner->initialize (preconditioner_matrix.block(1,1),
                                     Amg_data_p);
      Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                      Amg_data);
      Mp_preconditioner->initialize (matrix.block(1,1));

      auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >
                   (matrix.block(1,1) );
      auto Kp    = linear_operator< TrilinosWrappers::MPI::Vector >
                   (preconditioner_matrix.block(1,1) );

      static ReductionControl solver_control_pre(5000, 1e-8);
      static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
      auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);
      auto Mp_inv    = inverse_operator( Mp, solver_CG, *Mp_preconditioner);
      auto Kp_inv    = inverse_operator( Kp, solver_CG, *Kp_preconditioner);
      auto Schur_inv = nu * Mp_inv + alpha * Kp_inv;

      auto P00 = A_inv;
      auto P01 = null_operator(Bt);
      auto P10 = null_operator(B);
      auto P11 = -1 * Schur_inv;

      prec_op = block_operator<2, 2, VEC >({{
          {{ P00, P01 }} ,
          {{ P10, P11 }}
        }
      });
    }
  else if (prec_name=="BFBt_identity")
    {
      Kp_preconditioner.reset  (new TrilinosWrappers::PreconditionAMG());
      Amg_preconditioner.reset  (new TrilinosWrappers::PreconditionAMG());

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
      Amg_data.constant_modes = constant_modes;
      Amg_data.elliptic = true;
      Amg_data.higher_order_elements = true;
      Amg_data.smoother_sweeps = 2;
      Amg_data.aggregation_threshold = 0.02;

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_p;

      Amg_data_p.elliptic = true;
      Amg_data_p.higher_order_elements = true;
      Amg_data_p.smoother_sweeps = 1;
      Amg_data_p.aggregation_threshold = 0.2;

      Kp_preconditioner->initialize (preconditioner_matrix.block(1,1),
                                     Amg_data_p);
      Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                      Amg_data);


      static ReductionControl solver_control_pre(5000, 1e-8);
      static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
      auto A_inv     = inverse_operator( A,   solver_CG, *Amg_preconditioner);
      auto BBt       = B*Bt;
      auto BBt_inv   = inverse_operator( BBt, solver_CG, *Kp_preconditioner);
      auto Schur_inv = BBt_inv * B * A * Bt * BBt_inv;

      auto P00 = A_inv;
      auto P01 = null_operator(Bt);
      auto P10 = Schur_inv * B * A_inv;
      auto P11 = -1 * Schur_inv;

      prec_op = block_operator<2, 2, VEC >({{
          {{ P00, P01 }} ,
          {{ P10, P11 }}
        }
      });
    }
  else if (prec_name=="BFBt_diagA")
    {
      Kp_preconditioner.reset  (new TrilinosWrappers::PreconditionAMG());
      Amg_preconditioner.reset  (new TrilinosWrappers::PreconditionAMG());

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
      Amg_data.constant_modes = constant_modes;
      Amg_data.elliptic = true;
      Amg_data.higher_order_elements = true;
      Amg_data.smoother_sweeps = 2;
      Amg_data.aggregation_threshold = 0.02;

      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_p;

      Amg_data_p.elliptic = true;
      Amg_data_p.higher_order_elements = true;
      Amg_data_p.smoother_sweeps = 2;
      Amg_data_p.aggregation_threshold = 0.02;

      Kp_preconditioner->initialize (preconditioner_matrix.block(1,1),
                                     Amg_data_p);
      Amg_preconditioner->initialize (preconditioner_matrix.block(0,0),
                                      Amg_data);


      static ReductionControl solver_control_pre(5000, 1e-8);
      static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);

      auto inv_diag_A  = linear_operator< TrilinosWrappers::MPI::Vector >( matrix.block(0,0) );
      inv_diag_A.vmult = [&matrix](TrilinosWrappers::MPI::Vector &v, const TrilinosWrappers::MPI::Vector &u)
      {
        // for (unsigned int i = 0; i<matrix.block(0,0).m(); ++i)
        for (auto i : v.locally_owned_elements())
          v(i)=u(i)/matrix.block(0,0)(i,i);
      };

      auto A_inv     = inverse_operator( A,   solver_CG, *Amg_preconditioner);
      auto BBt       = B*inv_diag_A*Bt;
      auto BBt_inv   = inverse_operator( BBt, solver_CG, *Kp_preconditioner);
      auto Schur_inv = BBt_inv * B * inv_diag_A *A * inv_diag_A * Bt * BBt_inv;

      auto P00 = A_inv;
      auto P01 = null_operator(Bt);
      auto P10 = Schur_inv * B * A_inv;
      auto P11 = -1 * Schur_inv;

      prec_op = block_operator<2, 2, VEC >({{
          {{ P00, P01 }} ,
          {{ P10, P11 }}
        }
      });
    }
  else
    {
      AssertThrow(false, ExcMessage("Preconditioner not recognized."));
    }

  //const auto S = linear_operator<VEC>(matrix);

  system_op  = block_operator<2, 2, VEC >({{
      {{ A, Bt }} ,
      {{ B, ZeroP }}
    }
  });
}


template class NavierStokes <2>;
template class NavierStokes <3>;

#endif