#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <vector>

using namespace dealii;

// Define the problem class
template <int dim>
class AdvectionDiffusionProblem : public Function<dim>
{
public:
  AdvectionDiffusionProblem() : Function<dim>(1) {}

  double coefficient(const Point<dim> &p)
  {
    // Modify as needed based on the desired coefficients
    if (p.square() < inner_distance * inner_distance)
      return 20;
    else
      return 1;
  }

  virtual double value(const Point<dim> &p, const unsigned int /*component*/) const override
  {
    const double advection_speed = 0.5;
    const double diffusivity = 0.01;

    return std::exp(-50.0 * (std::pow(p[0] - 0.5, 2) + std::pow(p[1] - 0.5, 2))) +
           0.2 * std::sin(2.0 * numbers::PI * p[0]) +
           advection_speed * p[0] + diffusivity * p[1];
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component) const override
  {
    const double advection_speed = 0.5;
    const double diffusivity = 0.01;

    Tensor<1, dim> grad;

    grad[0] = -50.0 * (p[0] - 0.5) * std::exp(-50.0 * (std::pow(p[0] - 0.5, 2) + std::pow(p[1] - 0.5, 2))) +
              numbers::PI * 0.1 * std::cos(2.0 * numbers::PI * p[0]);
    grad[1] = -50.0 * (p[1] - 0.5) * std::exp(-50.0 * (std::pow(p[0] - 0.5, 2) + std::pow(p[1] - 0.5, 2)));

    if (component == 0)
    {
      grad[0] *= advection_speed;
      grad[1] *= advection_speed;
    }

    if (component == 0)
    {
      grad[0] -= diffusivity * 2.0 * numbers::PI * std::cos(2.0 * numbers::PI * p[0]) * 0.1 * std::sin(2.0 * numbers::PI * p[0]);
      grad[1] -= diffusivity * 0.1 * std::cos(2.0 * numbers::PI * p[0]);
    }

    return grad;
  }

  // Add public members for inner and outer distances
  double inner_distance = 0.2;
  double outer_distance = 1.0;
};

// Define the main class for the simulation
template <int dim>
class Step6
{
public:
  Step6();

  void run();
  void export_mesh_info(const unsigned int cycle) const;
  void export_solver_info(const unsigned int cycle,
                          const unsigned int iterations,
                          const double cpu_time) const;

private:
  void setup_system();
  void create_initial_mesh();
  void assemble_system();
  void solve();
  void refine_grid();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;

  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  SparseMatrix<double> system_matrix;
  SparsityPattern sparsity_pattern;

  Vector<double> solution;
  Vector<double> system_rhs;

  const double inflow_velocity = 0.1;

  AdvectionDiffusionProblem<dim> advection_diffusion_problem;

  std::vector<unsigned int> active_cells_per_cycle;
};

// Implementations of the member functions
template <int dim>
Step6<dim>::Step6()
    : fe(2),
      dof_handler(triangulation)
{
}
template <int dim>
void Step6<dim>::create_initial_mesh()
{
  const unsigned int n_cells_x =6; // Adjust the number of cells in x-direction
  const unsigned int n_cells_y = 6; // Adjust the number of cells in y-direction

  GridGenerator::subdivided_hyper_rectangle(triangulation, {n_cells_x, n_cells_y},
                                           Point<dim>(-advection_diffusion_problem.outer_distance, -advection_diffusion_problem.outer_distance),
                                           Point<dim>(advection_diffusion_problem.outer_distance, advection_diffusion_problem.outer_distance));

  // Refine the cells based on some criterion
  for (unsigned int step = 0; step < 2; ++step)
  {
    for (auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->diameter() > 0.1)
        cell->set_refine_flag();
    }

    triangulation.execute_coarsening_and_refinement();
  }
}

template <int dim>
void Step6<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  false);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void Step6<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
      const double current_coefficient =
          advection_diffusion_problem.coefficient(fe_values.quadrature_point(q_index));
      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
              (current_coefficient *
               fe_values.shape_grad(i, q_index) *
               fe_values.shape_grad(j, q_index) *
               fe_values.JxW(q_index));

        cell_rhs(i) += (inflow_velocity *
                        fe_values.shape_value(i, q_index) *
                        fe_values.JxW(q_index));
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }
}

template <int dim>
void Step6<dim>::solve()
{
  Timer timer;
  timer.start();

  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

PreconditionJacobi<SparseMatrix<double>> preconditioner;
preconditioner.initialize(system_matrix);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  timer.stop();

  constraints.distribute(solution);

  export_solver_info(triangulation.n_active_cells(), solver_control.last_step(), timer.wall_time());
}

template <int dim>
void Step6<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     {},
                                     solution,
                                     estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                    estimated_error_per_cell,
                                                    0.75,
                                                    0.005);

  triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{
  {
    GridOut grid_out;
    std::ofstream output("grid-" + std::to_string(cycle) + ".vtk");
    grid_out.write_vtk(triangulation, output);
  }

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
  }
}

template <int dim>
void Step6<dim>::export_mesh_info(const unsigned int cycle) const
{
  std::ofstream mesh_info_file("mesh_info.txt", std::ios::app);
  mesh_info_file << cycle << ' ' << triangulation.n_active_cells() << '\n';
  mesh_info_file.close();
}

template <int dim>
void Step6<dim>::export_solver_info(const unsigned int active_cells,
                                    const unsigned int iterations,
                                    const double cpu_time) const
{
  std::ofstream solver_info_file("solver_info.txt", std::ios::app);
  solver_info_file << active_cells << ' ' << iterations << ' ' << cpu_time << '\n';
  solver_info_file.close();
}

template <int dim>
void Step6<dim>::run()
{
  for (unsigned int cycle = 0; cycle < 10; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    if (cycle == 0)
    {
      create_initial_mesh();
    }
    else
      refine_grid();

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    active_cells_per_cycle.push_back(triangulation.n_active_cells());

    setup_system();

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    assemble_system();
    solve();
    output_results(cycle);
    export_mesh_info(cycle);
  }
}

// Main function
int main()
{
  try
  {
    Step6<2> advection_diffusion_problem_2d;
    advection_diffusion_problem_2d.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
       return 1;
  }

  return 0;
}

