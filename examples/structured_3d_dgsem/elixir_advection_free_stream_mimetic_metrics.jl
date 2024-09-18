using LinearAlgebra
using OrdinaryDiffEq
using Trixi
using StaticArrays
using Plots

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

# Mapping as described in https://arxiv.org/abs/2012.12040
function mapping(xi, eta, zeta)
  # Transform input variables between -1 and 1 onto our crazy domain

  y = eta + 0.1 * (cos(pi * xi) *
                   cos(pi * eta) *
                   cos(pi * zeta))

  x = xi + 0.1 * (cos(pi * xi) *
                   cos(pi * eta) *
                   cos(pi * zeta))

  z = zeta + 0.1 * (cos(pi * xi) *
                   cos(pi * eta) *
                   cos(pi * zeta))

  #= x = xi
  y = eta
  z = zeta =#
  return SVector(x, y, z)
end

function exact_contravariant_vectors!(Ja, xi, eta, zeta, dxi, deta, dzeta)
    theta_xi = theta_der1(xi, eta, zeta)
    theta_eta = theta_der2(xi, eta, zeta)
    theta_zeta = theta_der3(xi, eta, zeta)
    Ja[1,1] = deta*dzeta*(1 + theta_eta + theta_zeta)
    Ja[1,2] = dxi*dzeta*(-theta_xi)
    Ja[1,3] = dxi*deta*(-theta_xi)
    Ja[2,1] = deta*dzeta*(-theta_eta)
    Ja[2,2] = dxi*dzeta*(1 + theta_xi + theta_zeta)
    Ja[2,3] = dxi*deta*(-theta_eta)
    Ja[3,1] = deta*dzeta*(-theta_zeta)
    Ja[3,2] = dxi*dzeta*(-theta_zeta)
    Ja[3,3] = dxi*deta*(1 + theta_xi + theta_eta)
end

function compute_error(solver, semi, cells_per_dimension)
  @unpack nodes, weights = solver.basis
  exact_Ja = zero(MMatrix{3, 3, Float64})
  error = zero(Float64)
  error_L2 = zero(Float64)
  linear_indices = LinearIndices(size(semi.mesh))
  xi_scale = 1/cells_per_dimension[1]
  eta_scale = 1/cells_per_dimension[2]
  zeta_scale = 1/cells_per_dimension[3]

  for d3 in 1:cells_per_dimension[3]
    for d2 in 1:cells_per_dimension[2]
      for d1 in 1:cells_per_dimension[1]
        node_coordinates_comp = zeros(3, nnodes(solver.basis))
        Trixi.calc_node_coordinates_computational!(node_coordinates_comp, d1, d2, d3, semi.mesh, solver.basis)
        element = linear_indices[d1,d2,d3]
        for k in eachnode(solver.basis)
          for j in eachnode(solver.basis)
            for i in eachnode(solver.basis)
              exact_contravariant_vectors!(exact_Ja, node_coordinates_comp[1,i], node_coordinates_comp[2,j], node_coordinates_comp[3,k], xi_scale, eta_scale, zeta_scale)
              error = max(error, maximum(abs.(semi.cache.elements.contravariant_vectors[:,:,i,j,k,element] - exact_Ja)))
              error_L2 += norm(semi.cache.elements.contravariant_vectors[:,:,i,j,k,element] - exact_Ja) * weights[i] * weights[j] * weights[k]
            end
          end
        end
      end
    end
  end
  return error, error_L2 / (8 * prod(cells_per_dimension))
end

f(x,t,equations::LinearScalarAdvectionEquation3D) = SVector(sin(pi*(x[1]+x[2]+x[3])))

cells_per_dimension = (4,4,4)

max_polydeg = 25

errors_normals_inf = zeros(max_polydeg,3)
errors_normals_L2 = zeros(max_polydeg,3)
errors_sol_inf = zeros(max_polydeg,3)
errors_sol_L2 = zeros(max_polydeg,3)
exact_jacobian = true
final_time = 1e0
initial_condition = f

for polydeg in 1:max_polydeg
  println("Computing polydeg = ", polydeg)
  # Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
  solver = DGSEM(polydeg, flux_lax_friedrichs)

  # Create curved mesh with 8 x 8 x 8 elements
  mesh = StructuredMesh(cells_per_dimension, mapping; mimetic = false, exact_jacobian = false)

  # A semidiscre  tization collects data structures and functions for the spatial discretization
  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

  error_inf, error_L2 = compute_error(solver, semi, cells_per_dimension)
  errors_normals_inf[polydeg,1] = error_inf
  errors_normals_L2[polydeg,1] = error_L2

  # Create ODE problem with time span from 0.0 to 1.0
  ode = semidiscretize(semi, (0.0, final_time));

  # The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
  analysis_callback = AnalysisCallback(semi, interval=100, analysis_polydeg = 50)

  # The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
  stepsize_callback = StepsizeCallback(cfl=0.1)

  # Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
  callbacks = CallbackSet(analysis_callback, stepsize_callback)


  ###############################################################################
  # run the simulation

  # OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
  sol = solve(ode, Euler(), #CarpenterKennedy2N54(williamson_condition=false),
              dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
              save_everystep=false, callback=callbacks);

  
  errors = analysis_callback(sol)

  errors_sol_L2[polydeg, 1] = errors.l2[1]
  errors_sol_inf[polydeg, 1] = errors.linf[1]
  #=
  # Create curved mesh with 8 x 8 x 8 elements
  mesh = StructuredMesh(cells_per_dimension, mapping; mimetic = true, exact_jacobian = true)

  # A semidiscretization collects data structures and functions for the spatial discretization
  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

  error_inf, error_L2 = compute_error(solver, semi, cells_per_dimension)
  errors_normals_inf[polydeg,2] = error_inf
  errors_normals_L2[polydeg,2] = error_L2

  # Create ODE problem with time span from 0.0 to 1.0
  ode = semidiscretize(semi, (0.0, final_time));

  # The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
  analysis_callback = AnalysisCallback(semi, interval=100)

  # The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
  stepsize_callback = StepsizeCallback(cfl=0.1)

  # Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
  callbacks = CallbackSet(analysis_callback, stepsize_callback)


  ###############################################################################
  # run the simulation

  # OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
  sol = solve(ode, Euler(), #, CarpenterKennedy2N54(williamson_condition=false),
              dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
              save_everystep=false, callback=callbacks);

  errors = analysis_callback(sol)

  errors_sol_L2[polydeg, 2] = errors.l2[1]
  errors_sol_inf[polydeg, 2] = errors.linf[1]
  =#
  
 # Create curved mesh with 8 x 8 x 8 elements
 mesh = StructuredMesh(cells_per_dimension, mapping; mimetic = true, exact_jacobian = false)

 # A semidiscretization collects data structures and functions for the spatial discretization
 semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

 error_inf, error_L2 = compute_error(solver, semi, cells_per_dimension)
 errors_normals_inf[polydeg,3] = error_inf
 errors_normals_L2[polydeg,3] = error_L2

 # Create ODE problem with time span from 0.0 to 1.0
 ode = semidiscretize(semi, (0.0, final_time));

 # The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
 analysis_callback = AnalysisCallback(semi, interval=100, analysis_polydeg = 50)

 # The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
 stepsize_callback = StepsizeCallback(cfl=0.1)

 # Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
 callbacks = CallbackSet(analysis_callback, stepsize_callback)


 ###############################################################################
 # run the simulation

 # OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
 sol = solve(ode, Euler(), #, CarpenterKennedy2N54(williamson_condition=false),
             dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
             save_everystep=false, callback=callbacks);

 errors = analysis_callback(sol)

 errors_sol_L2[polydeg, 2] = errors.l2[1]
 errors_sol_inf[polydeg, 2] = errors.linf[1]
 
end
#plot(errors_normals_L2[2:end,1], yaxis=:log, ylabel = "discrete L2 norm", xlabel = "polynomial degree", label = "standard", linewidth=2, thickness_scaling = 1)
#plot!(errors_normals_L2[2:end,2], yaxis=:log, ylabel = "discrete L2 norm", xlabel = "polynomial degree", label = "mimetic", linewidth=2, thickness_scaling = 1)


plot(3:max_polydeg,errors_sol_L2[3:end,1], yaxis=:log, label = "standard", linewidth=2, thickness_scaling = 1)
plot!(3:max_polydeg,errors_sol_L2[3:end,2], yaxis=:log, label = "mimetic", linewidth=2, thickness_scaling = 1)


#= 
###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.1)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback() =#