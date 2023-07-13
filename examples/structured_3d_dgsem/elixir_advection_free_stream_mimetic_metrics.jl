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

theta_der1(xi, eta, zeta) = -(0.1 * pi) * sin(pi * xi) * cos(pi * eta) * cos(pi * zeta) 
theta_der2(xi, eta, zeta) = -(0.1 * pi) * cos(pi * xi) * sin(pi * eta) * cos(pi * zeta) 
theta_der3(xi, eta, zeta) = -(0.1 * pi) * cos(pi * xi) * cos(pi * eta) * sin(pi * zeta) 

function exact_contravariant_vectors!(Ja, xi, eta, zeta)
    theta_xi = theta_der1(xi, eta, zeta)
    theta_eta = theta_der2(xi, eta, zeta)
    theta_zeta = theta_der3(xi, eta, zeta)
    Ja[1,1] = 1 +  theta_eta + theta_zeta
    Ja[1,2] = -theta_xi
    Ja[1,3] = -theta_xi
    Ja[2,1] = -theta_eta
    Ja[2,2] = 1 +  theta_xi + theta_zeta
    Ja[2,3] = -theta_eta
    Ja[3,1] = -theta_zeta
    Ja[3,2] = -theta_zeta
    Ja[3,3] = 1 +  theta_xi + theta_eta
end

function compute_error(solver, semi)
  @unpack nodes, weights = solver.basis
  exact_Ja = zero(MMatrix{3, 3, Float64})
  error = zero(Float64)
  error_L2 = zero(Float64)
  for k in eachnode(solver.basis)
    for j in eachnode(solver.basis)
      for i in eachnode(solver.basis)
        exact_contravariant_vectors!(exact_Ja, nodes[i], nodes[j], nodes[k])
        error = max(error, maximum(abs.(semi.cache.elements.contravariant_vectors[:,:,i,j,k,1] - exact_Ja)))
        error_L2 += norm(semi.cache.elements.contravariant_vectors[:,1,i,j,k,1] - exact_Ja[:,1]) * weights[i] * weights[j] * weights[k]
      end
    end
  end
  return error, error_L2 / 8
end

cells_per_dimension = (1,1,1)

errors_inf = zeros(20,2)
errors_L2 = zeros(20,2)
for polydeg in 1:20
  println("Computing polydeg = ", polydeg)
  # Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
  solver = DGSEM(polydeg, flux_lax_friedrichs)

  # Create curved mesh with 8 x 8 x 8 elements
  mesh = StructuredMesh(cells_per_dimension, mapping; mimetic = false)

  # A semidiscretization collects data structures and functions for the spatial discretization
  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_constant, solver)

  error_inf, error_L2 = compute_error(solver, semi)
  errors_inf[polydeg,1] = error_inf
  errors_L2[polydeg,1] = error_L2

  # Create curved mesh with 8 x 8 x 8 elements
  mesh = StructuredMesh(cells_per_dimension, mapping; mimetic = true)

  # A semidiscretization collects data structures and functions for the spatial discretization
  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_constant, solver)

  error_inf, error_L2 = compute_error(solver, semi)
  errors_inf[polydeg,2] = error_inf
  errors_L2[polydeg,2] = error_L2

end
plot(errors_L2[:,1], yaxis=:log, title = "standard")
plot!(errors_L2[:,2], yaxis=:log, title = "mimetic")

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