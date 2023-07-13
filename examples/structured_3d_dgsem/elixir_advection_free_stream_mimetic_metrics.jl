
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(20, flux_lax_friedrichs)

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

  return SVector(x, y, z)
end

cells_per_dimension = (1, 1, 1)

# Create curved mesh with 8 x 8 x 8 elements
mesh = StructuredMesh(cells_per_dimension, mapping)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_constant, solver)


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
stepsize_callback = StepsizeCallback(cfl=2.0)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
