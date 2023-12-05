
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.0, 0.0, 0.0)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

########
# For diffusion
diffusivity() = 5.0e-2
equations_parabolic = LaplaceDiffusion3D(diffusivity(), equations)
########

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
polydeg = 4
solver = DGSEM(polydeg=polydeg, surface_flux=flux_lax_friedrichs)

initial_condition = initial_condition_convergence_test # A sinusoidal wave gets diffused
#initial_condition = initial_condition_constant # A constant solution stays constant

mesh = Trixi.P4estMeshCubedSphere2D(5, 1.0, polydeg=polydeg, initial_refinement_level=0)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver)

# compute area of the sphere
area = zero(Float64)
for element in 1:Trixi.ncells(mesh)
    for j in 1:polydeg+1, i in 1:polydeg+1
        global area += solver.basis.weights[i] * solver.basis.weights[j] / semi.cache.elements.inverse_jacobian[i, j, element]
    end
end
println("Area of sphere: ", area)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 10,
                                     save_analysis = true,
                                     extra_analysis_integrals = (Trixi.energy_total,))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=1,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
#stepsize_callback = StepsizeCallback(cfl=0.001)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution) #, stepsize_callback


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
#= sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0e-3, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks); =#

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
alg = RDPK3SpFSAL49()
time_int_tol = 1.0e-11
sol = solve(ode, alg; abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)

# Print the timer summary
summary_callback()