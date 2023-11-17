
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

equations = CompressibleEulerEquations3D(1.4)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
polydeg = 3
solver = DGSEM(polydeg=polydeg, surface_flux=flux_lax_friedrichs)

function initial_condition_advection_sphere(x, t, equations::CompressibleEulerEquations3D)
    # Constant velocity and pressure
    rho = 1.0
    #if x[2] < 0
        rho = 1.0 + exp(-20*(x[1]^2 + x[3]^2))
    #end

    p = 1.0

    colat = acos(x[3] / sqrt(x[1]^2 + x[2]^2 + x[3]^2))
    lat = -colat + 0.5 * pi # Latitude, not co-latitude!
    phi = sign(x[2]) * acos(x[1] / sqrt(x[1]^2 + x[2]^2)) 

    v0 = 1.0
    alpha = 0.0
    v_long = v0 * (cos(lat) * cos(alpha) + sin(lat) * cos(phi) * sin(alpha))
    v_lat = -v0 * sin(phi) * sin(alpha)

    v1 = cos(colat) * cos(phi) * v_lat - sin(phi) * v_long
    v2 = cos(colat) * sin(phi) * v_lat + cos(phi) * v_long
    v3 = - sin(colat) * v_lat

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

initial_condition = initial_condition_advection_sphere

mesh = Trixi.P4estMeshCubedSphere2D(5, 1.0, polydeg=polydeg, initial_refinement_level=0)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

# compute area of the sphere to test
area = zero(Float64)
for element in 1:Trixi.ncells(mesh)
    for j in 1:polydeg+1, i in 1:polydeg+1
        global area += solver.basis.weights[i] * solver.basis.weights[j] / semi.cache.elements.inverse_jacobian[i, j, element]
    end
end
println("Area of sphere: ", area)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to π
tspan = (0.0, pi)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=10)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=10,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.8)

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