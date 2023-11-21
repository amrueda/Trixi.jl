
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

equations = CompressibleEulerEquations3D(1.4)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)

# Initial condition for a Gaussian density profile with constant pressure
# and the velocity of a rotating solid body
function initial_condition_advection_sphere(x, t, equations::CompressibleEulerEquations3D)
    # Gaussian density
    rho = 1.0 + exp(-20 * (x[1]^2 + x[3]^2))
    # Constant pressure
    p = 1.0

    # Spherical coordinates for the point x
    if sign(x[2]) == 0.0
        signy = 1.0
    else
        signy = sign(x[2])
    end
    # Co-latitude
    colat = acos(x[3] / sqrt(x[1]^2 + x[2]^2 + x[3]^2))
    # Latitude (auxiliary variable)
    lat = -colat + 0.5 * pi
    # Longitude
    r_xy = sqrt(x[1]^2 + x[2]^2)
    if r_xy == 0.0
        phi = pi / 2
    else
        phi = signy * acos(x[1] / r_xy)
    end

    # Compute the velocity of a rotating solid body
    # (alpha is the angle between the rotation axis and the polar axis of the spherical coordinate system)
    v0 = 1.0 # Velocity at the "equator"
    alpha = 0.0 #pi / 4
    v_long = v0 * (cos(lat) * cos(alpha) + sin(lat) * cos(phi) * sin(alpha))
    v_lat = -v0 * sin(phi) * sin(alpha)

    # Transform to Cartesian coordinate system
    v1 = -cos(colat) * cos(phi) * v_lat - sin(phi) * v_long
    v2 = -cos(colat) * sin(phi) * v_lat + cos(phi) * v_long
    v3 = sin(colat) * v_lat

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

# Source term function to transform the Euler equations into the linear advection equations with variable advection velocity
function source_terms_convert_to_linear_advection(u, du, x, t,
                                                  equations::CompressibleEulerEquations3D)
    v1 = u[2] / u[1]
    v2 = u[3] / u[1]
    v3 = u[4] / u[1]

    s2 = du[1] * v1 - du[2]
    s3 = du[1] * v2 - du[3]
    s4 = du[1] * v3 - du[4]
    s5 = 0.5 * (s2 * v1 + s3 * v2 + s4 * v3) - du[5]

    return SVector(0.0, s2, s3, s4, s5)
end

initial_condition = initial_condition_advection_sphere

mesh = Trixi.P4estMeshCubedSphere2D(5, 1.0, polydeg = polydeg, initial_refinement_level = 0)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convert_to_linear_advection)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to π
tspan = (0.0, pi)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 10,
                                     save_analysis = true,
                                     extra_analysis_integrals = (Trixi.density,))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 10,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.7)

# AMR callback
amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.02,
                                      max_level = 2, max_threshold = 0.03)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback,
                        stepsize_callback, amr_callback, save_solution)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
