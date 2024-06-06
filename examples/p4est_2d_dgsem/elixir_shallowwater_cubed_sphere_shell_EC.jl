
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

equations = ShallowWaterEquations3D(gravity_constant = 9.81)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
polydeg = 3
volume_flux = flux_wintermeyer_etal # flux_fjordholm_etal #flux_wintermeyer_etal
surface_flux = flux_wintermeyer_etal # flux_fjordholm_etal #flux_lax_friedrichs # flux_fjordholm_etal #flux_wintermeyer_etal
solver = DGSEM(polydeg = polydeg, 
               surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# Initial condition for a Gaussian density profile with constant pressure
# and the velocity of a rotating solid body
function initial_condition_advection_sphere(x, t, equations::ShallowWaterEquations3D)
    # Gaussian density
    rho = 2.0 #+ exp(-20 * (x[1]^2 + x[3]^2))

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

    return prim2cons(SVector(rho, v1, v2, v3, 0), equations)
end

# Source term function to apply Lagrange multiplier
function source_terms_lagrange_multiplier(u, du, x, t,
                                                  equations::ShallowWaterEquations3D)
    v1 = u[2] / u[1]
    v2 = u[3] / u[1]
    v3 = u[4] / u[1]

    x_dot_div_f = (x[1] * du[2] + x[2] * du[3] + x[3] * du[4]) / sum(x.^2)
    
    s2 = -x[1] * x_dot_div_f
    s3 = -x[2] * x_dot_div_f
    s4 = -x[3] * x_dot_div_f
    #println(s2, " ", s3, " ", s4)
    #= s2 = s3 = s4 = 0 =#

    return SVector(0.0, s2, s3, s4, 0.0)
end

# Custom RHS that applies a source term that depends on du (used to convert apply Lagrange multiplier)
function rhs_semi_custom!(du_ode, u_ode, semi, t)
    # Compute standard Trixi RHS
    Trixi.rhs!(du_ode, u_ode, semi, t)

    # Now apply the custom source term
    Trixi.@trixi_timeit Trixi.timer() "custom source term" begin
        @unpack solver, equations, cache = semi
        @unpack node_coordinates = cache.elements

        # Wrap the solution and RHS
        u = Trixi.wrap_array(u_ode, semi)
        du = Trixi.wrap_array(du_ode, semi)

        # Compute the dsdt without the source terms correction
        dsdu_ut = 0
        dsdu_ut_after = 0
        total_volume = 0
        @unpack weights = semi.solver.basis

        #Trixi.@threaded 
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                u_local = Trixi.get_node_vars(u, equations, solver, i, j, element)
                du_local = Trixi.get_node_vars(du, equations, solver, i, j, element)
                x_local = Trixi.get_node_coords(node_coordinates, equations, solver,
                                                i, j, element)

                volume_jacobian = abs(inv(semi.cache.elements.inverse_jacobian[i, j, element]))
                dsdu_ut += sum(cons2entropy(u_local, equations)[1:end-1] .* du_local[1:end-1]) * volume_jacobian * weights[i] * weights[j]
                total_volume += volume_jacobian * weights[i] * weights[j]

                source = source_terms_lagrange_multiplier(u_local, du_local,
                                                                  x_local, t, equations)

                du_local = Trixi.get_node_vars(du, equations, solver, i, j, element)
                dsdu_ut_after += Trixi.dot(cons2entropy(u_local, equations), source) * volume_jacobian * weights[i] * weights[j]
                Trixi.add_to_node_vars!(du, source, equations, solver, i, j, element)
            end
        end
        #println((dsdu_ut) / total_volume, "  ", (dsdu_ut_after) / total_volume)
    end
end

initial_condition = initial_condition_advection_sphere

mesh = Trixi.P4estMeshCubedSphere2D(5, 1.0, polydeg = polydeg, initial_refinement_level = 0)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to π
tspan = (0.0, pi)
ode = semidiscretize(semi, tspan)

#for cfl in [0.800, 0.400, 0.200, 0.100, 0.050, 0.025]
cfl = 0.7

# Create custom discretization that runs with the custom RHS
ode_semi_custom = ODEProblem(rhs_semi_custom!,
                            ode.u0,
                            ode.tspan,
                            semi)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 10,
                                    save_analysis = true,
                                    #analysis_filename = "analysis_cfl"*string(cfl)*".dat",
                                    extra_analysis_errors = (:conservation_error, ),
                                    extra_analysis_integrals = (Trixi.waterheight, Trixi.energy_total))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 10,
                                    solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = cfl)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode_semi_custom, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
#end
