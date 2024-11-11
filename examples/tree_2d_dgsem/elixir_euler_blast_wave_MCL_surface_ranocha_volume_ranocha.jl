
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A weak blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.3416
    v1 = r > 0.5 ? 0.0 : 0.3615 * cos_phi
    v2 = r > 0.5 ? 0.0 : 0.3615 * sin_phi
    p = r > 0.5 ? 1.0 : 1.5133

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_blast_wave

# Numerical fluces
# This combination is EC for the high-order solver, but entropy_limiter_semidiscrete with 
# volume_flux_fv = flux_lax_friedrichs adds extra dissipation because it enforces a low-order Tadmor condition 
surface_flux = flux_ranocha
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)
limiter_mcl = SubcellLimiterMCL(equations, basis;
                                density_limiter = false,
                                density_coefficient_for_all = false,
                                sequential_limiter = false,
                                conservative_limiter = false,
                                positivity_limiter_density = false,
                                positivity_limiter_pressure = false,
                                positivity_limiter_pressure_exact = false,
                                entropy_limiter_semidiscrete = true,
                                smoothness_indicator = false,
                                Plotting = true)
volume_integral = VolumeIntegralSubcellLimiting(limiter_mcl;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = flux_lax_friedrichs)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 1,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     output_directory = "out",
                                     analysis_filename = "analysis.dat",
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.9)

limiting_analysis_callback = LimitingAnalysisCallback(output_directory = "out",
                                                      interval = 1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        limiting_analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation
stage_callbacks = (BoundsCheckCallback(save_errors = false),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
