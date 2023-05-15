
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
equations = IdealMhdMultiIonEquations2D(gammas = (5/3, 5/3),
                                        charge_to_mass = (1.0, 1.0))

"""
    initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)
Modified to fit the multi-ion framework (two ions), as proposed by
- Rueda-Ramírez et al. (2023)
The classical Orszag-Tang vortex test case. Here, the setup is taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_orszag_tang(x, t, equations::IdealMhdMultiIonEquations2D)
  # setup taken from Derigs et al. DMV article (2018)
  # domain must be [0, 1] x [0, 1], γ_1 = γ_2 = 5/3
  
  v1 = -sin(2.0*pi*x[2])
  v2 =  sin(2.0*pi*x[1])
  v3 = 0.0
  T = 1.0 / equations.gammas[1]
  B1 = -sin(2.0*pi*x[2]) / equations.gammas[1]
  B2 =  sin(4.0*pi*x[1]) / equations.gammas[1]
  B3 = 0.0
  
  rho1 = zero(x[1])
  if x[1] > 0.75
    rho1 = 0.49 * (tanh(50 * (x[1] - 1.0)) + 1) + 0.02
  elseif x[1] > 0.25
    rho1 = 0.49 * (-tanh(50 * (x[1] - 0.5)) + 1) + 0.02
  else
    rho1 = 0.49 * (tanh(50 * (x[1])) + 1) + 0.02
  end

  if x[1] < 0.25
    rho2 = 0.49 * (-tanh(50 * (x[1])) + 1) + 0.02
  elseif x[1] < 0.75
    rho2 = 0.49 * (tanh(50 * (x[1] - 0.5)) + 1) + 0.02
  else
    rho2 = 0.49 * (-tanh(50 * (x[1] - 1.0)) + 1) + 0.02
  end

  p1 = rho1 * T
  p2 = rho2 * T

  return prim2cons(SVector(B1, B2, B3, rho1, v1, v2, v3, p1, rho2, v1, v2, v3, p2), equations)

end
initial_condition = initial_condition_orszag_tang

volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)

basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_standard)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

amr_indicator = IndicatorLöhner(semi,
                                variable=density)

amr_controller = ControllerThreeLevelCombined(semi, amr_indicator,indicator_sc,
                                      base_level=4,
                                      med_level =5, med_threshold=0.02, # med_level = current level
                                      max_level =6, max_threshold=0.04,
                                      max_threshold_secondary=0.2)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=6,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

save_restart = SaveRestartCallback(interval=1000,
                           save_final_restart=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
			                  amr_callback,
                        save_solution,
                        save_restart,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
