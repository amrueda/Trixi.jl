
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.001 # almost isothermal when gamma reaches 1
equations = CompressibleEulerEquations2D(gamma)

# This is a hand made colliding flow setup without reference. Features Mach=70 inflow from both
# sides, with relative low temperature, such that pressure keeps relatively small
# Computed with gamma close to 1, to simulate isothermal gas
function initial_condition_colliding_flow_astro(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # resolution 128^2 elements (refined close to the interface) and polydeg=3 (total of 512^2 DOF)
  # domain size is [-64,+64]^2
  @unpack gamma = equations
  # the quantities are chosen such, that they are as close as possible to the astro examples
  # keep in mind, that in the astro example, the physical units are weird (parsec, mega years, ...)
  rho = 0.0247
  c = 0.2
  p = c^2 / gamma * rho
  vel = 13.907432274789372
  slope = 1.0
  v1 = -vel*tanh(slope * x[1])
  # add small initial disturbance to the field, but only close to the interface
  if abs(x[1]) < 10
    v1 = v1 * (1 + 0.01 * sin(pi * x[2]))
  end
  v2 = 0.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_colliding_flow_astro


boundary_conditions = (
                       x_neg=BoundaryConditionDirichlet(initial_condition_colliding_flow_astro),
                       x_pos=BoundaryConditionDirichlet(initial_condition_colliding_flow_astro),
                       y_neg=boundary_condition_periodic,
                       y_pos=boundary_condition_periodic,
                      )



surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# shock capturing necessary for this tough example
indicator_sc = IndicatorIDP(equations, basis;
                            IDPDensityTVD=true,
                            IDPPressureTVD=false,
                            IDPPositivity=true)
volume_integral=VolumeIntegralShockCapturingSubcell(indicator_sc; volume_flux_dg=volume_flux,
                                                                  volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-64.0, -64.0)
coordinates_max = ( 64.0,  64.0)

# only refinment in a patch. Needs x=-17/+17 to trigger refinment due to coarse base mesh
refinement_patches = (
  (type="box", coordinates_min=(-17, -64), coordinates_max=(17, 64)),
  (type="box", coordinates_min=(-17, -64), coordinates_max=(17, 64)),
  (type="box", coordinates_min=(-17, -64), coordinates_max=(17, 64)),
  (type="box", coordinates_min=(-17, -64), coordinates_max=(17, 64)),
  #(type="box", coordinates_min=(-17, -64), coordinates_max=(17, 64)), # very high resolution, takes about 1000s on 2 cores
)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                refinement_patches=refinement_patches,
                periodicity=(false,true),
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 25.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.4)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback,
                        save_solution)

###############################################################################
# run the simulation
sol = Trixi.solve_IDP(ode, semi,
                      dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                      callback=callbacks);
summary_callback() # print the timer summary
