using OrdinaryDiffEq
using Trixi

###############################################################################
# This elixir describes the counter-streaming of two plasma beams, as relevant in inertial confinement fusion devices. It is described in:
# - Ghosh, D., Chapman, T. D., Berger, R. L., Dimits, A., & Banks, J. W. (2019). A 
#   multispecies, multifluid model for laser–induced counterstreaming plasma simulations. 
#   Computers & Fluids, 186, 38-57. [DOI: 10.1016/j.compfluid.2019.04.012](https://doi.org/10.1016/j.compfluid.2019.04.012).
#
# This is effectively a one-dimensional case.
#

# Return the electron pressure for a constant electron temperature Te = 1 keV
function electron_pressure_constantTe(u, equations::IdealGlmMhdMultiIonEquations3D)
    @unpack charge_to_mass = equations
    Te = 0.5
    total_electron_charge = zero(eltype(u))
    for k in eachcomponent(equations)
        rho_k = u[3 + (k - 1) * 5 + 1]
        total_electron_charge += rho_k * charge_to_mass[k]
    end

    # Boltzmann constant divided by elementary charge
    kB_e = 2.39629174724586E-03

    return total_electron_charge * kB_e * Te
end

function electron_temperature_constantTe(u, equations::IdealGlmMhdMultiIonEquations3D)
    return 0.5
end

# semidiscretization of the ideal MHD equations
equations = IdealGlmMhdMultiIonEquations3D(gammas = (5 / 3, 5 / 3),
                                           charge_to_mass = (417.3114568162800000,
                                                             417.3114568162800000),
                                           gas_constants = (1.0, 1.0),
                                           molar_masses = (1.0, 1.0),
                                           ion_ion_collision_constants = [0.0 0.5*440.1240021904530000;
                                                                          0.5*440.1240021904530000 0.0], #660.1860032856790000, #0.2855567736309880 ) #0.20396912402213403) #0.6119073720664020) #  this is with ln(Lambda)=15
                                           ion_electron_collision_constants = (3.4993604479301E-02,
                                                                               3.4993604479301E-02),
                                           electron_pressure = electron_pressure_constantTe,
                                           electron_temperature = electron_temperature_constantTe)

"""
    initial_condition_thermeq1(x, t, equations::IdealGlmMhdEquations2D)
"""
function initial_condition_interpenetration(x, t, equations::IdealGlmMhdMultiIonEquations3D)
    # setup taken from Derigs et al. DMV article (2018)
    # domain must be [0, 1] x [0, 1], γ_1 = γ_2 = 5/3
    dx = 0.005

    v1 = v2 = v3 = 0.0
    B1 = B2 = B3 = 0.0

    rho1 = 0.5 * smoothed_slab(x[1], dx, -1.0, 0.2)
    rho2 = 0.5 * smoothed_slab(x[1], dx, 0.8, 2.0)

    rho_vac = 1e-14
    rho1 += rho_vac
    rho2 += rho_vac

    p1 = rho1
    p2 = rho2

    return prim2cons(SVector(B1, B2, B3, rho1, v1, v2, v3, p1, rho2, v1, v2, v3, p2, 0.0),
                     equations)
end
function smoothed_slab(x, dx, xmin, xmax)
    exp(x / dx) *
    (1.0 / (exp(xmin / dx) + exp(x / dx)) - 1.0 / (exp(xmax / dx) + exp(x / dx)))
end

"""
Free-slip reflective wall for species 1 only  
"""
function boundary_condition_lleft(u_inner, normal_direction::AbstractVector,
                                  x, t,
                                  surface_flux_function,
                                  equations::IdealGlmMhdMultiIonEquations3D)
    u_outer = SVector(u_inner[1], u_inner[2], u_inner[3], u_inner[4], -u_inner[5],
                      u_inner[6], u_inner[7], u_inner[8], u_inner[9], u_inner[10],
                      u_inner[11], u_inner[12], u_inner[13], u_inner[14])

    flux_conservative, flux_noncons = surface_flux_function

    return flux_conservative(u_inner, u_outer, normal_direction, equations),
           flux_noncons(u_inner, u_outer, normal_direction, equations)
end

function Trixi.get_boundary_outer_state(u_inner, t,
                                        boundary_condition::typeof(boundary_condition_lleft),
                                        normal_direction::AbstractVector,
                                        mesh::P4estMesh, equations, dg, cache,
                                        indices...)
    return SVector(u_inner[1], u_inner[2], u_inner[3], u_inner[4], -u_inner[5], u_inner[6],
                   u_inner[7], u_inner[8], u_inner[9], u_inner[10], u_inner[11],
                   u_inner[12], u_inner[13], u_inner[14])
end

"""
Free-slip reflective wall for species 2 only  
"""
function boundary_condition_rright(u_inner, normal_direction::AbstractVector,
                                   x, t,
                                   surface_flux_function,
                                   equations::IdealGlmMhdMultiIonEquations3D)
    u_outer = SVector(u_inner[1], u_inner[2], u_inner[3], u_inner[4], u_inner[5],
                      u_inner[6], u_inner[7], u_inner[8], u_inner[9], -u_inner[10],
                      u_inner[11], u_inner[12], u_inner[13], u_inner[14])

    # return flux
    flux_conservative, flux_noncons = surface_flux_function

    return flux_conservative(u_inner, u_outer, normal_direction, equations),
           flux_noncons(u_inner, u_outer, normal_direction, equations)
end

function Trixi.get_boundary_outer_state(u_inner, t,
                                        boundary_condition::typeof(boundary_condition_rright),
                                        normal_direction::AbstractVector,
                                        mesh::P4estMesh, equations, dg, cache,
                                        indices...)
    return SVector(u_inner[1], u_inner[2], u_inner[3], u_inner[4], u_inner[5], u_inner[6],
                   u_inner[7], u_inner[8], u_inner[9], -u_inner[10], u_inner[11],
                   u_inner[12], u_inner[13], u_inner[14])
end

boundary_conditions = Dict(:x_neg => boundary_condition_lleft,
                           :x_pos => boundary_condition_rright)

function temperature1(cons, equations::IdealGlmMhdMultiIonEquations3D)
    prim = cons2prim(cons, equations)
    rho, _, _, _, p = Trixi.get_component(1, prim, equations)

    return p / rho / equations.gas_constants[1]
end
function temperature2(cons, equations::IdealGlmMhdMultiIonEquations3D)
    prim = cons2prim(cons, equations)
    rho, _, _, _, p = Trixi.get_component(2, prim, equations)

    return p / rho / equations.gas_constants[2]
end
@inline function vel11(cons, equations::IdealGlmMhdMultiIonEquations3D)
    prim = cons2prim(cons, equations)
    _, v1, _, _, _ = Trixi.get_component(1, prim, equations)

    return v1
end
@inline function vel21(cons, equations::IdealGlmMhdMultiIonEquations3D)
    prim = cons2prim(cons, equations)
    _, v1, _, _, _ = Trixi.get_component(2, prim, equations)

    return v1
end

@inline function pressure1(u, equations::IdealGlmMhdMultiIonEquations3D)
    pres = pressure(u, equations)
    return pres[1]
end

@inline function pressure2(u, equations::IdealGlmMhdMultiIonEquations3D)
    pres = pressure(u, equations)
    return pres[2]
end

@inline function Trixi.gradient_conservative(::typeof(pressure1),
                                             u, equations::IdealGlmMhdMultiIonEquations3D)
    return gradient_conservative_pressure(1, u, equations)
end

@inline function Trixi.gradient_conservative(::typeof(pressure2),
                                             u, equations::IdealGlmMhdMultiIonEquations3D)
    return gradient_conservative_pressure(2, u, equations)
end

# Transformation from conservative variables u to d(p)/d(u)
function gradient_conservative_pressure(k::Integer, u,
                                        equations::IdealGlmMhdMultiIonEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = Trixi.get_component(k, u, equations)
    B1, B2, B3 = magnetic_field(u, equations)
    psi = divergence_cleaning_field(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_square = v1^2 + v2^2 + v3^2

    dp_du = zero(Trixi.MVector{nvariables(equations), eltype(u)})

    Trixi.set_component!(dp_du, k, 0.5f0 * v_square, -v1, -v2, -v3, 1, equations)
    dp_du[1] = -B1
    dp_du[2] = -B2
    dp_du[3] = -B3
    dp_du[end] = -psi

    for i in Trixi.eachvariable(equations)
        dp_du[i] *= (equations.gammas[k] - 1.0)
    end

    return SVector(dp_du)
end

initial_condition = initial_condition_interpenetration

volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)

basis = LobattoLegendreBasis(3)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho_1", "rho_2"],
                                positivity_variables_nonlinear = (pressure1, pressure2),
                                local_twosided_variables_cons = [], #["rho_1", "rho_2"] 
                                local_onesided_variables_nonlinear = [],
                                max_iterations_newton = 40, # Default parameters are not sufficient to fulfill bounds properly.
                                newton_tolerances = (1.0e-14, 1.0e-15))

volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)

# volume_integral = VolumeIntegralPureLGLFiniteVolume(surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

### MESH ###

coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (1.0, 1.0, 1.0)

trees_per_dimension = (128, 1, 1)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 1,
                 coordinates_min = coordinates_min,
                 coordinates_max = coordinates_max,
                 initial_refinement_level = 0,
                 periodicity = (false, true, true))

# io-ion and ion-electron source terms
# We don't need Lorentz terms because there's no electric field
function source_terms_counterstreaming(u, x, t, equations::IdealGlmMhdMultiIonEquations3D)
    Sii = source_terms_collision_ion_ion(u, x, t, equations::IdealGlmMhdMultiIonEquations3D)
    Sie = source_terms_collision_ion_electron(u, x, t,
                                              equations::IdealGlmMhdMultiIonEquations3D)
    return Sii + Sie
end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_counterstreaming,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 50
analysis_callback = AnalysisCallback(semi,
                                     save_analysis = true,
                                     interval = analysis_interval,
                                     extra_analysis_integrals = (temperature1, temperature2,
                                                                 vel11, vel21),
                                     output_directory = joinpath(@__DIR__, "out"))
alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.01, # interval = 50, # 
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = joinpath(@__DIR__, "out")
                                     #  extra_node_variables = (:limiting_coefficient,)
                                     )

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true,
                                   output_directory = joinpath(@__DIR__, "out"))

glm_speed_callback = GlmSpeedCallback(glm_scale = 1.0, cfl = cfl)
callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        save_restart,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################

# stage_callbacks = ()
stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);

#= sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false); #SSPRK43(stage_limiter!); # stage_limiter!, 
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks); =#
