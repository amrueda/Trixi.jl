# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    AntidiffusiveStage()

Perform antidiffusive stage for IDP limiting.
"""
struct AntidiffusiveStage end

function (antidiffusive_stage!::AntidiffusiveStage)(u_ode, integrator, stage)

  antidiffusive_stage!(u_ode, integrator.p, integrator.t, integrator.dt, integrator.p.solver.volume_integral)
end

(::AntidiffusiveStage)(u_ode, semi, t, dt, volume_integral::AbstractVolumeIntegral) = nothing

function (antidiffusive_stage!::AntidiffusiveStage)(u_ode, semi, t, dt, volume_integral::VolumeIntegralShockCapturingSubcell)

  @trixi_timeit timer() "antidiffusive_stage!" antidiffusive_stage!(u_ode, semi, t, dt, volume_integral.indicator)
end

(::AntidiffusiveStage)(u_ode, semi, t, dt, indicator::IndicatorMCL) = nothing

function (antidiffusive_stage!::AntidiffusiveStage)(u_ode, semi, t, dt, indicator::IndicatorIDP)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)

  @trixi_timeit timer() "alpha calculation" semi.solver.volume_integral.indicator(u, semi, solver, t, dt)

  perform_idp_correction!(u, dt, mesh, equations, solver, cache)

  return nothing
end

init_callback(callback::AntidiffusiveStage, semi) = nothing

finalize_callback(antidiffusive_stage!::AntidiffusiveStage, semi) = nothing

include("antidiffusive_stage_2d.jl")

end # @muladd
