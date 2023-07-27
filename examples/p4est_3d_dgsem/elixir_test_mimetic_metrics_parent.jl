#using LinearAlgebra
using OrdinaryDiffEq
using Trixi
#using StaticArrays
#using Plots

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

# Mapping as described in https://arxiv.org/abs/2012.12040
function mapping(xi, eta, zeta)
  # Transform input variables between -1 and 1 onto our crazy domain

  y = eta + 0.1 * (cos(pi * xi) *
                   cos(pi * eta) *
                   cos(pi * zeta))

  x = xi + 0.1 * (cos(pi * xi) *
                   cos(pi * y) *
                   cos(pi * zeta))

  z = zeta + 0.1 * (cos(pi * x) *
                   cos(pi * y) *
                   cos(pi * zeta))

  #= x = xi
  y = eta
  z = zeta =#
  return SVector(x, y, z)
end

cells_per_dimension = (1, 1, 1) # The p4est implementation works with one tree per direction only


exact_jacobian = false
mimetic = false
final_time = 1e-3

initial_condition = initial_condition_constant

max_polydeg = 1
n_polydeg_geo = 4
errors_sol_inf = zeros(max_polydeg,n_polydeg_geo)
errors_sol_L2 = zeros(max_polydeg,n_polydeg_geo)
polydeg_geos = [2, 3, 5, 10]
#polydeg_geos = [5]

polydeg = 5

# Refine bottom left quadrant of each tree to level 3
function refine_fn(p8est, which_tree, quadrant)
    quadrant_obj = unsafe_load(quadrant)
    if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.z == 0 && quadrant_obj.level < 2
        # return true (refine)
        return Cint(1)
    else
        # return false (don't refine)
        return Cint(0)
    end
end

for i in 1:n_polydeg_geo
    for polydeg in 1:max_polydeg
        polydeg_geo = polydeg_geos[i]
        
        solver = DGSEM(polydeg, flux_lax_friedrichs)

        # Create curved mesh with 8 x 8 x 8 elements
        boundary_condition = BoundaryConditionDirichlet(initial_condition)
        boundary_conditions = Dict(
            :x_neg => boundary_condition,
            :x_pos => boundary_condition,
            :y_neg => boundary_condition,
            :y_pos => boundary_condition,
            :z_neg => boundary_condition,
            :z_pos => boundary_condition
        )
        println("polydeg_geo: ", polydeg_geo)
        #mesh = P4estMesh(cells_per_dimension; polydeg = polydeg_geo, mapping = mapping, mimetic = mimetic, exact_jacobian = exact_jacobian, initial_refinement_level = 0, periodicity = true)
        mesh = P4estMesh(cells_per_dimension; polydeg = polydeg_geo, mapping = mapping, mimetic = mimetic, exact_jacobian = exact_jacobian, initial_refinement_level = 0, periodicity = false, polydeg_parent_metrics = polydeg_geo)
            
        # Refine recursively until each bottom left quadrant of a tree has level 3
        # The mesh will be rebalanced before the simulation starts
        refine_fn_c = @cfunction(refine_fn, Cint, (Ptr{Trixi.p8est_t}, Ptr{Trixi.p4est_topidx_t}, Ptr{Trixi.p8est_quadrant_t}))
        Trixi.refine_p4est!(mesh.p4est, true, refine_fn_c, C_NULL)

        # A semidiscre  tization collects data structures and functions for the spatial discretization
        #semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions = boundary_conditions)

        # Create ODE problem with time span from 0.0 to 1.0
        ode = semidiscretize(semi, (0.0, final_time));

        summary_callback = SummaryCallback()

        # The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
        analysis_callback = AnalysisCallback(semi, interval=100)

        # The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
        stepsize_callback = StepsizeCallback(cfl=0.1)

        # The SaveSolutionCallback allows to save the solution to a file in regular intervals
        save_solution = SaveSolutionCallback(interval=100,
                                            solution_variables=cons2prim)

        #= amr_indicator = IndicatorHennemannGassner(semi,
                                                    alpha_max=1.0,
                                                    alpha_min=0.0001,
                                                    alpha_smooth=false,
                                                    variable=Trixi.energy_total)

        amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                                base_level=4,
                                                max_level=6, max_threshold=0.01)

        #= amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                                base_level=4,
                                                med_level=5, med_threshold=0.1,
                                                max_level=6, max_threshold=0.6) =#
        amr_callback = AMRCallback(semi, amr_controller,
                                    interval=5,
                                    adapt_initial_condition=true,
                                    adapt_initial_condition_only_refine=true) =#

        # Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
        callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback, save_solution)


        ###############################################################################
        # run the simulation

        # OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
        sol = solve(ode, Euler(), #CarpenterKennedy2N54(williamson_condition=false),
                    dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                    save_everystep=false, callback=callbacks);


        summary_callback()
        errors = analysis_callback(sol)

        errors_sol_L2[polydeg, i] = errors.l2[1]
        errors_sol_inf[polydeg, i] = errors.linf[1]
    end
end
#=end

for i in 1:n_polydeg_geo
  plot!(errors_sol_inf[:,i], xaxis=:log, yaxis=:log, label = "polydeg_geo="*string(polydeg_geos[i]), linewidth=2, thickness_scaling = 1)
end
plot!(title = "mimetic="*string(mimetic)*", exact_jacobian="*string(exact_jacobian))
plot!(xlabel = "polydeg", ylabel = "|u_ex - u_disc|_inf")

plot!(ylims=(1e-15,1e-1))

plot!(xticks=([2, 4, 8, 16], ["2", "4", "8", "16"])) =#