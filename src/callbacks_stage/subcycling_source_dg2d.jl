# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    SubCyclingSource()

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct SubCyclingSource end

function (perform_subcycling!::SubCyclingSource)(u_ode, integrator, stage)

  perform_subcycling!(u_ode, integrator.p, integrator.t, integrator.dt)
end

function (perform_subcycling!::SubCyclingSource)(u_ode, semi, t, dt)

  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)

  perform_subcycling!(u, dt, mesh, equations, solver, cache)

  return nothing
end

function (perform_subcycling!::SubCyclingSource)(u, dt, mesh::AbstractMesh{2}, equations, dg, cache)
  max_iter = 1_000_000
  tol = 1.0e-10
  println("subcycling")

  @threaded for element in eachelement(dg, cache)
    
    for j in eachnode(dg), i in eachnode(dg)
      
      u_n = get_node_vars(u, equations, dg, i, j, element)
      x_local = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      
      for iter in 1:max_iter
        u_np1 = get_node_vars(u, equations, dg, i, j, element)
        du_local = source_terms_collision(u_np1, x_local, t, equations) 
        
        for v in eachvariable(equations)
          u[v, i, j, element] = u_n[v] + dt * du_local[v]
        end

        # Check if the subcycling converged
        println(string(iter), " ", string(maximum(abs.(u[:, i, j, element] - u_np1))))
        if maximum(abs.(u[:, i, j, element] - u_np1)) < tol
          break
        end
      end
    end
  end

  return nothing
end

init_callback(callback::SubCyclingSource, semi) = nothing

finalize_callback(perform_subcycling!::SubCyclingSource, semi) = nothing

end # @muladd
