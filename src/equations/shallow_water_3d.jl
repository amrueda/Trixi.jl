# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    ShallowWaterEquations3D(; gravity, H0 = 0, threshold_limiter = nothing, threshold_wet = nothing)

Shallow water equations (SWE) in three space dimensions. The equations are given by
```math
\begin{aligned}
  \frac{\partial h}{\partial t} + \frac{\partial}{\partial x}(h v_1)
    + \frac{\partial}{\partial y}(h v_2) &= 0 \\
    \frac{\partial}{\partial t}(h v_1) + \frac{\partial}{\partial x}\left(h v_1^2 + \frac{g}{2}h^2\right)
    + \frac{\partial}{\partial y}(h v_1 v_2) + g h \frac{\partial b}{\partial x} &= 0 \\
    \frac{\partial}{\partial t}(h v_2) + \frac{\partial}{\partial x}(h v_1 v_2)
    + \frac{\partial}{\partial y}\left(h v_2^2 + \frac{g}{2}h^2\right) + g h \frac{\partial b}{\partial y} &= 0.
\end{aligned}
```
The unknown quantities of the SWE are the water height ``h`` and the velocities ``\mathbf{v} = (v_1, v_2)^T``.
The gravitational constant is denoted by `g` and the (possibly) variable bottom topography function ``b(x,y)``.
Conservative variable water height ``h`` is measured from the bottom topography ``b``, therefore one
also defines the total water height as ``H = h + b``.

The additional quantity ``H_0`` is also available to store a reference value for the total water height that
is useful to set initial conditions or test the "lake-at-rest" well-balancedness.

Also, there are two thresholds which prevent numerical problems as well as instabilities. Both of them do not
have to be passed, as default values are defined within the struct. The first one, `threshold_limiter`, is
used in [`PositivityPreservingLimiterShallowWater`](@ref) on the water height, as a (small) shift on the initial
condition and cutoff before the next time step. The second one, `threshold_wet`, is applied on the water height to
define when the flow is "wet" before calculating the numerical flux.

The bottom topography function ``b(x,y)`` is set inside the initial condition routine
for a particular problem setup. To test the conservative form of the SWE one can set the bottom topography
variable `b` to zero.

In addition to the unknowns, Trixi.jl currently stores the bottom topography values at the approximation points
despite being fixed in time. This is done for convenience of computing the bottom topography gradients
on the fly during the approximation as well as computing auxiliary quantities like the total water height ``H``
or the entropy variables.
This affects the implementation and use of these equations in various ways:
* The flux values corresponding to the bottom topography must be zero.
* The bottom topography values must be included when defining initial conditions, boundary conditions or
  source terms.
* [`AnalysisCallback`](@ref) analyzes this variable.
* Trixi.jl's visualization tools will visualize the bottom topography by default.

References for the SWE are many but a good introduction is available in Chapter 13 of the book:
- Randall J. LeVeque (2002)
  Finite Volume Methods for Hyperbolic Problems
  [DOI: 10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)
"""
struct ShallowWaterEquations3D{RealT <: Real} <: AbstractShallowWaterEquations{3, 5}
    # TODO: TrixiShallowWater: where should the `threshold_limiter` and `threshold_wet` live?
    # how to "properly" export these constants across the two packages?
    gravity::RealT # gravitational constant
    H0::RealT      # constant "lake-at-rest" total water height
    # `threshold_limiter` used in `PositivityPreservingLimiterShallowWater` on water height,
    # as a (small) shift on the initial condition and cutoff before the next time step.
    # Default is 500*eps() which in double precision is ≈1e-13.
    threshold_limiter::RealT
    # `threshold_wet` applied on water height to define when the flow is "wet"
    # before calculating the numerical flux.
    # Default is 5*eps() which in double precision is ≈1e-15.
    threshold_wet::RealT
end

# Allow for flexibility to set the gravitational constant within an elixir depending on the
# application where `gravity_constant=1.0` or `gravity_constant=9.81` are common values.
# The reference total water height H0 defaults to 0.0 but is used for the "lake-at-rest"
# well-balancedness test cases.
# Strict default values for thresholds that performed well in many numerical experiments
function ShallowWaterEquations3D(; gravity_constant, H0 = zero(gravity_constant),
                                 threshold_limiter = nothing, threshold_wet = nothing)
    T = promote_type(typeof(gravity_constant), typeof(H0))
    if threshold_limiter === nothing
        threshold_limiter = 500 * eps(T)
    end
    if threshold_wet === nothing
        threshold_wet = 5 * eps(T)
    end
    ShallowWaterEquations3D(gravity_constant, H0, threshold_limiter, threshold_wet)
end

have_nonconservative_terms(::ShallowWaterEquations3D) = False() # Deactivate non-conservative terms for the moment...
varnames(::typeof(cons2cons), ::ShallowWaterEquations3D) = ("h", "h_v1", "h_v2", "h_v3", "b")
# Note, we use the total water height, H = h + b, as the first primitive variable for easier
# visualization and setting initial conditions
varnames(::typeof(cons2prim), ::ShallowWaterEquations3D) = ("H", "v1", "v2", "v3", "b")

# Calculate 1D flux for a single point
# Note, the bottom topography has no flux
@inline function flux(u, orientation::Integer, equations::ShallowWaterEquations3D)
    h, h_v1, h_v2, h_v3, _ = u
    v1, v2, v3 = velocity(u, equations)

    p = 0.5 * equations.gravity * h^2
    if orientation == 1
        f1 = h_v1
        f2 = h_v1 * v1 + p
        f3 = h_v1 * v2
        f4 = h_v1 * v3
    elseif orientation == 2
        f1 = h_v2
        f2 = h_v2 * v1
        f3 = h_v2 * v2 + p
        f4 = h_v2 * v3
    else # orientation == 3
        f1 = h_v3
        f2 = h_v3 * v1
        f3 = h_v3 * v2
        f4 = h_v3 * v3 + p
    end
    return SVector(f1, f2, f3, f4, zero(eltype(u)))
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized and the bottom topography has no flux
@inline function flux(u, normal_direction::AbstractVector,
                      equations::ShallowWaterEquations3D)
    h = waterheight(u, equations)
    v1, v2, v3 = velocity(u, equations)

    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2] + v3 * normal_direction[3]
    h_v_normal = h * v_normal
    p = 0.5 * equations.gravity * h^2

    f1 = h_v_normal
    f2 = h_v_normal * v1 + p * normal_direction[1]
    f3 = h_v_normal * v2 + p * normal_direction[2]
    f4 = h_v_normal * v3 + p * normal_direction[3]
    return SVector(f1, f2, f3, f4, zero(eltype(u)))
end

"""
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                          equations::ShallowWaterEquations3D)
    flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                          normal_direction_ll     ::AbstractVector,
                                          normal_direction_average::AbstractVector,
                                          equations::ShallowWaterEquations3D)

Non-symmetric two-point volume flux discretizing the nonconservative (source) term
that contains the gradient of the bottom topography [`ShallowWaterEquations3D`](@ref).

On curvilinear meshes, this nonconservative flux depends on both the
contravariant vector (normal direction) at the current node and the averaged
one. This is different from numerical fluxes used to discretize conservative
terms.

Further details are available in the paper:
- Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
  An entropy stable nodal discontinuous Galerkin method for the two dimensional
  shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
  [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)
"""
@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                                       equations::ShallowWaterEquations3D)
    # Pull the necessary left and right state information
    h_ll = waterheight(u_ll, equations)
    b_rr = u_rr[5]

    z = zero(eltype(u_ll))
    # Bottom gradient nonconservative term: (0, g h b_x, g h b_y, 0)
    if orientation == 1
        f = SVector(z, equations.gravity * h_ll * b_rr, z, z, z)
    elseif orientation == 2
        f = SVector(z, z, equations.gravity * h_ll * b_rr, z, z)
    else # orientation == 3
        f = SVector(z, z, z, equations.gravity * h_ll * b_rr, z)
    end
    return f
end

@inline function flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                                       normal_direction_ll::AbstractVector,
                                                       normal_direction_average::AbstractVector,
                                                       equations::ShallowWaterEquations3D)
    # Pull the necessary left and right state information
    h_ll = waterheight(u_ll, equations)
    b_rr = u_rr[5]
    # Note this routine only uses the `normal_direction_average` and the average of the
    # bottom topography to get a quadratic split form DG gradient on curved elements
    return SVector(zero(eltype(u_ll)),
                   normal_direction_average[1] * equations.gravity * h_ll * b_rr,
                   normal_direction_average[2] * equations.gravity * h_ll * b_rr,
                   normal_direction_average[3] * equations.gravity * h_ll * b_rr,
                   zero(eltype(u_ll)))
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::ShallowWaterEquations3D)
    # Get the velocity quantities in the appropriate direction
    if orientation == 1
        v_ll, _, _ = velocity(u_ll, equations)
        v_rr, _, _ = velocity(u_rr, equations)
    elseif orientation == 2
        _, v_ll, _ = velocity(u_ll, equations)
        _, v_rr, _ = velocity(u_rr, equations)
    else #if orientation == 3
        _, _, v_ll = velocity(u_ll, equations)
        _, _, v_rr = velocity(u_rr, equations)
    end

    # Calculate the wave celerity on the left and right
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    c_ll = sqrt(equations.gravity * h_ll)
    c_rr = sqrt(equations.gravity * h_rr)

    return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::ShallowWaterEquations3D)
    # Extract and compute the velocities in the normal direction
    v1_ll, v2_ll, v3_ll = velocity(u_ll, equations)
    v1_rr, v2_rr, v3_rr = velocity(u_rr, equations)
    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] + v3_ll * normal_direction[3]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] + v3_rr * normal_direction[3]

    # Compute the wave celerity on the left and right
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    c_ll = sqrt(equations.gravity * h_ll)
    c_rr = sqrt(equations.gravity * h_rr)

    # The normal velocities are already scaled by the norm
    return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end

# Specialized `DissipationLocalLaxFriedrichs` to avoid spurious dissipation in the bottom topography
@inline function (dissipation::DissipationLocalLaxFriedrichs)(u_ll, u_rr,
                                                              orientation_or_normal_direction,
                                                              equations::ShallowWaterEquations3D)
    λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                                  equations)
    diss = -0.5 * λ * (u_rr - u_ll)
    return SVector(diss[1], diss[2], diss[3], diss[4], zero(eltype(u_ll)))
end

@inline function max_abs_speeds(u, equations::ShallowWaterEquations3D)
    h = waterheight(u, equations)
    v1, v2, v3 = velocity(u, equations)

    c = equations.gravity * sqrt(h)
    return abs(v1) + c, abs(v2) + c, abs(v3) + c
end

# Helper function to extract the velocity vector from the conservative variables
@inline function velocity(u, equations::ShallowWaterEquations3D)
    h, h_v1, h_v2, h_v3, _ = u

    v1 = h_v1 / h
    v2 = h_v2 / h
    v3 = h_v3 / h
    return SVector(v1, v2, v3)
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::ShallowWaterEquations3D)
    h, _, _, _, b = u

    H = h + b
    v1, v2, v3 = velocity(u, equations)
    return SVector(H, v1, v2, v3, b)
end

# Convert conservative variables to entropy
# Note, only the first three are the entropy variables, the fourth entry still
# just carries the bottom topography values for convenience
@inline function cons2entropy(u, equations::ShallowWaterEquations3D)
    h, h_v1, h_v2, h_v3, b = u

    v1, v2, v3 = velocity(u, equations)
    v_square = v1^2 + v2^2 + v3^2

    w1 = equations.gravity * (h + b) - 0.5 * v_square
    w2 = v1
    w3 = v2
    w4 = v3
    return SVector(w1, w2, w3, w4, b)
end

# Convert entropy variables to conservative
@inline function entropy2cons(w, equations::ShallowWaterEquations3D)
    w1, w2, w3, w4, b = w

    h = (w1 + 0.5 * (w2^2 + w3^2 + w4^2)) / equations.gravity - b
    h_v1 = h * w2
    h_v2 = h * w3
    h_v3 = h * w4
    return SVector(h, h_v1, h_v2, h_v3, b)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::ShallowWaterEquations3D)
    H, v1, v2, v3, b = prim

    h = H - b
    h_v1 = h * v1
    h_v2 = h * v2
    h_v3 = h * v3
    return SVector(h, h_v1, h_v2, h_v3, b)
end

@inline function waterheight(u, equations::ShallowWaterEquations3D)
    return u[1]
end

@inline function pressure(u, equations::ShallowWaterEquations3D)
    h = waterheight(u, equations)
    p = 0.5 * equations.gravity * h^2
    return p
end

@inline function waterheight_pressure(u, equations::ShallowWaterEquations3D)
    return waterheight(u, equations) * pressure(u, equations)
end

# Entropy function for the shallow water equations is the total energy
@inline function entropy(cons, equations::ShallowWaterEquations3D)
    energy_total(cons, equations)
end

# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equations::ShallowWaterEquations3D)
    h, h_v1, h_v2, h_v3, b = cons

    e = (h_v1^2 + h_v2^2 + h_v3^2) / (2 * h) + 0.5 * equations.gravity * h^2 +
        equations.gravity * h * b
    return e
end

# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::ShallowWaterEquations3D)
    h, h_v1, h_v2, h_v3, _ = u
    return (h_v1^2 + h_v2^2 + h_v3^2) / (2 * h)
end

# Calculate potential energy for a conservative state `cons`
@inline function energy_internal(cons, equations::ShallowWaterEquations3D)
    return energy_total(cons, equations) - energy_kinetic(cons, equations)
end

# Calculate the error for the "lake-at-rest" test case where H = h+b should
# be a constant value over time. Note, assumes there is a single reference
# water height `H0` with which to compare.
#
# TODO: TrixiShallowWater: where should `threshold_limiter` live? May need
# to modify or have different versions of the `lake_at_rest_error` function
@inline function lake_at_rest_error(u, equations::ShallowWaterEquations3D)
    h, _, _, _, b = u

    # For well-balancedness testing with possible wet/dry regions the reference
    # water height `H0` accounts for the possibility that the bottom topography
    # can emerge out of the water as well as for the threshold offset to avoid
    # division by a "hard" zero water heights as well.
    H0_wet_dry = max(equations.H0, b + equations.threshold_limiter)

    return abs(H0_wet_dry - (h + b))
end
end # @muladd
