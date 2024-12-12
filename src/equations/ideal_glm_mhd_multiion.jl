# This file includes functions that are common for all AbstractIdealGlmMhdMultiIonEquations

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

have_nonconservative_terms(::AbstractIdealGlmMhdMultiIonEquations) = True()

function varnames(::typeof(cons2cons), equations::AbstractIdealGlmMhdMultiIonEquations)
    cons = ("B1", "B2", "B3")
    for i in eachcomponent(equations)
        cons = (cons...,
                tuple("rho_" * string(i), "rho_v1_" * string(i), "rho_v2_" * string(i),
                      "rho_v3_" * string(i), "rho_e_" * string(i))...)
    end
    cons = (cons..., "psi")

    return cons
end

function varnames(::typeof(cons2prim), equations::AbstractIdealGlmMhdMultiIonEquations)
    prim = ("B1", "B2", "B3")
    for i in eachcomponent(equations)
        prim = (prim...,
                tuple("rho_" * string(i), "v1_" * string(i), "v2_" * string(i),
                      "v3_" * string(i), "p_" * string(i))...)
    end
    prim = (prim..., "psi")

    return prim
end

function default_analysis_integrals(::AbstractIdealGlmMhdMultiIonEquations)
    (entropy_timederivative, Val(:l2_divb), Val(:linf_divb))
end

"""
    source_terms_lorentz(u, x, t, equations::AbstractIdealGlmMhdMultiIonEquations)

Source terms due to the Lorentz' force for plasmas with more than one ion species. These source 
terms are a fundamental, inseparable part of the multi-ion GLM-MHD equations, and vanish for 
a single-species plasma. In particular, they have to be used for every
simulation of [`IdealGlmMhdMultiIonEquations1D`](@ref), [`IdealGlmMhdMultiIonEquations2D`](@ref),
and [`IdealGlmMhdMultiIonEquations3D`](@ref).
"""
function source_terms_lorentz(u, x, t, equations::AbstractIdealGlmMhdMultiIonEquations)
    @unpack charge_to_mass = equations
    B1, B2, B3 = magnetic_field(u, equations)
    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus = charge_averaged_velocities(u,
                                                                                         equations)

    s = zero(MVector{nvariables(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        rho_inv = 1 / rho
        v1 = rho_v1 * rho_inv
        v2 = rho_v2 * rho_inv
        v3 = rho_v3 * rho_inv
        v1_diff = v1_plus - v1
        v2_diff = v2_plus - v2
        v3_diff = v3_plus - v3
        r_rho = charge_to_mass[k] * rho
        s2 = r_rho * (v2_diff * B3 - v3_diff * B2)
        s3 = r_rho * (v3_diff * B1 - v1_diff * B3)
        s4 = r_rho * (v1_diff * B2 - v2_diff * B1)
        s5 = v1 * s2 + v2 * s3 + v3 * s4

        set_component!(s, k, 0, s2, s3, s4, s5, equations)
    end

    return SVector(s)
end

"""
    electron_pressure_zero(u, equations::AbstractIdealGlmMhdMultiIonEquations)

Returns the value of zero for the electron pressure. Needed for consistency with the 
single-fluid MHD equations in the limit of one ion species.
"""
function electron_pressure_zero(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    return zero(u[1])
end

"""
    v1, v2, v3, vk1, vk2, vk3 = charge_averaged_velocities(u,
                                                       equations::AbstractIdealGlmMhdMultiIonEquations)


Compute the charge-averaged velocities (`v1`, `v2`, and `v3`) and each ion species' contribution
to the charge-averaged velocities (`vk1`, `vk2`, and `vk3`). The output variables `vk1`, `vk2`, and `vk3`
are `SVectors` of size `ncomponents(equations)`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function charge_averaged_velocities(u,
                                            equations::AbstractIdealGlmMhdMultiIonEquations)
    total_electron_charge = zero(real(equations))

    vk1_plus = zero(MVector{ncomponents(equations), eltype(u)})
    vk2_plus = zero(MVector{ncomponents(equations), eltype(u)})
    vk3_plus = zero(MVector{ncomponents(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, _ = get_component(k, u, equations)

        total_electron_charge += rho * equations.charge_to_mass[k]
        vk1_plus[k] = rho_v1 * equations.charge_to_mass[k]
        vk2_plus[k] = rho_v2 * equations.charge_to_mass[k]
        vk3_plus[k] = rho_v3 * equations.charge_to_mass[k]
    end
    vk1_plus ./= total_electron_charge
    vk2_plus ./= total_electron_charge
    vk3_plus ./= total_electron_charge
    v1_plus = sum(vk1_plus)
    v2_plus = sum(vk2_plus)
    v3_plus = sum(vk3_plus)

    return v1_plus, v2_plus, v3_plus, SVector(vk1_plus), SVector(vk2_plus),
           SVector(vk3_plus)
end

"""
    get_component(k, u, equations::AbstractIdealGlmMhdMultiIonEquations)

Get the hydrodynamic variables of component (ion species) `k`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function get_component(k, u, equations::AbstractIdealGlmMhdMultiIonEquations)
    return SVector(u[3 + (k - 1) * 5 + 1],
                   u[3 + (k - 1) * 5 + 2],
                   u[3 + (k - 1) * 5 + 3],
                   u[3 + (k - 1) * 5 + 4],
                   u[3 + (k - 1) * 5 + 5])
end

"""
    set_component!(u, k, u1, u2, u3, u4, u5,
                   equations::AbstractIdealGlmMhdMultiIonEquations)

Set the hydrodynamic variables (`u1` to `u5`) of component (ion species) `k`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function set_component!(u, k, u1, u2, u3, u4, u5,
                                equations::AbstractIdealGlmMhdMultiIonEquations)
    u[3 + (k - 1) * 5 + 1] = u1
    u[3 + (k - 1) * 5 + 2] = u2
    u[3 + (k - 1) * 5 + 3] = u3
    u[3 + (k - 1) * 5 + 4] = u4
    u[3 + (k - 1) * 5 + 5] = u5

    return u
end

# Extract magnetic field from solution vector
magnetic_field(u, equations::AbstractIdealGlmMhdMultiIonEquations) = SVector(u[1], u[2],
                                                                             u[3])

# Extract GLM divergence-cleaning field from solution vector
divergence_cleaning_field(u, equations::AbstractIdealGlmMhdMultiIonEquations) = u[end]

# Get total density as the sum of the individual densities of the ion species
@inline function density(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    rho = zero(real(equations))
    for k in eachcomponent(equations)
        rho += u[3 + (k - 1) * 5 + 1]
    end
    return rho
end

@inline function pressure(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    B1, B2, B3, _ = u
    p = zero(MVector{ncomponents(equations), real(equations)})
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        gamma = equations.gammas[k]
        p[k] = (gamma - 1) * (rho_e - 0.5f0 * rho * v_mag^2 - 0.5f0 * (B1^2 + B2^2 + B3^2))
    end
    return SVector{ncomponents(equations), real(equations)}(p)
end

# Computes the sum of the densities times the sum of the pressures
@inline function density_pressure(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    B1, B2, B3 = magnetic_field(u, equations)
    psi = divergence_cleaning_field(cons, equations)

    rho_total = zero(real(equations))
    p_total = zero(real(equations))
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        gamma = equations.gammas[k]

        p = (gamma - 1) *
            (rho_e - 0.5f0 * rho * v_mag^2 - 0.5f0 * (B1^2 + B2^2 + B3^2) - 0.5f0 * psi^2)

        rho_total += rho
        p_total += p
    end
    return rho_total * p_total
end

#Convert conservative variables to primitive
function cons2prim(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(u, equations)
    psi = divergence_cleaning_field(u, equations)

    prim = zero(MVector{nvariables(equations), eltype(u)})
    prim[1] = B1
    prim[2] = B2
    prim[3] = B3
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        rho_inv = 1 / rho
        v1 = rho_inv * rho_v1
        v2 = rho_inv * rho_v2
        v3 = rho_inv * rho_v3

        p = (gammas[k] - 1) * (rho_e -
             0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3
              + B1 * B1 + B2 * B2 + B3 * B3
              + psi * psi))

        set_component!(prim, k, rho, v1, v2, v3, p, equations)
    end
    prim[end] = psi

    return SVector(prim)
end

#Convert conservative variables to entropy variables
@inline function cons2entropy(u, equations::AbstractIdealGlmMhdMultiIonEquations)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(u, equations)
    psi = divergence_cleaning_field(u, equations)

    prim = cons2prim(u, equations)
    entropy = zero(MVector{nvariables(equations), eltype(u)})
    rho_p_plus = zero(real(equations))
    for k in eachcomponent(equations)
        rho, v1, v2, v3, p = get_component(k, prim, equations)
        s = log(p) - gammas[k] * log(rho)
        rho_p = rho / p
        w1 = (gammas[k] - s) / (gammas[k] - 1) - 0.5f0 * rho_p * (v1^2 + v2^2 + v3^2)
        w2 = rho_p * v1
        w3 = rho_p * v2
        w4 = rho_p * v3
        w5 = -rho_p
        rho_p_plus += rho_p

        set_component!(entropy, k, w1, w2, w3, w4, w5, equations)
    end

    # Additional non-conservative variables
    entropy[1] = rho_p_plus * B1
    entropy[2] = rho_p_plus * B2
    entropy[3] = rho_p_plus * B3
    entropy[end] = rho_p_plus * psi

    return SVector(entropy)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::AbstractIdealGlmMhdMultiIonEquations)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(prim, equations)
    psi = divergence_cleaning_field(prim, equations)

    cons = zero(MVector{nvariables(equations), eltype(prim)})
    cons[1] = B1
    cons[2] = B2
    cons[3] = B3
    for k in eachcomponent(equations)
        rho, v1, v2, v3, p = get_component(k, prim, equations)
        rho_v1 = rho * v1
        rho_v2 = rho * v2
        rho_v3 = rho * v3

        rho_e = p / (gammas[k] - 1) +
                0.5f0 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) +
                0.5f0 * (B1^2 + B2^2 + B3^2) +
                0.5f0 * psi^2

        set_component!(cons, k, rho, rho_v1, rho_v2, rho_v3, rho_e, equations)
    end
    cons[end] = psi

    return SVector(cons)
end

# Specialization of DissipationEntropyStable for the multi-ion GLM-MHD equations
@inline function (dissipation::DissipationEntropyStable)(u_ll, u_rr,
                                                         orientation_or_normal_direction,
                                                         equations::AbstractIdealGlmMhdMultiIonEquations)
    @unpack gammas = equations
    λ = dissipation.max_abs_speed(u_ll, u_rr, orientation_or_normal_direction,
                                  equations)

    w_ll = cons2entropy(u_ll, equations)
    w_rr = cons2entropy(u_rr, equations)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)
    psi_ll = divergence_cleaning_field(u_ll, equations)
    psi_rr = divergence_cleaning_field(u_rr, equations)

    # Some global averages
    B1_avg = 0.5f0 * (B1_ll + B1_rr)
    B2_avg = 0.5f0 * (B2_ll + B2_rr)
    B3_avg = 0.5f0 * (B3_ll + B3_rr)
    psi_avg = 0.5f0 * (psi_ll + psi_rr)

    dissipation = zero(MVector{nvariables(equations), eltype(u_ll)})

    beta_plus_ll = 0
    beta_plus_rr = 0
    # Get the lumped dissipation for all components
    for k in eachcomponent(equations)
        rho_ll, v1_ll, v2_ll, v3_ll, p_ll = get_component(k, prim_ll, equations)
        rho_rr, v1_rr, v2_rr, v3_rr, p_rr = get_component(k, prim_rr, equations)

        w1_ll, w2_ll, w3_ll, w4_ll, w5_ll = get_component(k, w_ll, equations)
        w1_rr, w2_rr, w3_rr, w4_rr, w5_rr = get_component(k, w_rr, equations)

        # Auxiliary variables
        beta_ll = 0.5f0 * rho_ll / p_ll
        beta_rr = 0.5f0 * rho_rr / p_rr
        vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
        vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2

        # Mean variables
        rho_ln = ln_mean(rho_ll, rho_rr)
        beta_ln = ln_mean(beta_ll, beta_rr)
        rho_avg = 0.5f0 * (rho_ll + rho_rr)
        v1_avg = 0.5f0 * (v1_ll + v1_rr)
        v2_avg = 0.5f0 * (v2_ll + v2_rr)
        v3_avg = 0.5f0 * (v3_ll + v3_rr)
        beta_avg = 0.5f0 * (beta_ll + beta_rr)
        tau = 1 / (beta_ll + beta_rr)
        p_mean = 0.5f0 * rho_avg / beta_avg
        p_star = 0.5f0 * rho_ln / beta_ln
        vel_norm_avg = 0.5f0 * (vel_norm_ll + vel_norm_rr)
        vel_avg_norm = v1_avg^2 + v2_avg^2 + v3_avg^2
        E_bar = p_star / (gammas[k] - 1) +
                0.5f0 * rho_ln * (2 * vel_avg_norm - vel_norm_avg)

        h11 = rho_ln
        h12 = rho_ln * v1_avg
        h13 = rho_ln * v2_avg
        h14 = rho_ln * v3_avg
        h15 = E_bar
        d1 = -0.5f0 * λ *
             (h11 * (w1_rr - w1_ll) +
              h12 * (w2_rr - w2_ll) +
              h13 * (w3_rr - w3_ll) +
              h14 * (w4_rr - w4_ll) +
              h15 * (w5_rr - w5_ll))

        h21 = h12
        h22 = rho_ln * v1_avg^2 + p_mean
        h23 = h21 * v2_avg
        h24 = h21 * v3_avg
        h25 = (E_bar + p_mean) * v1_avg
        d2 = -0.5f0 * λ *
             (h21 * (w1_rr - w1_ll) +
              h22 * (w2_rr - w2_ll) +
              h23 * (w3_rr - w3_ll) +
              h24 * (w4_rr - w4_ll) +
              h25 * (w5_rr - w5_ll))

        h31 = h13
        h32 = h23
        h33 = rho_ln * v2_avg^2 + p_mean
        h34 = h31 * v3_avg
        h35 = (E_bar + p_mean) * v2_avg
        d3 = -0.5f0 * λ *
             (h31 * (w1_rr - w1_ll) +
              h32 * (w2_rr - w2_ll) +
              h33 * (w3_rr - w3_ll) +
              h34 * (w4_rr - w4_ll) +
              h35 * (w5_rr - w5_ll))

        h41 = h14
        h42 = h24
        h43 = h34
        h44 = rho_ln * v3_avg^2 + p_mean
        h45 = (E_bar + p_mean) * v3_avg
        d4 = -0.5f0 * λ *
             (h41 * (w1_rr - w1_ll) +
              h42 * (w2_rr - w2_ll) +
              h43 * (w3_rr - w3_ll) +
              h44 * (w4_rr - w4_ll) +
              h45 * (w5_rr - w5_ll))

        h51 = h15
        h52 = h25
        h53 = h35
        h54 = h45
        h55 = ((p_star^2 / (gammas[k] - 1) + E_bar * E_bar) / rho_ln
               +
               vel_avg_norm * p_mean)
        d5 = -0.5f0 * λ *
             (h51 * (w1_rr - w1_ll) +
              h52 * (w2_rr - w2_ll) +
              h53 * (w3_rr - w3_ll) +
              h54 * (w4_rr - w4_ll) +
              h55 * (w5_rr - w5_ll))

        beta_plus_ll += beta_ll
        beta_plus_rr += beta_rr

        set_component!(dissipation, k, d1, d2, d3, d4, d5, equations)
    end

    # Set the magnetic field and psi terms
    h_B_psi = 1 / (beta_plus_ll + beta_plus_rr)

    # diagonal entries
    dissipation[1] = -0.5f0 * λ * h_B_psi * (w_rr[1] - w_ll[1])
    dissipation[2] = -0.5f0 * λ * h_B_psi * (w_rr[2] - w_ll[2])
    dissipation[3] = -0.5f0 * λ * h_B_psi * (w_rr[3] - w_ll[3])
    dissipation[end] = -0.5f0 * λ * h_B_psi * (w_rr[end] - w_ll[end])
    # Off-diagonal entries
    for k in eachcomponent(equations)
        _, _, _, _, w5_ll = get_component(k, w_ll, equations)
        _, _, _, _, w5_rr = get_component(k, w_rr, equations)

        dissipation[1] -= 0.5f0 * λ * h_B_psi * B1_avg * (w5_rr - w5_ll)
        dissipation[2] -= 0.5f0 * λ * h_B_psi * B2_avg * (w5_rr - w5_ll)
        dissipation[3] -= 0.5f0 * λ * h_B_psi * B3_avg * (w5_rr - w5_ll)
        dissipation[end] -= 0.5f0 * λ * h_B_psi * psi_avg * (w5_rr - w5_ll)

        # Dissipation for the energy equation of species k depending on w_1, w_2, w_3 and w_end
        ind_E = 3 + (k - 1) * 5 + 5
        dissipation[ind_E] -= 0.5f0 * λ * h_B_psi * B1_avg * (w_rr[1] - w_ll[1])
        dissipation[ind_E] -= 0.5f0 * λ * h_B_psi * B2_avg * (w_rr[2] - w_ll[2])
        dissipation[ind_E] -= 0.5f0 * λ * h_B_psi * B3_avg * (w_rr[3] - w_ll[3])
        dissipation[ind_E] -= 0.5f0 * λ * h_B_psi * psi_avg * (w_rr[end] - w_ll[end])

        # Dissipation for the energy equation of all ion species depending on w_5
        for kk in eachcomponent(equations)
            ind_E = 3 + (kk - 1) * 5 + 5
            dissipation[ind_E] -= 0.5f0 * λ *
                                  (h_B_psi *
                                   (B1_avg^2 + B2_avg^2 + B3_avg^2 + psi_avg^2)) *
                                  (w5_rr - w5_ll)
        end
    end

    return dissipation
end

"""
    source_terms_collision_ion_ion(u, x, t,
                                   equations::AbstractIdealGlmMhdMultiIonEquations)

Ion-ion collision source terms, cf. Rueda-Ramirez et al. (2023) and Rubin et al. (2015)
"""
function source_terms_collision_ion_ion(u, x, t,
                                        equations::AbstractIdealGlmMhdMultiIonEquations)
    S_std = source_terms_standard(u, x, t, equations)

    s = zero(MVector{nvariables(equations), eltype(u)})
    @unpack gammas, gas_constants, molar_masses, collision_frequency = equations

    prim = cons2prim(u, equations)

    for k in eachcomponent(equations)
        rho_k, v1_k, v2_k, v3_k, p_k = get_component(k, prim, equations)
        T_k = p_k / (rho_k * gas_constants[k])

        S_q1 = 0
        S_q2 = 0
        S_q3 = 0
        S_E = 0
        for l in eachcomponent(equations)
            # Skip computation for same species
            l == k && continue

            rho_l, v1_l, v2_l, v3_l, p_l = get_component(l, prim, equations)
            T_l = p_l / (rho_l * gas_constants[l])

            # Reduced temperature (without scaling with molar mass)
            T_kl = (molar_masses[l] * T_k + molar_masses[k] * T_l)

            delta_v2 = (v1_l - v1_k)^2 + (v2_l - v2_k)^2 + (v3_l - v3_k)^2

            # Scale T_kl with molar mass
            T_kl /= (molar_masses[k] + molar_masses[l])

            # Compute effective collision frequency
            v_kl = (collision_frequency * (rho_l * molar_masses[1] / molar_masses[l]) /
                    T_kl^(3 / 2))

            # Correct the collision frequency with the drifting effect (NEW - Rambo & Denavit, Rambo & Procassini)
            z2 = delta_v2 / (p_l / rho_l + p_k / rho_k)
            v_kl /= (1 + (2 / (9 * pi))^(1 / 3) * z2)^(3/2)

            S_q1 += rho_k * v_kl * (v1_l - v1_k)
            S_q2 += rho_k * v_kl * (v2_l - v2_k)
            S_q3 += rho_k * v_kl * (v3_l - v3_k)

            S_E += (3 * molar_masses[1] * gas_constants[1] * (T_l - T_k)
                    +
                    molar_masses[l] * delta_v2) * v_kl * rho_k /
                   (molar_masses[k] + molar_masses[l])
        end

        S_E += (v1_k * S_q1 + v2_k * S_q2 + v3_k * S_q3)

        set_component!(s, k, 0, S_q1, S_q2, S_q3, S_E, equations)
    end
    return SVector{nvariables(equations), real(equations)}(S_std .+ s)
end

"""
    source_terms_collision_ion_electron(u, x, t,
                                        equations::AbstractIdealGlmMhdMultiIonEquations)

Ion-electron collision source terms, cf. Rueda-Ramirez et al. (2023) and Rubin et al. (2015)
Here we assume v_e = v⁺ (no effect of currents on the electron velocity)
"""
function source_terms_collision_ion_electron(u, x, t,
                                             equations::AbstractIdealGlmMhdMultiIonEquations)
    s = zero(MVector{nvariables(equations), eltype(u)})
    @unpack gammas, gas_constants, molar_masses, ion_electron_collision_constants, electron_temperature = equations

    prim = cons2prim(u, equations)
    T_e = electron_temperature(u, equations)
    T_e32 = T_e^(3 / 2)

    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus, total_electron_charge = charge_averaged_velocities(u,
                                                                                                                equations)

    for k in eachcomponent(equations)
        rho_k, v1_k, v2_k, v3_k, p_k = get_component(k, prim, equations)
        T_k = p_k / (rho_k * gas_constants[k])

        # Compute effective collision frequency
        v_ke = ion_electron_collision_constants[k] * total_electron_charge / T_e32

        S_q1 = rho_k * v_ke * (v1_plus - v1_k)
        S_q2 = rho_k * v_ke * (v2_plus - v2_k)
        S_q3 = rho_k * v_ke * (v3_plus - v3_k)

        S_E = 3 * molar_masses[1] * gas_constants[1] * (T_e - T_k) * v_ke * rho_k /
              molar_masses[k]

        S_E += (v1_k * S_q1 + v2_k * S_q2 + v3_k * S_q3)

        set_component!(s, k, 0, S_q1, S_q2, S_q3, S_E, equations)
    end
    return SVector{nvariables(equations), real(equations)}(s)
end

"""
    function source_terms_collision_ion_electron_ohm(u, x, t,
                                                     equations::AbstractIdealGlmMhdMultiIonEquations)

Ion-electron collision source terms, including the effect on Ohm's law, cf. Rueda-Ramirez et al. (2023) and Rubin et al. (2015)
Here we assume v_e = v⁺ (no effect of currents on the electron velocity)
"""
function source_terms_collision_ion_electron_ohm(u, x, t,
                                                 equations::AbstractIdealGlmMhdMultiIonEquations)
    s = zero(MVector{nvariables(equations), eltype(u)})
    @unpack gammas, gas_constants, charge_to_mass, molar_masses, ion_electron_collision_constants, electron_temperature = equations

    prim = cons2prim(u, equations)
    T_e = electron_temperature(u, equations)
    T_e32 = T_e^(3 / 2)

    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus, total_electron_charge = charge_averaged_velocities(u,
                                                                                                                equations)

    Se_q1 = 0
    Se_q2 = 0
    Se_q3 = 0

    for k in eachcomponent(equations)
        rho_k, v1_k, v2_k, v3_k, p_k = get_component(k, prim, equations)
        T_k = p_k / (rho_k * gas_constants[k])

        # Compute effective collision frequency
        v_ke = ion_electron_collision_constants[k] * total_electron_charge / T_e32

        S_q1 = rho_k * v_ke * (v1_plus - v1_k)
        S_q2 = rho_k * v_ke * (v2_plus - v2_k)
        S_q3 = rho_k * v_ke * (v3_plus - v3_k)
        Se_q1 -= S_q1
        Se_q2 -= S_q2
        Se_q3 -= S_q3

        S_E = 3 * molar_masses[1] * gas_constants[1] * (T_e - T_k) * v_ke * rho_k /
              molar_masses[k]

        S_E += (v1_k * S_q1 + v2_k * S_q2 + v3_k * S_q3)

        set_component!(s, k, 0, S_q1, S_q2, S_q3, S_E, equations)
    end

    s_ohm = zero(MVector{nvariables(equations), eltype(u)})
    for k in eachcomponent(equations)
        rho_k, v1_k, v2_k, v3_k, _ = get_component(k, prim, equations)

        S_q1 = rho_k * charge_to_mass[k] * Se_q1 / total_electron_charge
        S_q2 = rho_k * charge_to_mass[k] * Se_q2 / total_electron_charge
        S_q3 = rho_k * charge_to_mass[k] * Se_q3 / total_electron_charge

        S_E = (v1_k * S_q1 + v2_k * S_q2 + v3_k * S_q3)

        set_component!(s_ohm, k, 0, S_q1, S_q2, S_q3, S_E, equations)
    end

    return SVector{nvariables(equations), real(equations)}(s + s_ohm)
end
end
