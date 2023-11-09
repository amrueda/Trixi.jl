# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    IdealMhdMultiIonEquations2D

The ideal compressible multi-ion MHD equations in two space dimensions.
"""
mutable struct IdealMhdMultiIonEquations2D{NVARS, NCOMP, RealT <: Real,
                                           ElectronPressure, ElectronTemperature} <:
               AbstractIdealMhdMultiIonEquations{2, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT} # Heat capacity ratios
    charge_to_mass::SVector{NCOMP, RealT} # Charge to mass ratios
    gas_constants::SVector{NCOMP, RealT} # Specific gas constants
    molar_masses::SVector{NCOMP, RealT} # Molar masses (can be provided in any units as they are only used to compute ratios)
    collision_frequency::RealT               # Single collision frequency scaled with molecular mass of ion 1 (TODO: Replace by matrix of collision frequencies)
    ion_electron_collision_constants::SVector{NCOMP, RealT} # Constants for the ion-electron collision frequencies. The collision frequency is obtained as constant * (e * n_e) / T_e^1.5
    electron_pressure::ElectronPressure       # Function to compute the electron pressure
    electron_temperature::ElectronTemperature # Function to compute the electron temperature

    function IdealMhdMultiIonEquations2D{NVARS, NCOMP, RealT, ElectronPressure,
                                         ElectronTemperature}(gammas
                                                              ::SVector{NCOMP, RealT},
                                                              charge_to_mass
                                                              ::SVector{NCOMP, RealT},
                                                              gas_constants
                                                              ::SVector{NCOMP, RealT},
                                                              molar_masses
                                                              ::SVector{NCOMP, RealT},
                                                              collision_frequency
                                                              ::RealT,
                                                              ion_electron_collision_constants
                                                              ::SVector{NCOMP, RealT},
                                                              electron_pressure
                                                              ::ElectronPressure,
                                                              electron_temperature
                                                              ::ElectronTemperature) where
        {NVARS, NCOMP, RealT <: Real, ElectronPressure, ElectronTemperature}
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_to_mass` have to be filled with at least one value"))

        new(gammas, charge_to_mass, gas_constants, molar_masses, collision_frequency,
            ion_electron_collision_constants, electron_pressure, electron_temperature)
    end
end

function IdealMhdMultiIonEquations2D(; gammas, charge_to_mass, gas_constants,
                                     molar_masses, collision_frequency,
                                     ion_electron_collision_constants,
                                     electron_pressure = electron_pressure_zero,
                                     electron_temperature = electron_pressure_zero)
    _gammas = promote(gammas...)
    _charge_to_mass = promote(charge_to_mass...)
    _gas_constants = promote(gas_constants...)
    _molar_masses = promote(molar_masses...)
    _ion_electron_collision_constants = promote(ion_electron_collision_constants...)
    RealT = promote_type(eltype(_gammas), eltype(_charge_to_mass),
                         eltype(_gas_constants), eltype(_molar_masses),
                         eltype(collision_frequency),
                         eltype(_ion_electron_collision_constants))

    NVARS = length(_gammas) * 5 + 3
    NCOMP = length(_gammas)

    __gammas = SVector(map(RealT, _gammas))
    __charge_to_mass = SVector(map(RealT, _charge_to_mass))
    __gas_constants = SVector(map(RealT, _gas_constants))
    __molar_masses = SVector(map(RealT, _molar_masses))
    __collision_frequency = map(RealT, collision_frequency)
    __ion_electron_collision_constants = SVector(map(RealT,
                                                     _ion_electron_collision_constants))

    return IdealMhdMultiIonEquations2D{NVARS, NCOMP, RealT, typeof(electron_pressure),
                                       typeof(electron_temperature)}(__gammas,
                                                                     __charge_to_mass,
                                                                     __gas_constants,
                                                                     __molar_masses,
                                                                     __collision_frequency,
                                                                     __ion_electron_collision_constants,
                                                                     electron_pressure,
                                                                     electron_temperature)
end

@inline function Base.real(::IdealMhdMultiIonEquations2D{NVARS, NCOMP, RealT}) where {
                                                                                      NVARS,
                                                                                      NCOMP,
                                                                                      RealT
                                                                                      }
    RealT
end

have_nonconservative_terms(::IdealMhdMultiIonEquations2D) = True()

function varnames(::typeof(cons2cons), equations::IdealMhdMultiIonEquations2D)
    cons = ("B1", "B2", "B3")
    for i in eachcomponent(equations)
        cons = (cons...,
                tuple("rho_" * string(i), "rho_v1_" * string(i), "rho_v2_" * string(i),
                      "rho_v3_" * string(i), "rho_e_" * string(i))...)
    end

    return cons
end

function varnames(::typeof(cons2prim), equations::IdealMhdMultiIonEquations2D)
    prim = ("B1", "B2", "B3")
    for i in eachcomponent(equations)
        prim = (prim...,
                tuple("rho_" * string(i), "v1_" * string(i), "v2_" * string(i),
                      "v3_" * string(i), "p_" * string(i))...)
    end

    return prim
end

function default_analysis_integrals(::IdealMhdMultiIonEquations2D)
    (entropy_timederivative, Val(:l2_divb), Val(:linf_divb))
end

# """
#     initial_condition_convergence_test(x, t, equations::IdealMhdMultiIonEquations2D)

# An Alfvén wave as smooth initial condition used for convergence tests.
# """
# function initial_condition_convergence_test(x, t, equations::IdealMhdMultiIonEquations2D)
#   # smooth Alfvén wave test from Derigs et al. FLASH (2016)
#   # domain must be set to [0, 1], γ = 5/3

#   rho = 1.0
#   prim_rho  = SVector{ncomponents(equations), real(equations)}(2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * rho for i in eachcomponent(equations))
#   v1 = zero(real(equations))
#   si, co = sincos(2 * pi * x[1])
#   v2 = 0.1 * si
#   v3 = 0.1 * co
#   p = 0.1
#   B1 = 1.0
#   B2 = v2
#   B3 = v3
#   prim_other = SVector{7, real(equations)}(v1, v2, v3, p, B1, B2, B3)
#   return prim2cons(vcat(prim_other, prim_rho), equations)
# end

"""
    initial_condition_weak_blast_wave(x, t, equations::IdealMhdMultiIonEquations2D)

A weak blast wave adapted from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::IdealMhdMultiIonEquations2D)
    # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    # Same discontinuity in the velocities but with magnetic fields
    # Set up polar coordinates
    inicenter = (0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)

    # Calculate primitive variables
    rho = zero(real(equations))
    if r > 0.5
        rho = 1.0
    else
        rho = 1.1691
    end
    v1 = r > 0.5 ? 0.0 : 0.1882 * cos(phi)
    v2 = r > 0.5 ? 0.0 : 0.1882 * sin(phi)
    p = r > 0.5 ? 1.0 : 1.245

    prim = zero(MVector{nvariables(equations), real(equations)})
    prim[1] = 1.0
    prim[2] = 1.0
    prim[3] = 1.0
    for k in eachcomponent(equations)
        set_component!(prim, k,
                       2^(k - 1) * (1 - 2) / (1 - 2^ncomponents(equations)) * rho, v1,
                       v2, 0, p, equations)
    end

    return prim2cons(SVector(prim), equations)
end

# TODO: Add initial condition equilibrium

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::IdealMhdMultiIonEquations2D)
    B1, B2, B3 = magnetic_field(u, equations)

    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus, total_electron_charge = charge_averaged_velocities(u,
                                                                                                                equations)

    mag_en = 0.5 * (B1^2 + B2^2 + B3^2)

    f = zero(MVector{nvariables(equations), eltype(u)})

    if orientation == 1
        f[1] = 0
        f[2] = v1_plus * B2 - v2_plus * B1
        f[3] = v1_plus * B3 - v3_plus * B1

        for k in eachcomponent(equations)
            rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
            v1 = rho_v1 / rho
            v2 = rho_v2 / rho
            v3 = rho_v3 / rho
            kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)

            gamma = equations.gammas[k]
            p = (gamma - 1) * (rho_e - kin_en - mag_en)

            f1 = rho_v1
            f2 = rho_v1 * v1 + p
            f3 = rho_v1 * v2
            f4 = rho_v1 * v3
            f5 = (kin_en + gamma * p / (gamma - 1)) * v1 + 2 * mag_en * vk1_plus[k] -
                 B1 * (vk1_plus[k] * B1 + vk2_plus[k] * B2 + vk3_plus[k] * B3)

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end

    else #if orientation == 2
        f[1] = v2_plus * B1 - v1_plus * B2
        f[2] = 0
        f[3] = v2_plus * B3 - v3_plus * B2

        for k in eachcomponent(equations)
            rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
            v1 = rho_v1 / rho
            v2 = rho_v2 / rho
            v3 = rho_v3 / rho
            kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)

            gamma = equations.gammas[k]
            p = (gamma - 1) * (rho_e - kin_en - mag_en)

            f1 = rho_v2
            f2 = rho_v2 * v1
            f3 = rho_v2 * v2 + p
            f4 = rho_v2 * v3
            f5 = (kin_en + gamma * p / (gamma - 1)) * v2 + 2 * mag_en * vk2_plus[k] -
                 B2 * (vk1_plus[k] * B1 + vk2_plus[k] * B2 + vk3_plus[k] * B3)

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
    end

    return SVector(f)
end

"""
Standard source terms of the multi-ion MHD equations
"""
function source_terms_standard(u, x, t, equations::IdealMhdMultiIonEquations2D)
    @unpack charge_to_mass = equations
    B1, B2, B3 = magnetic_field(u, equations)
    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus, total_electron_charge = charge_averaged_velocities(u,
                                                                                                                equations)

    s = zero(MVector{nvariables(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
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
Ion-ion collision source terms, cf. Rueda-Ramirez et al. (2023) and Rubin et al. (2015)
"""
function source_terms_collision_ion_ion(u, x, t, equations::IdealMhdMultiIonEquations2D)
    S_std = source_terms_standard(u, x, t, equations)

    s = zero(MVector{nvariables(equations), eltype(u)})
    @unpack gammas, gas_constants, molar_masses, collision_frequency = equations

    prim = cons2prim(u, equations)

    for k in eachcomponent(equations)
        rho_k, v1_k, v2_k, v3_k, p_k = get_component(k, prim, equations)
        T_k = p_k / (rho_k * gas_constants[k])

        S_q1 = 0.0
        S_q2 = 0.0
        S_q3 = 0.0
        S_E = 0.0
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
            v_kl /= (1 + (2.0 / (9.0 * pi))^(1.0 / 3.0) * z2)^(1.5)

            S_q1 += rho_k * v_kl * (v1_l - v1_k)
            S_q2 += rho_k * v_kl * (v2_l - v2_k)
            S_q3 += rho_k * v_kl * (v3_l - v3_k)

            S_E += (3 * molar_masses[1] * gas_constants[1] * (T_l - T_k)
                    +
                    molar_masses[l] * delta_v2) * v_kl * rho_k /
                   (molar_masses[k] + molar_masses[l])
        end

        S_E += (v1_k * S_q1 + v2_k * S_q2 + v3_k * S_q3)

        set_component!(s, k, 0.0, S_q1, S_q2, S_q3, S_E, equations)
    end
    return SVector{nvariables(equations), real(equations)}(S_std .+ s)
end

"""
Ion-electron collision source terms, cf. Rueda-Ramirez et al. (2023) and Rubin et al. (2015)
Here we assume v_e = v⁺ (no effect of currents on the electron velocity)
"""
function source_terms_collision_ion_electron(u, x, t,
                                             equations::IdealMhdMultiIonEquations2D)
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

        set_component!(s, k, 0.0, S_q1, S_q2, S_q3, S_E, equations)
    end
    return SVector{nvariables(equations), real(equations)}(s)
end

"""
Ion-electron collision source terms, including the effect on Ohm's law, cf. Rueda-Ramirez et al. (2023) and Rubin et al. (2015)
Here we assume v_e = v⁺ (no effect of currents on the electron velocity)
"""
function source_terms_collision_ion_electron_ohm(u, x, t,
                                                 equations::IdealMhdMultiIonEquations2D)
    s = zero(MVector{nvariables(equations), eltype(u)})
    @unpack gammas, gas_constants, charge_to_mass, molar_masses, ion_electron_collision_constants, electron_temperature = equations

    prim = cons2prim(u, equations)
    T_e = electron_temperature(u, equations)
    T_e32 = T_e^(3 / 2)

    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus, total_electron_charge = charge_averaged_velocities(u,
                                                                                                                equations)

    Se_q1 = 0.0
    Se_q2 = 0.0
    Se_q3 = 0.0

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

        set_component!(s, k, 0.0, S_q1, S_q2, S_q3, S_E, equations)
    end

    s_ohm = zero(MVector{nvariables(equations), eltype(u)})
    for k in eachcomponent(equations)
        rho_k, v1_k, v2_k, v3_k, _ = get_component(k, prim, equations)

        S_q1 = rho_k * charge_to_mass[k] * Se_q1 / total_electron_charge
        S_q2 = rho_k * charge_to_mass[k] * Se_q2 / total_electron_charge
        S_q3 = rho_k * charge_to_mass[k] * Se_q3 / total_electron_charge

        S_E = (v1_k * S_q1 + v2_k * S_q2 + v3_k * S_q3)

        set_component!(s_ohm, k, 0.0, S_q1, S_q2, S_q3, S_E, equations)
    end

    return SVector{nvariables(equations), real(equations)}(s + s_ohm)
end

function electron_pressure_zero(u, equations::IdealMhdMultiIonEquations2D)
    return zero(u[1])
end

"""
Total entropy-conserving non-conservative two-point "flux"" as described in 
- Rueda-Ramírez et al. (2023)
The term is composed of three parts
* The Powell term: Implemented
* The MHD term: Implemented
* The "term 3": Implemented
"""
@inline function flux_nonconservative_ruedaramirez_etal(u_ll, u_rr,
                                                        orientation::Integer,
                                                        equations::IdealMhdMultiIonEquations2D)
    @unpack charge_to_mass = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)

    # Compute important averages
    B1_avg = 0.5 * (B1_ll + B1_rr)
    B2_avg = 0.5 * (B2_ll + B2_rr)
    B3_avg = 0.5 * (B3_ll + B3_rr)
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
    mag_norm_avg = 0.5 * (mag_norm_ll + mag_norm_rr)

    # Mean electron pressure
    pe_mean = 0.5 * (equations.electron_pressure(u_ll, equations) +
               equations.electron_pressure(u_rr, equations))

    # Compute charge ratio of u_ll
    charge_ratio_ll = zero(MVector{ncomponents(equations), eltype(u_ll)})
    total_electron_charge = zero(eltype(u_ll))
    for k in eachcomponent(equations)
        rho_k = u_ll[3 + (k - 1) * 5 + 1]
        charge_ratio_ll[k] = rho_k * charge_to_mass[k]
        total_electron_charge += charge_ratio_ll[k]
    end
    charge_ratio_ll ./= total_electron_charge

    # Compute auxiliary variables
    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll, _ = charge_averaged_velocities(u_ll,
                                                                                                              equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr, _ = charge_averaged_velocities(u_rr,
                                                                                                              equations)

    f = zero(MVector{nvariables(equations), eltype(u_ll)})

    if orientation == 1
        # Entries of Powell term for induction equation (already in Trixi's non-conservative form)
        f[1] = v1_plus_ll * B1_rr
        f[2] = v2_plus_ll * B1_rr
        f[3] = v3_plus_ll * B1_rr

        for k in eachcomponent(equations)
            # Compute term 2 (MHD)
            f2 = charge_ratio_ll[k] * (0.5 * mag_norm_avg - B1_avg * B1_avg + pe_mean)
            f3 = charge_ratio_ll[k] * (-B1_avg * B2_avg)
            f4 = charge_ratio_ll[k] * (-B1_avg * B3_avg)
            f5 = vk1_plus_ll[k] * pe_mean

            # Compute term 3 (only needed for NCOMP>1)
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            vk1_minus_avg = 0.5 * (vk1_minus_ll + vk1_minus_rr)
            vk2_minus_avg = 0.5 * (vk2_minus_ll + vk2_minus_rr)
            vk3_minus_avg = 0.5 * (vk3_minus_ll + vk3_minus_rr)
            f5 += (B2_ll * (vk1_minus_avg * B2_avg - vk2_minus_avg * B1_avg) +
                   B3_ll * (vk1_minus_avg * B3_avg - vk3_minus_avg * B1_avg))

            # Adjust non-conservative terms 2 and 3 to Trixi discretization: CHANGE!?!
            f2 = 2 * f2 - charge_ratio_ll[k] * (0.5 * mag_norm_ll - B1_ll * B1_ll)
            f3 = 2 * f3 + charge_ratio_ll[k] * B1_ll * B2_ll
            f4 = 2 * f4 + charge_ratio_ll[k] * B1_ll * B3_ll
            f5 = (2 * f5 - B2_ll * (vk1_minus_ll * B2_ll - vk2_minus_ll * B1_ll)
                  -
                  B3_ll * (vk1_minus_ll * B3_ll - vk3_minus_ll * B1_ll))

            # Compute Powell term (already consistent with Trixi's non-conservative discretization)
            f2 += charge_ratio_ll[k] * B1_ll * B1_rr
            f3 += charge_ratio_ll[k] * B2_ll * B1_rr
            f4 += charge_ratio_ll[k] * B3_ll * B1_rr
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) * B1_rr

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end

    else #if orientation == 2
        # Entries of Powell term for induction equation (already in Trixi's non-conservative form)
        f[1] = v1_plus_ll * B2_rr
        f[2] = v2_plus_ll * B2_rr
        f[3] = v3_plus_ll * B2_rr

        for k in eachcomponent(equations)
            # Compute term 2 (MHD)
            f2 = charge_ratio_ll[k] * (-B2_avg * B1_avg)
            f3 = charge_ratio_ll[k] * (-B2_avg * B2_avg + 0.5 * mag_norm_avg + pe_mean)
            f4 = charge_ratio_ll[k] * (-B2_avg * B3_avg)
            f5 = vk2_plus_ll[k] * pe_mean

            # Compute term 3 (only needed for NCOMP>1)
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            vk1_minus_avg = 0.5 * (vk1_minus_ll + vk1_minus_rr)
            vk2_minus_avg = 0.5 * (vk2_minus_ll + vk2_minus_rr)
            vk3_minus_avg = 0.5 * (vk3_minus_ll + vk3_minus_rr)
            f5 += (B1_ll * (vk2_minus_avg * B1_avg - vk1_minus_avg * B2_avg) +
                   B3_ll * (vk2_minus_avg * B3_avg - vk3_minus_avg * B2_avg))

            # Adjust non-conservative terms 2 and 3 to Trixi discretization: CHANGE!?!
            f2 = 2 * f2 + charge_ratio_ll[k] * B2_ll * B1_ll
            f3 = 2 * f3 - charge_ratio_ll[k] * (0.5 * mag_norm_ll - B2_ll * B2_ll)
            f4 = 2 * f4 + charge_ratio_ll[k] * B2_ll * B3_ll
            f5 = (2 * f5 - B1_ll * (vk2_minus_ll * B1_ll - vk1_minus_ll * B2_ll)
                  -
                  B3_ll * (vk2_minus_ll * B3_ll - vk3_minus_ll * B2_ll))

            # Compute Powell term (already consistent with Trixi's non-conservative discretization)
            f2 += charge_ratio_ll[k] * B1_ll * B2_rr
            f3 += charge_ratio_ll[k] * B2_ll * B2_rr
            f4 += charge_ratio_ll[k] * B3_ll * B2_rr
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) * B2_rr

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
    end

    return SVector(f)
end

"""
Total central non-conservative two-point "flux"", where the symmetric parts are computed with standard averages
The term is composed of three parts
* The Powell term: Implemented. The central Powell "flux" is equivalent to the EC Powell "flux".
* The MHD term: Implemented
* The "term 3": Implemented
"""
@inline function flux_nonconservative_central(u_ll, u_rr, orientation::Integer,
                                              equations::IdealMhdMultiIonEquations2D)
    @unpack charge_to_mass = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)

    # Compute important averages
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2

    # Electron pressure 
    pe_rr = equations.electron_pressure(u_rr, equations)

    # Compute charge ratio of u_ll
    charge_ratio_ll = zero(MVector{ncomponents(equations), eltype(u_ll)})
    total_electron_charge = zero(real(equations))
    for k in eachcomponent(equations)
        rho_k = u_ll[3 + (k - 1) * 5 + 1]
        charge_ratio_ll[k] = rho_k * charge_to_mass[k]
        total_electron_charge += charge_ratio_ll[k]
    end
    charge_ratio_ll ./= total_electron_charge

    # Compute auxiliary variables
    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll, _ = charge_averaged_velocities(u_ll,
                                                                                                              equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr, _ = charge_averaged_velocities(u_rr,
                                                                                                              equations)

    f = zero(MVector{nvariables(equations), eltype(u_ll)})

    if orientation == 1
        # Entries of Powell term for induction equation (already in Trixi's non-conservative form)
        f[1] = v1_plus_ll * B1_rr
        f[2] = v2_plus_ll * B1_rr
        f[3] = v3_plus_ll * B1_rr
        for k in eachcomponent(equations)
            # Compute term 2 (MHD)
            f2 = charge_ratio_ll[k] * (0.5 * mag_norm_rr - B1_rr * B1_rr + pe_rr)
            f3 = charge_ratio_ll[k] * (-B1_rr * B2_rr)
            f4 = charge_ratio_ll[k] * (-B1_rr * B3_rr)
            f5 = vk1_plus_ll[k] * pe_rr

            # Compute term 3 (only needed for NCOMP>1)
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            f5 += (B2_ll * (vk1_minus_rr * B2_rr - vk2_minus_rr * B1_rr) +
                   B3_ll * (vk1_minus_rr * B3_rr - vk3_minus_rr * B1_rr))

            # Compute Powell term (already consistent with Trixi's non-conservative discretization)
            f2 += charge_ratio_ll[k] * B1_ll * B1_rr
            f3 += charge_ratio_ll[k] * B2_ll * B1_rr
            f4 += charge_ratio_ll[k] * B3_ll * B1_rr
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) * B1_rr

            # It's not needed to adjust to Trixi's non-conservative form

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
    else #if orientation == 2
        # Entries of Powell term for induction equation (already in Trixi's non-conservative form)
        f[1] = v1_plus_ll * B2_rr
        f[2] = v2_plus_ll * B2_rr
        f[3] = v3_plus_ll * B2_rr

        for k in eachcomponent(equations)
            # Compute term 2 (MHD)
            f2 = charge_ratio_ll[k] * (-B2_rr * B1_rr)
            f3 = charge_ratio_ll[k] * (-B2_rr * B2_rr + 0.5 * mag_norm_rr + pe_rr)
            f4 = charge_ratio_ll[k] * (-B2_rr * B3_rr)
            f5 = vk2_plus_ll[k] * pe_rr

            # Compute term 3 (only needed for NCOMP>1)
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            f5 += (B1_ll * (vk2_minus_rr * B1_rr - vk1_minus_rr * B2_rr) +
                   B3_ll * (vk2_minus_rr * B3_rr - vk3_minus_rr * B2_rr))

            # Compute Powell term (already consistent with Trixi's non-conservative discretization)
            f2 += charge_ratio_ll[k] * B1_ll * B2_rr
            f3 += charge_ratio_ll[k] * B2_ll * B2_rr
            f4 += charge_ratio_ll[k] * B3_ll * B2_rr
            f5 += (v1_plus_ll * B1_ll + v2_plus_ll * B2_ll + v3_plus_ll * B3_ll) * B2_rr

            # It's not needed to adjust to Trixi's non-conservative form

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
    end

    return SVector(f)
end

"""
flux_ruedaramirez_etal(u_ll, u_rr, orientation, equations::IdealMhdMultiIonEquations2D)

Entropy conserving two-point flux adapted by:
- Rueda-Ramírez et al. (2023)
This flux (together with the MHD non-conservative term) is consistent in the case of one species with the flux of:
- Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations for multi-ion
  [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_ruedaramirez_etal(u_ll, u_rr, orientation::Integer,
                                equations::IdealMhdMultiIonEquations2D)
    @unpack gammas = equations
    # Unpack left and right states to get the magnetic field
    B1_ll, B2_ll, B3_ll = magnetic_field(u_ll, equations)
    B1_rr, B2_rr, B3_rr = magnetic_field(u_rr, equations)

    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll, _ = charge_averaged_velocities(u_ll,
                                                                                                              equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr, _ = charge_averaged_velocities(u_rr,
                                                                                                              equations)

    f = zero(MVector{nvariables(equations), eltype(u_ll)})

    # Compute averages for global variables
    v1_plus_avg = 0.5 * (v1_plus_ll + v1_plus_rr)
    v2_plus_avg = 0.5 * (v2_plus_ll + v2_plus_rr)
    v3_plus_avg = 0.5 * (v3_plus_ll + v3_plus_rr)
    B1_avg = 0.5 * (B1_ll + B1_rr)
    B2_avg = 0.5 * (B2_ll + B2_rr)
    B3_avg = 0.5 * (B3_ll + B3_rr)
    mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
    mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
    mag_norm_avg = 0.5 * (mag_norm_ll + mag_norm_rr)

    if orientation == 1
        # Magnetic field components from f^MHD
        f6 = 0
        f7 = v1_plus_avg * B2_avg - v2_plus_avg * B1_avg
        f8 = v1_plus_avg * B3_avg - v3_plus_avg * B1_avg

        # Start building the flux
        f[1] = f6
        f[2] = f7
        f[3] = f8

        # Iterate over all components
        for k in eachcomponent(equations)
            # Unpack left and right states
            rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = get_component(k, u_ll,
                                                                              equations)
            rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = get_component(k, u_rr,
                                                                              equations)

            v1_ll = rho_v1_ll / rho_ll
            v2_ll = rho_v2_ll / rho_ll
            v3_ll = rho_v3_ll / rho_ll
            v1_rr = rho_v1_rr / rho_rr
            v2_rr = rho_v2_rr / rho_rr
            v3_rr = rho_v3_rr / rho_rr
            vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
            vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2

            p_ll = (gammas[k] - 1) *
                   (rho_e_ll - 0.5 * rho_ll * vel_norm_ll - 0.5 * mag_norm_ll)
            p_rr = (gammas[k] - 1) *
                   (rho_e_rr - 0.5 * rho_rr * vel_norm_rr - 0.5 * mag_norm_rr)
            beta_ll = 0.5 * rho_ll / p_ll
            beta_rr = 0.5 * rho_rr / p_rr
            # for convenience store vk_plus⋅B
            vel_dot_mag_ll = vk1_plus_ll[k] * B1_ll + vk2_plus_ll[k] * B2_ll +
                             vk3_plus_ll[k] * B3_ll
            vel_dot_mag_rr = vk1_plus_rr[k] * B1_rr + vk2_plus_rr[k] * B2_rr +
                             vk3_plus_rr[k] * B3_rr

            # Compute the necessary mean values needed for either direction
            rho_avg = 0.5 * (rho_ll + rho_rr)
            rho_mean = ln_mean(rho_ll, rho_rr)
            beta_mean = ln_mean(beta_ll, beta_rr)
            beta_avg = 0.5 * (beta_ll + beta_rr)
            p_mean = 0.5 * rho_avg / beta_avg
            v1_avg = 0.5 * (v1_ll + v1_rr)
            v2_avg = 0.5 * (v2_ll + v2_rr)
            v3_avg = 0.5 * (v3_ll + v3_rr)
            vel_norm_avg = 0.5 * (vel_norm_ll + vel_norm_rr)
            vel_dot_mag_avg = 0.5 * (vel_dot_mag_ll + vel_dot_mag_rr)
            vk1_plus_avg = 0.5 * (vk1_plus_ll[k] + vk1_plus_rr[k])
            vk2_plus_avg = 0.5 * (vk2_plus_ll[k] + vk2_plus_rr[k])
            vk3_plus_avg = 0.5 * (vk3_plus_ll[k] + vk3_plus_rr[k])
            # v_minus
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            vk1_minus_avg = 0.5 * (vk1_minus_ll + vk1_minus_rr)
            vk2_minus_avg = 0.5 * (vk2_minus_ll + vk2_minus_rr)
            vk3_minus_avg = 0.5 * (vk3_minus_ll + vk3_minus_rr)

            # Ignore orientation since it is always "1" in 1D
            f1 = rho_mean * v1_avg
            f2 = f1 * v1_avg + p_mean
            f3 = f1 * v2_avg
            f4 = f1 * v3_avg

            # total energy flux is complicated and involves the previous eight components
            v1_plus_mag_avg = 0.5 * (vk1_plus_ll[k] * mag_norm_ll +
                               vk1_plus_rr[k] * mag_norm_rr)
            # Euler part
            f5 = f1 * 0.5 * (1 / (gammas[k] - 1) / beta_mean - vel_norm_avg) +
                 f2 * v1_avg + f3 * v2_avg + f4 * v3_avg
            # MHD part
            f5 += (f6 * B1_avg + f7 * B2_avg + f8 * B3_avg - 0.5 * v1_plus_mag_avg +
                   B1_avg * vel_dot_mag_avg                                               # Same terms as in Derigs (but with v_plus)
                   + 0.5 * vk1_plus_avg * mag_norm_avg -
                   vk1_plus_avg * B1_avg * B1_avg - vk2_plus_avg * B1_avg * B2_avg -
                   vk3_plus_avg * B1_avg * B3_avg   # Additional terms coming from the MHD non-conservative term (momentum eqs)
                   -
                   B2_avg * (vk1_minus_avg * B2_avg - vk2_minus_avg * B1_avg) -
                   B3_avg * (vk1_minus_avg * B3_avg - vk3_minus_avg * B1_avg))             # Terms coming from the non-conservative term 3 (induction equation!)

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
    else #if orientation == 2
        # Magnetic field components from f^MHD
        f6 = v2_plus_avg * B1_avg - v1_plus_avg * B2_avg
        f7 = 0
        f8 = v2_plus_avg * B3_avg - v3_plus_avg * B2_avg

        # Start building the flux
        f[1] = f6
        f[2] = f7
        f[3] = f8

        # Iterate over all components
        for k in eachcomponent(equations)
            # Unpack left and right states
            rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll = get_component(k, u_ll,
                                                                              equations)
            rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr = get_component(k, u_rr,
                                                                              equations)

            v1_ll = rho_v1_ll / rho_ll
            v2_ll = rho_v2_ll / rho_ll
            v3_ll = rho_v3_ll / rho_ll
            v1_rr = rho_v1_rr / rho_rr
            v2_rr = rho_v2_rr / rho_rr
            v3_rr = rho_v3_rr / rho_rr
            vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
            vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2

            p_ll = (gammas[k] - 1) *
                   (rho_e_ll - 0.5 * rho_ll * vel_norm_ll - 0.5 * mag_norm_ll)
            p_rr = (gammas[k] - 1) *
                   (rho_e_rr - 0.5 * rho_rr * vel_norm_rr - 0.5 * mag_norm_rr)
            beta_ll = 0.5 * rho_ll / p_ll
            beta_rr = 0.5 * rho_rr / p_rr
            # for convenience store vk_plus⋅B
            vel_dot_mag_ll = vk1_plus_ll[k] * B1_ll + vk2_plus_ll[k] * B2_ll +
                             vk3_plus_ll[k] * B3_ll
            vel_dot_mag_rr = vk1_plus_rr[k] * B1_rr + vk2_plus_rr[k] * B2_rr +
                             vk3_plus_rr[k] * B3_rr

            # Compute the necessary mean values needed for either direction
            rho_avg = 0.5 * (rho_ll + rho_rr)
            rho_mean = ln_mean(rho_ll, rho_rr)
            beta_mean = ln_mean(beta_ll, beta_rr)
            beta_avg = 0.5 * (beta_ll + beta_rr)
            p_mean = 0.5 * rho_avg / beta_avg
            v1_avg = 0.5 * (v1_ll + v1_rr)
            v2_avg = 0.5 * (v2_ll + v2_rr)
            v3_avg = 0.5 * (v3_ll + v3_rr)
            vel_norm_avg = 0.5 * (vel_norm_ll + vel_norm_rr)
            vel_dot_mag_avg = 0.5 * (vel_dot_mag_ll + vel_dot_mag_rr)
            vk1_plus_avg = 0.5 * (vk1_plus_ll[k] + vk1_plus_rr[k])
            vk2_plus_avg = 0.5 * (vk2_plus_ll[k] + vk2_plus_rr[k])
            vk3_plus_avg = 0.5 * (vk3_plus_ll[k] + vk3_plus_rr[k])
            # v_minus
            vk1_minus_ll = v1_plus_ll - vk1_plus_ll[k]
            vk2_minus_ll = v2_plus_ll - vk2_plus_ll[k]
            vk3_minus_ll = v3_plus_ll - vk3_plus_ll[k]
            vk1_minus_rr = v1_plus_rr - vk1_plus_rr[k]
            vk2_minus_rr = v2_plus_rr - vk2_plus_rr[k]
            vk3_minus_rr = v3_plus_rr - vk3_plus_rr[k]
            vk1_minus_avg = 0.5 * (vk1_minus_ll + vk1_minus_rr)
            vk2_minus_avg = 0.5 * (vk2_minus_ll + vk2_minus_rr)
            vk3_minus_avg = 0.5 * (vk3_minus_ll + vk3_minus_rr)

            # Ignore orientation since it is always "1" in 1D
            f1 = rho_mean * v2_avg
            f2 = f1 * v1_avg
            f3 = f1 * v2_avg + p_mean
            f4 = f1 * v3_avg

            # total energy flux is complicated and involves the previous eight components
            v2_plus_mag_avg = 0.5 * (vk2_plus_ll[k] * mag_norm_ll +
                               vk2_plus_rr[k] * mag_norm_rr)
            # Euler part
            f5 = f1 * 0.5 * (1 / (gammas[k] - 1) / beta_mean - vel_norm_avg) +
                 f2 * v1_avg + f3 * v2_avg + f4 * v3_avg
            # MHD part
            f5 += (f6 * B1_avg + f7 * B2_avg + f8 * B3_avg - 0.5 * v2_plus_mag_avg +
                   B2_avg * vel_dot_mag_avg                                               # Same terms as in Derigs (but with v_plus)
                   + 0.5 * vk2_plus_avg * mag_norm_avg -
                   vk1_plus_avg * B2_avg * B1_avg - vk2_plus_avg * B2_avg * B2_avg -
                   vk3_plus_avg * B2_avg * B3_avg   # Additional terms coming from the MHD non-conservative term (momentum eqs)
                   -
                   B1_avg * (vk2_minus_avg * B1_avg - vk1_minus_avg * B2_avg) -
                   B3_avg * (vk2_minus_avg * B3_avg - vk3_minus_avg * B2_avg))             # Terms coming from the non-conservative term 3 (induction equation!)

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
    end

    return SVector(f)
end

"""
# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
  !!!ATTENTION: This routine is provisional. TODO: Update with the right max_abs_speed
"""
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::IdealMhdMultiIonEquations2D)
    # Calculate fast magnetoacoustic wave speeds
    # left
    cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
    # right
    cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

    # Calculate velocities
    v_ll = zero(eltype(u_ll))
    v_rr = zero(eltype(u_rr))
    if orientation == 1
        for k in eachcomponent(equations)
            rho, rho_v1, _ = get_component(k, u_ll, equations)
            v_ll = max(v_ll, abs(rho_v1 / rho))
            rho, rho_v1, _ = get_component(k, u_rr, equations)
            v_rr = max(v_rr, abs(rho_v1 / rho))
        end
    else #if orientation == 2
        for k in eachcomponent(equations)
            rho, rho_v1, rho_v2, _ = get_component(k, u_ll, equations)
            v_ll = max(v_ll, abs(rho_v2 / rho))
            rho, rho_v1, rho_v2, _ = get_component(k, u_rr, equations)
            v_rr = max(v_rr, abs(rho_v2 / rho))
        end
    end

    λ_max = max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
end

@inline function max_abs_speeds(u, equations::IdealMhdMultiIonEquations2D)
    v1 = zero(real(equations))
    v2 = zero(real(equations))
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, _ = get_component(k, u, equations)
        v1 = max(v1, abs(rho_v1 / rho))
        v2 = max(v2, abs(rho_v2 / rho))
    end

    cf_x_direction = calc_fast_wavespeed(u, 1, equations)
    cf_y_direction = calc_fast_wavespeed(u, 2, equations)

    return (abs(v1) + cf_x_direction, abs(v2) + cf_y_direction)
end

"""
Convert conservative variables to primitive
"""
function cons2prim(u, equations::IdealMhdMultiIonEquations2D)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(u, equations)

    prim = zero(MVector{nvariables(equations), eltype(u)})
    prim[1] = B1
    prim[2] = B2
    prim[3] = B3
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        srho = 1 / rho
        v1 = srho * rho_v1
        v2 = srho * rho_v2
        v3 = srho * rho_v3

        p = (gammas[k] - 1) * (rho_e -
             0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3
              + B1 * B1 + B2 * B2 + B3 * B3))

        set_component!(prim, k, rho, v1, v2, v3, p, equations)
    end

    return SVector(prim)
end

"""
Convert conservative variables to entropy
"""
@inline function cons2entropy(u, equations::IdealMhdMultiIonEquations2D)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(u, equations)

    prim = cons2prim(u, equations)
    entropy = zero(MVector{nvariables(equations), eltype(u)})
    rho_p_plus = zero(real(equations))
    for k in eachcomponent(equations)
        rho, v1, v2, v3, p = get_component(k, prim, equations)
        s = log(p) - gammas[k] * log(rho)
        rho_p = rho / p
        w1 = (gammas[k] - s) / (gammas[k] - 1) - 0.5 * rho_p * (v1^2 + v2^2 + v3^2)
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

    return SVector(entropy)
end

"""
Convert primitive to conservative variables
"""
@inline function prim2cons(prim, equations::IdealMhdMultiIonEquations2D)
    @unpack gammas = equations
    B1, B2, B3 = magnetic_field(prim, equations)

    cons = zero(MVector{nvariables(equations), eltype(prim)})
    cons[1] = B1
    cons[2] = B2
    cons[3] = B3
    for k in eachcomponent(equations)
        rho, v1, v2, v3, p = get_component(k, prim, equations)
        rho_v1 = rho * v1
        rho_v2 = rho * v2
        rho_v3 = rho * v3

        rho_e = p / (gammas[k] - 1.0) +
                0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) +
                0.5 * (B1^2 + B2^2 + B3^2)

        set_component!(cons, k, rho, rho_v1, rho_v2, rho_v3, rho_e, equations)
    end

    return SVector(cons)
end

"""
Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
  !!! ATTENTION: This routine is provisional.. Change once the fastest wave speed is known!!
"""
@inline function calc_fast_wavespeed(cons, orientation::Integer,
                                     equations::IdealMhdMultiIonEquations2D)
    B1, B2, B3 = magnetic_field(cons, equations)

    c_f = zero(real(equations))
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, cons, equations)

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        gamma = equations.gammas[k]
        p = (gamma - 1) * (rho_e - 0.5 * rho * v_mag^2 - 0.5 * (B1^2 + B2^2 + B3^2))
        a_square = gamma * p / rho
        sqrt_rho = sqrt(rho)

        b1 = B1 / sqrt_rho
        b2 = B2 / sqrt_rho
        b3 = B3 / sqrt_rho
        b_square = b1^2 + b2^2 + b3^2

        if orientation == 1
            c_f = max(c_f,
                      sqrt(0.5 * (a_square + b_square) +
                           0.5 * sqrt((a_square + b_square)^2 - 4.0 * a_square * b1^2)))
        else #if orientation == 2
            c_f = max(c_f,
                      sqrt(0.5 * (a_square + b_square) +
                           0.5 * sqrt((a_square + b_square)^2 - 4.0 * a_square * b2^2)))
        end
    end

    return c_f
end

"""
Routine to compute the Charge-averaged velocities:
* v*_plus: Charge-averaged velocity
* vk*_plus: Contribution of each species to the charge-averaged velocity
"""
@inline function charge_averaged_velocities(u, equations::IdealMhdMultiIonEquations2D)
    total_electron_charge = zero(real(equations))

    vk1_plus = zero(MVector{ncomponents(equations), eltype(u)})
    vk2_plus = zero(MVector{ncomponents(equations), eltype(u)})
    vk3_plus = zero(MVector{ncomponents(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, _ = get_component(k, u,
                                                       equations::IdealMhdMultiIonEquations2D)

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
           SVector(vk3_plus), total_electron_charge
end

"""
Get the flow variables of component k
"""
@inline function get_component(k, u, equations::IdealMhdMultiIonEquations2D)
    # The first 3 entries of u contain the magnetic field. The following entries contain the density, momentum (3 entries), and energy of each component.
    return SVector(u[3 + (k - 1) * 5 + 1],
                   u[3 + (k - 1) * 5 + 2],
                   u[3 + (k - 1) * 5 + 3],
                   u[3 + (k - 1) * 5 + 4],
                   u[3 + (k - 1) * 5 + 5])
end

"""
Set the flow variables of component k
"""
@inline function set_component!(u, k, u1, u2, u3, u4, u5,
                                equations::IdealMhdMultiIonEquations2D)
    # The first 3 entries of u contain the magnetic field. The following entries contain the density, momentum (3 entries), and energy of each component.
    u[3 + (k - 1) * 5 + 1] = u1
    u[3 + (k - 1) * 5 + 2] = u2
    u[3 + (k - 1) * 5 + 3] = u3
    u[3 + (k - 1) * 5 + 4] = u4
    u[3 + (k - 1) * 5 + 5] = u5

    return u
end

magnetic_field(u, equations::IdealMhdMultiIonEquations2D) = SVector(u[1], u[2], u[3])

@inline function density(u, equations::IdealMhdMultiIonEquations2D)
    rho = zero(real(equations))
    for k in eachcomponent(equations)
        rho += u[3 + (k - 1) * 5 + 1]
    end
    return rho
end

@inline function pressure(u, equations::IdealMhdMultiIonEquations2D)
    B1, B2, B3, _ = u
    p = zero(MVector{ncomponents(equations), real(equations)})
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        gamma = equations.gammas[k]
        p[k] = (gamma - 1) * (rho_e - 0.5 * rho * v_mag^2 - 0.5 * (B1^2 + B2^2 + B3^2))
    end
    return SVector{ncomponents(equations), real(equations)}(p)
end

"""
Computes the sum of the densities times the sum of the pressures
"""
@inline function density_pressure(u, equations::IdealMhdMultiIonEquations2D)
    B1, B2, B3 = magnetic_field(u, equations)
    rho_total = zero(real(equations))
    p_total = zero(real(equations))
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        gamma = equations.gammas[k]

        p = (gamma - 1) * (rho_e - 0.5 * rho * v_mag^2 - 0.5 * (B1^2 + B2^2 + B3^2))

        rho_total += rho
        p_total += p
    end
    return rho_total * p_total
end
end # @muladd
