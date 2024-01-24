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
                                           ElectronPressure} <:
               AbstractIdealMhdMultiIonEquations{2, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT} # Heat capacity ratios
    charge_to_mass::SVector{NCOMP, RealT} # Charge to mass ratios
    electron_pressure::ElectronPressure       # Function to compute the electron pressure
    c_h::RealT                 # GLM cleaning speed
    function IdealMhdMultiIonEquations2D{NVARS, NCOMP, RealT,
                                         ElectronPressure}(gammas
                                                           ::SVector{NCOMP, RealT},
                                                           charge_to_mass
                                                           ::SVector{NCOMP, RealT},
                                                           electron_pressure
                                                           ::ElectronPressure,
                                                           c_h::RealT) where
             {NVARS, NCOMP, RealT <: Real, ElectronPressure}
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_to_mass` have to be filled with at least one value"))

        new(gammas, charge_to_mass, electron_pressure, c_h)
    end
end

function IdealMhdMultiIonEquations2D(; gammas, charge_to_mass,
                                     electron_pressure = electron_pressure_zero,
                                     initial_c_h = convert(eltype(gammas), NaN))
    _gammas = promote(gammas...)
    _charge_to_mass = promote(charge_to_mass...)
    RealT = promote_type(eltype(_gammas), eltype(_charge_to_mass))

    NVARS = length(_gammas) * 5 + 4
    NCOMP = length(_gammas)

    __gammas = SVector(map(RealT, _gammas))
    __charge_to_mass = SVector(map(RealT, _charge_to_mass))

    return IdealMhdMultiIonEquations2D{NVARS, NCOMP, RealT, typeof(electron_pressure)}(__gammas,
                                                                                       __charge_to_mass,
                                                                                       electron_pressure,
                                                                                       initial_c_h)
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
    cons = (cons..., "psi")

    return cons
end

function varnames(::typeof(cons2prim), equations::IdealMhdMultiIonEquations2D)
    prim = ("B1", "B2", "B3")
    for i in eachcomponent(equations)
        prim = (prim...,
                tuple("rho_" * string(i), "v1_" * string(i), "v2_" * string(i),
                      "v3_" * string(i), "p_" * string(i))...)
    end
    prim = (prim..., "psi")

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
    psi = divergence_cleaning_field(u, equations)
    
    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus = charge_averaged_velocities(u,
                                                                                         equations)

    mag_en = 0.5 * (B1^2 + B2^2 + B3^2)
    div_clean_energy = 0.5 * psi^2

    f = zero(MVector{nvariables(equations), eltype(u)})

    if orientation == 1
        f[1] = equations.c_h * psi
        f[2] = v1_plus * B2 - v2_plus * B1
        f[3] = v1_plus * B3 - v3_plus * B1

        for k in eachcomponent(equations)
            rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
            v1 = rho_v1 / rho
            v2 = rho_v2 / rho
            v3 = rho_v3 / rho
            kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)

            gamma = equations.gammas[k]
            p = (gamma - 1) * (rho_e - kin_en - mag_en - div_clean_energy)

            f1 = rho_v1
            f2 = rho_v1 * v1 + p
            f3 = rho_v1 * v2
            f4 = rho_v1 * v3
            f5 = (kin_en + gamma * p / (gamma - 1)) * v1 + 2 * mag_en * vk1_plus[k] -
                 B1 * (vk1_plus[k] * B1 + vk2_plus[k] * B2 + vk3_plus[k] * B3) + equations.c_h * psi * B1

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
        f[end] = equations.c_h * B1
    else #if orientation == 2
        f[1] = v2_plus * B1 - v1_plus * B2
        f[2] = equations.c_h * psi
        f[3] = v2_plus * B3 - v3_plus * B2

        for k in eachcomponent(equations)
            rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
            v1 = rho_v1 / rho
            v2 = rho_v2 / rho
            v3 = rho_v3 / rho
            kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)

            gamma = equations.gammas[k]
            p = (gamma - 1) * (rho_e - kin_en - mag_en - div_clean_energy)

            f1 = rho_v2
            f2 = rho_v2 * v1
            f3 = rho_v2 * v2 + p
            f4 = rho_v2 * v3
            f5 = (kin_en + gamma * p / (gamma - 1)) * v2 + 2 * mag_en * vk2_plus[k] -
                 B2 * (vk1_plus[k] * B1 + vk2_plus[k] * B2 + vk3_plus[k] * B3) + equations.c_h * psi * B2

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
        f[end] = equations.c_h * B2
    end

    return SVector(f)
end

"""
Standard source terms of the multi-ion MHD equations
"""
function source_terms_standard(u, x, t, equations::IdealMhdMultiIonEquations2D)
    @unpack charge_to_mass = equations
    B1, B2, B3 = magnetic_field(u, equations)
    v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus = charge_averaged_velocities(u,
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
  electron_pressure_zero(u, equations::IdealMhdMultiIonEquations2D)
Returns the value of zero for the electron pressure. Consistent with the single-fluid MHD equations.
"""
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
    psi_ll = divergence_cleaning_field(u_ll, equations)
    psi_rr = divergence_cleaning_field(u_rr, equations)

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
    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll = charge_averaged_velocities(u_ll,
                                                                                                           equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr = charge_averaged_velocities(u_rr,
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

            # Compute GLM term for the energy
            f5 += v1_plus_ll * psi_ll * psi_rr

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
        # Compute GLM term for psi
        f[end] = v1_plus_ll * psi_rr

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

            # Compute GLM term for the energy
            f5 += v2_plus_ll * psi_ll * psi_rr

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
        # Compute GLM term for psi
        f[end] = v2_plus_ll * psi_rr
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
    psi_ll = divergence_cleaning_field(u_ll, equations)
    psi_rr = divergence_cleaning_field(u_rr, equations)

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
    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll = charge_averaged_velocities(u_ll,
                                                                                                           equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr = charge_averaged_velocities(u_rr,
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

            # Compute GLM term for the energy
            f5 += v1_plus_ll * psi_ll * psi_rr

            # It's not needed to adjust to Trixi's non-conservative form

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
        # Compute GLM term for psi
        f[end] = v1_plus_ll * psi_rr

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

            # Compute GLM term for the energy
            f5 += v2_plus_ll * psi_ll * psi_rr

            # It's not needed to adjust to Trixi's non-conservative form

            # Append to the flux vector
            set_component!(f, k, 0, f2, f3, f4, f5, equations)
        end
        # Compute GLM term for psi
        f[end] = v2_plus_ll * psi_rr
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
    psi_ll = divergence_cleaning_field(u_ll, equations)
    psi_rr = divergence_cleaning_field(u_rr, equations)

    v1_plus_ll, v2_plus_ll, v3_plus_ll, vk1_plus_ll, vk2_plus_ll, vk3_plus_ll = charge_averaged_velocities(u_ll,
                                                                                                           equations)
    v1_plus_rr, v2_plus_rr, v3_plus_rr, vk1_plus_rr, vk2_plus_rr, vk3_plus_rr = charge_averaged_velocities(u_rr,
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
    psi_avg = 0.5 * (psi_ll + psi_rr)

    if orientation == 1
        psi_B1_avg = 0.5 * (B1_ll * psi_ll + B1_rr * psi_rr)

        # Magnetic field components from f^MHD
        f6 = equations.c_h * psi_avg
        f7 = v1_plus_avg * B2_avg - v2_plus_avg * B1_avg
        f8 = v1_plus_avg * B3_avg - v3_plus_avg * B1_avg
        f9 = equations.c_h * B1_avg

        # Start building the flux
        f[1] = f6
        f[2] = f7
        f[3] = f8
        f[end] = f9

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
                   (rho_e_ll - 0.5 * rho_ll * vel_norm_ll - 0.5 * mag_norm_ll - 0.5 * psi_ll^2)
            p_rr = (gammas[k] - 1) *
                   (rho_e_rr - 0.5 * rho_rr * vel_norm_rr - 0.5 * mag_norm_rr - 0.5 * psi_rr^2)
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
                   + f9 * psi_avg - equations.c_h * psi_B1_avg # GLM term
                   + 0.5 * vk1_plus_avg * mag_norm_avg -
                   vk1_plus_avg * B1_avg * B1_avg - vk2_plus_avg * B1_avg * B2_avg -
                   vk3_plus_avg * B1_avg * B3_avg   # Additional terms coming from the MHD non-conservative term (momentum eqs)
                   -
                   B2_avg * (vk1_minus_avg * B2_avg - vk2_minus_avg * B1_avg) -
                   B3_avg * (vk1_minus_avg * B3_avg - vk3_minus_avg * B1_avg))             # Terms coming from the non-conservative term 3 (induction equation!)

            set_component!(f, k, f1, f2, f3, f4, f5, equations)
        end
    else #if orientation == 2
        psi_B2_avg = 0.5 * (B2_ll * psi_ll + B2_rr * psi_rr)

        # Magnetic field components from f^MHD
        f6 = v2_plus_avg * B1_avg - v1_plus_avg * B2_avg
        f7 = equations.c_h * psi_avg
        f8 = v2_plus_avg * B3_avg - v3_plus_avg * B2_avg
        f9 = equations.c_h * B2_avg
        
        # Start building the flux
        f[1] = f6
        f[2] = f7
        f[3] = f8
        f[end] = f9

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
                   (rho_e_ll - 0.5 * rho_ll * vel_norm_ll - 0.5 * mag_norm_ll - 0.5 * psi_ll^2)
            p_rr = (gammas[k] - 1) *
                   (rho_e_rr - 0.5 * rho_rr * vel_norm_rr - 0.5 * mag_norm_rr - 0.5 * psi_rr^2)
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
                   + f9 * psi_avg - equations.c_h * psi_B2_avg # GLM term
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
    psi = divergence_cleaning_field(u, equations)

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
              + B1 * B1 + B2 * B2 + B3 * B3
              + psi * psi))

        set_component!(prim, k, rho, v1, v2, v3, p, equations)
    end
    prim[end] = psi

    return SVector(prim)
end

"""
Convert conservative variables to entropy
"""
@inline function cons2entropy(u, equations::IdealMhdMultiIonEquations2D)
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
    entropy[end] = rho_p_plus * psi

    return SVector(entropy)
end

"""
Convert primitive to conservative variables
"""
@inline function prim2cons(prim, equations::IdealMhdMultiIonEquations2D)
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

        rho_e = p / (gammas[k] - 1.0) +
                0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) +
                0.5 * (B1^2 + B2^2 + B3^2) +
                0.5 * psi^2

        set_component!(cons, k, rho, rho_v1, rho_v2, rho_v3, rho_e, equations)
    end
    cons[end] = psi

    return SVector(cons)
end

"""
Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
  !!! ATTENTION: This routine is provisional.. Change once the fastest wave speed is known!!
"""
@inline function calc_fast_wavespeed(cons, orientation::Integer,
                                     equations::IdealMhdMultiIonEquations2D)
    B1, B2, B3 = magnetic_field(cons, equations)
    psi = divergence_cleaning_field(cons, equations)

    c_f = zero(real(equations))
    for k in eachcomponent(equations)
        rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, cons, equations)

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        v_mag = sqrt(v1^2 + v2^2 + v3^2)
        gamma = equations.gammas[k]
        p = (gamma - 1) * (rho_e - 0.5 * rho * v_mag^2 - 0.5 * (B1^2 + B2^2 + B3^2) - 0.5 * psi^2)
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
           SVector(vk3_plus)
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
divergence_cleaning_field(u, equations::IdealMhdMultiIonEquations2D) = u[end]

@inline function density(u, equations::IdealMhdMultiIonEquations2D)
    rho = zero(real(equations))
    for k in eachcomponent(equations)
        rho += u[3 + (k - 1) * 5 + 1]
    end
    return rho
end

"""
Computes the sum of the densities times the sum of the pressures
"""
@inline function density_pressure(u, equations::IdealMhdMultiIonEquations2D)
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

        p = (gamma - 1) * (rho_e - 0.5 * rho * v_mag^2 - 0.5 * (B1^2 + B2^2 + B3^2) - 0.5 * psi^2)

        rho_total += rho
        p_total += p
    end
    return rho_total * p_total
end

"""
    DissipationEntropyStableLump(max_abs_speed=max_abs_speed_naive)

Create a local Lax-Friedrichs dissipation operator where the maximum absolute wave speed
is estimated as
`max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)`,
defaulting to [`max_abs_speed_naive`](@ref).
"""
struct DissipationEntropyStableLump{MaxAbsSpeed}
    max_abs_speed::MaxAbsSpeed
end

DissipationEntropyStableLump() = DissipationEntropyStableLump(max_abs_speed_naive)

@inline function (dissipation::DissipationEntropyStableLump)(u_ll, u_rr,
                                                              orientation_or_normal_direction,
                                                              equations::IdealMhdMultiIonEquations2D)
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
    B1_avg = 0.5 * (B1_ll + B1_rr)
    B2_avg = 0.5 * (B2_ll + B2_rr)
    B3_avg = 0.5 * (B3_ll + B3_rr)
    psi_avg = 0.5 * (psi_ll + psi_rr)

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
        beta_ll = 0.5 * rho_ll / p_ll
        beta_rr = 0.5 * rho_rr / p_rr
        vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
        vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2

        # Mean variables
        rho_ln = ln_mean(rho_ll, rho_rr)
        beta_ln = ln_mean(beta_ll, beta_rr)
        rho_avg = 0.5 * (rho_ll + rho_rr)
        v1_avg = 0.5 * (v1_ll + v1_rr)
        v2_avg = 0.5 * (v2_ll + v2_rr)
        v3_avg = 0.5 * (v3_ll + v3_rr)
        beta_avg = 0.5 * (beta_ll + beta_rr)
        tau = 1 / (beta_ll + beta_rr) 
        p_mean = 0.5 * rho_avg / beta_avg
        p_star = 0.5 * rho_ln / beta_ln
        vel_norm_avg = 0.5 * (vel_norm_ll + vel_norm_rr)
        vel_avg_norm = v1_avg^2 + v2_avg^2 + v3_avg^2
        E_bar = p_star / (gammas[k] - 1) +  0.5 * rho_ln * (2 * vel_avg_norm - vel_norm_avg)
        
        h11 = rho_ln
        h22 = rho_ln * v1_avg^2 + p_mean
        h33 = rho_ln * v2_avg^2 + p_mean
        h44 = rho_ln * v3_avg^2 + p_mean
        h55 = ((p_star^2 / (gammas[k] - 1) + E_bar * E_bar) / rho_ln 
              + vel_avg_norm * p_mean 
              + tau * ( B1_avg^2 + B2_avg^2 + B3_avg^2 + psi_avg^2 ))

        d1 = -0.5 * λ * h11 * (w1_rr - w1_ll)
        d2 = -0.5 * λ * h22 * (w2_rr - w2_ll)
        d3 = -0.5 * λ * h33 * (w3_rr - w3_ll)
        d4 = -0.5 * λ * h44 * (w4_rr - w4_ll)
        d5 = -0.5 * λ * h55 * (w5_rr - w5_ll)

        beta_plus_ll += beta_ll
        beta_plus_rr += beta_rr

        set_component!(dissipation, k, d1, d2, d3, d4, d5, equations)
    end

    # Set the magnetic field and psi terms
    h_B_psi = 1 / (beta_plus_ll + beta_plus_rr) 
    dissipation[1] = -0.5 * λ * h_B_psi * (w_rr[1] - w_ll[1])
    dissipation[2] = -0.5 * λ * h_B_psi * (w_rr[2] - w_ll[2])
    dissipation[3] = -0.5 * λ * h_B_psi * (w_rr[3] - w_ll[3])
    dissipation[end] = -0.5 * λ * h_B_psi * (w_rr[end] - w_ll[end])

    return dissipation
end

function Base.show(io::IO, d::DissipationEntropyStableLump)
    print(io, "DissipationEntropyStableLump(", d.max_abs_speed, ")")
end

"""
DissipationEntropyStable(max_abs_speed=max_abs_speed_naive)

Create a local Lax-Friedrichs dissipation operator where the maximum absolute wave speed
is estimated as
`max_abs_speed(u_ll, u_rr, orientation_or_normal_direction, equations)`,
defaulting to [`max_abs_speed_naive`](@ref).
"""
struct DissipationEntropyStable{MaxAbsSpeed}
    max_abs_speed::MaxAbsSpeed
end

DissipationEntropyStable() = DissipationEntropyStable(max_abs_speed_naive)

@inline function (dissipation::DissipationEntropyStable)(u_ll, u_rr,
                                                              orientation_or_normal_direction,
                                                              equations::IdealMhdMultiIonEquations2D)
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
    B1_avg = 0.5 * (B1_ll + B1_rr)
    B2_avg = 0.5 * (B2_ll + B2_rr)
    B3_avg = 0.5 * (B3_ll + B3_rr)
    psi_avg = 0.5 * (psi_ll + psi_rr)

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
        beta_ll = 0.5 * rho_ll / p_ll
        beta_rr = 0.5 * rho_rr / p_rr
        vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
        vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2

        # Mean variables
        rho_ln = ln_mean(rho_ll, rho_rr)
        beta_ln = ln_mean(beta_ll, beta_rr)
        rho_avg = 0.5 * (rho_ll + rho_rr)
        v1_avg = 0.5 * (v1_ll + v1_rr)
        v2_avg = 0.5 * (v2_ll + v2_rr)
        v3_avg = 0.5 * (v3_ll + v3_rr)
        beta_avg = 0.5 * (beta_ll + beta_rr)
        tau = 1 / (beta_ll + beta_rr) 
        p_mean = 0.5 * rho_avg / beta_avg
        p_star = 0.5 * rho_ln / beta_ln
        vel_norm_avg = 0.5 * (vel_norm_ll + vel_norm_rr)
        vel_avg_norm = v1_avg^2 + v2_avg^2 + v3_avg^2
        E_bar = p_star / (gammas[k] - 1) +  0.5 * rho_ln * (2 * vel_avg_norm - vel_norm_avg)
        
        h11 = rho_ln
        h12 = rho_ln * v1_avg
        h13 = rho_ln * v2_avg
        h14 = rho_ln * v3_avg
        h15 = E_bar
        d1 = -0.5 * λ * (h11 * (w1_rr - w1_ll) +
                         h12 * (w2_rr - w2_ll) +
                         h13 * (w3_rr - w3_ll) +
                         h14 * (w4_rr - w4_ll) +
                         h15 * (w5_rr - w5_ll) )

        h21 = h12
        h22 = rho_ln * v1_avg^2 + p_mean
        h23 = h21 * v2_avg
        h24 = h21 * v3_avg
        h25 = (E_bar + p_mean) * v1_avg
        d2 = -0.5 * λ * (h21 * (w1_rr - w1_ll) +
                         h22 * (w2_rr - w2_ll) +
                         h23 * (w3_rr - w3_ll) +
                         h24 * (w4_rr - w4_ll) +
                         h25 * (w5_rr - w5_ll) )

        h31 = h13
        h32 = h23
        h33 = rho_ln * v2_avg^2 + p_mean
        h34 = h31 * v3_avg
        h35 = (E_bar + p_mean) * v2_avg
        d3 = -0.5 * λ * (h31 * (w1_rr - w1_ll) +
                         h32 * (w2_rr - w2_ll) +
                         h33 * (w3_rr - w3_ll) +
                         h34 * (w4_rr - w4_ll) +
                         h35 * (w5_rr - w5_ll) )

        h41 = h14
        h42 = h24
        h43 = h34
        h44 = rho_ln * v3_avg^2 + p_mean
        h45 = (E_bar + p_mean) * v3_avg
        d4 = -0.5 * λ * (h41 * (w1_rr - w1_ll) +
                         h42 * (w2_rr - w2_ll) +
                         h43 * (w3_rr - w3_ll) +
                         h44 * (w4_rr - w4_ll) +
                         h45 * (w5_rr - w5_ll) )

        h51 = h15
        h52 = h25
        h53 = h35
        h54 = h45
        h55 = ((p_star^2 / (gammas[k] - 1) + E_bar * E_bar) / rho_ln 
              + vel_avg_norm * p_mean 
              + tau * ( B1_avg^2 + B2_avg^2 + B3_avg^2 + psi_avg^2 ))
        d5 = -0.5 * λ * (h51 * (w1_rr - w1_ll) +
                         h52 * (w2_rr - w2_ll) +
                         h53 * (w3_rr - w3_ll) +
                         h54 * (w4_rr - w4_ll) +
                         h55 * (w5_rr - w5_ll) )

        beta_plus_ll += beta_ll
        beta_plus_rr += beta_rr

        set_component!(dissipation, k, d1, d2, d3, d4, d5, equations)
    end

    # Set the magnetic field and psi terms
    h_B_psi = 1 / (beta_plus_ll + beta_plus_rr) 

    # diagonal entries
    dissipation[1] = -0.5 * λ * h_B_psi * (w_rr[1] - w_ll[1])
    dissipation[2] = -0.5 * λ * h_B_psi * (w_rr[2] - w_ll[2])
    dissipation[3] = -0.5 * λ * h_B_psi * (w_rr[3] - w_ll[3])
    dissipation[end] = -0.5 * λ * h_B_psi * (w_rr[end] - w_ll[end])
    # Off-diagonal entries
    for k in eachcomponent(equations)
        _, _, _, _, w5_ll = get_component(k, w_ll, equations)
        _, _, _, _, w5_rr = get_component(k, w_rr, equations)

        dissipation[1] -= 0.5 * λ * h_B_psi * B1_avg * (w5_rr - w5_ll)
        dissipation[2] -= 0.5 * λ * h_B_psi * B2_avg * (w5_rr - w5_ll)
        dissipation[3] -= 0.5 * λ * h_B_psi * B3_avg * (w5_rr - w5_ll)
        dissipation[end] -= 0.5 * λ * h_B_psi * psi_avg * (w5_rr - w5_ll)

        ind_E = 3 + (k - 1) * 5 + 5
        dissipation[ind_E] -= 0.5 * λ * h_B_psi * B1_avg * (w_rr[1] - w_ll[1])
        dissipation[ind_E] -= 0.5 * λ * h_B_psi * B2_avg * (w_rr[2] - w_ll[2])
        dissipation[ind_E] -= 0.5 * λ * h_B_psi * B3_avg * (w_rr[3] - w_ll[3])
        dissipation[ind_E] -= 0.5 * λ * h_B_psi * psi_avg * (w_rr[end] - w_ll[end])
    end

    return dissipation
end

function Base.show(io::IO, d::DissipationEntropyStable)
    print(io, "DissipationEntropyStable(", d.max_abs_speed, ")")
end

end # @muladd
