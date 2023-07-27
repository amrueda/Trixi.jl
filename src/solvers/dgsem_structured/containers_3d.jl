# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize data structures in element container
function init_elements!(elements, mesh::StructuredMesh{3}, basis::LobattoLegendreBasis)
    @unpack node_coordinates, left_neighbors,
    jacobian_matrix, contravariant_vectors, inverse_jacobian = elements

    linear_indices = LinearIndices(size(mesh))

    node_coordinates_comp = zeros(3, nnodes(basis))

    # Calculate node coordinates, Jacobian matrix, and inverse Jacobian determinant
    for cell_z in 1:size(mesh, 3), cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
        element = linear_indices[cell_x, cell_y, cell_z]

        calc_node_coordinates!(node_coordinates, element, cell_x, cell_y, cell_z,
                               mesh.mapping, mesh, basis)

        if mesh.exact_jacobian || mesh.mimetic
            calc_node_coordinates_computational!(node_coordinates_comp, cell_x, cell_y, cell_z, mesh, basis)
        end
        
        if mesh.exact_jacobian
            calc_jacobian_matrix_exact!(jacobian_matrix, element, node_coordinates, basis, node_coordinates_comp)
        else
            calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)
        end

        if mesh.mimetic
            calc_contravariant_vectors_mimetic!(contravariant_vectors, element, jacobian_matrix,
                                                 node_coordinates, basis, node_coordinates_comp)
        else
            calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix,
                                                 node_coordinates, basis)
        end

        calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix, basis)
    end

    initialize_left_neighbor_connectivity!(left_neighbors, mesh, linear_indices)

    return nothing
end

# Calculate physical coordinates to which every node of the reference element is mapped
# `mesh.mapping` is passed as an additional argument for type stability (function barrier)
function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y, cell_z,
                                mapping, mesh::StructuredMesh{3},
                                basis::LobattoLegendreBasis)
    @unpack nodes = basis

    # Get cell length in reference mesh
    dx = 2 / size(mesh, 1)
    dy = 2 / size(mesh, 2)
    dz = 2 / size(mesh, 3)

    # Calculate node coordinates of reference mesh
    cell_x_offset = -1 + (cell_x - 1) * dx + dx / 2
    cell_y_offset = -1 + (cell_y - 1) * dy + dy / 2
    cell_z_offset = -1 + (cell_z - 1) * dz + dz / 2

    for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
        # node_coordinates are the mapped reference node_coordinates
        node_coordinates[:, i, j, k, element] .= mapping(cell_x_offset +
                                                         dx / 2 * nodes[i],
                                                         cell_y_offset +
                                                         dy / 2 * nodes[j],
                                                         cell_z_offset +
                                                         dz / 2 * nodes[k])
    end
end

function calc_node_coordinates_computational!(node_coordinates_comp, cell_x, cell_y, cell_z, mesh, basis)
    @unpack nodes = basis

    # Get cell length in reference mesh
    dx = 2 / size(mesh, 1)
    dy = 2 / size(mesh, 2)
    dz = 2 / size(mesh, 3)

    # Calculate node coordinates of reference mesh
    cell_x_offset = -1 + (cell_x - 1) * dx + dx / 2
    cell_y_offset = -1 + (cell_y - 1) * dy + dy / 2
    cell_z_offset = -1 + (cell_z - 1) * dz + dz / 2

    for i in eachnode(basis)
        # node_coordinates are the mapped reference node_coordinates
        node_coordinates_comp[1, i] = cell_x_offset + dx / 2 * nodes[i]
        node_coordinates_comp[2, i] = cell_y_offset + dy / 2 * nodes[i]
        node_coordinates_comp[3, i] = cell_z_offset + dz / 2 * nodes[i]
    end
end

theta_der1(xi, eta, zeta) = -(0.1 * pi) * sin(pi * xi) * cos(pi * eta) * cos(pi * zeta) 
theta_der2(xi, eta, zeta) = -(0.1 * pi) * cos(pi * xi) * sin(pi * eta) * cos(pi * zeta) 
theta_der3(xi, eta, zeta) = -(0.1 * pi) * cos(pi * xi) * cos(pi * eta) * sin(pi * zeta) 

# Calculate Jacobian matrix of the mapping from the reference element to the element in the physical domain
function calc_jacobian_matrix_exact!(jacobian_matrix::AbstractArray{<:Any, 6}, element,
    node_coordinates, basis, nodes)
    # for dim in 1:3, j in eachnode(basis), i in eachnode(basis)
    #   # ∂/∂ξ
    #   jacobian_matrix[dim, 1, :, i, j, element] = basis.derivative_matrix * node_coordinates[dim, :, i, j, element]
    #   # ∂/∂η
    #   jacobian_matrix[dim, 2, i, :, j, element] = basis.derivative_matrix * node_coordinates[dim, i, :, j, element]
    #   # ∂/∂ζ
    #   jacobian_matrix[dim, 3, i, j, :, element] = basis.derivative_matrix * node_coordinates[dim, i, j, :, element]
    # end
    @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
        jacobian_matrix[1, 1, i, j, k, element] = 1 + theta_der1(nodes[1,i], nodes[2,j], nodes[3,k])
        jacobian_matrix[2, 1, i, j, k, element] = theta_der1(nodes[1,i], nodes[2,j], nodes[3,k])
        jacobian_matrix[3, 1, i, j, k, element] = theta_der1(nodes[1,i], nodes[2,j], nodes[3,k])

        jacobian_matrix[1, 2, i, j, k, element] = theta_der2(nodes[1,i], nodes[2,j], nodes[3,k])
        jacobian_matrix[2, 2, i, j, k, element] = 1 + theta_der2(nodes[1,i], nodes[2,j], nodes[3,k])
        jacobian_matrix[3, 2, i, j, k, element] = theta_der2(nodes[1,i], nodes[2,j], nodes[3,k])

        jacobian_matrix[1, 3, i, j, k, element] = theta_der3(nodes[1,i], nodes[2,j], nodes[3,k])
        jacobian_matrix[2, 3, i, j, k, element] = theta_der3(nodes[1,i], nodes[2,j], nodes[3,k])
        jacobian_matrix[3, 3, i, j, k, element] = 1 + theta_der3(nodes[1,i], nodes[2,j], nodes[3,k])
    end
end

# Calculate Jacobian matrix of the mapping from the reference element to the element in the physical domain
function calc_jacobian_matrix!(jacobian_matrix::AbstractArray{<:Any, 6}, element,
                               node_coordinates, basis)
    # The code below is equivalent to the following matrix multiplications but much faster.
    #
    # for dim in 1:3, j in eachnode(basis), i in eachnode(basis)
    #   # ∂/∂ξ
    #   jacobian_matrix[dim, 1, :, i, j, element] = basis.derivative_matrix * node_coordinates[dim, :, i, j, element]
    #   # ∂/∂η
    #   jacobian_matrix[dim, 2, i, :, j, element] = basis.derivative_matrix * node_coordinates[dim, i, :, j, element]
    #   # ∂/∂ζ
    #   jacobian_matrix[dim, 3, i, j, :, element] = basis.derivative_matrix * node_coordinates[dim, i, j, :, element]
    # end

    @turbo for dim in 1:3, k in eachnode(basis), j in eachnode(basis),
               i in eachnode(basis)

        result = zero(eltype(jacobian_matrix))

        for ii in eachnode(basis)
            result += basis.derivative_matrix[i, ii] *
                      node_coordinates[dim, ii, j, k, element]
        end

        jacobian_matrix[dim, 1, i, j, k, element] = result
    end

    @turbo for dim in 1:3, k in eachnode(basis), j in eachnode(basis),
               i in eachnode(basis)

        result = zero(eltype(jacobian_matrix))

        for ii in eachnode(basis)
            result += basis.derivative_matrix[j, ii] *
                      node_coordinates[dim, i, ii, k, element]
        end

        jacobian_matrix[dim, 2, i, j, k, element] = result
    end

    @turbo for dim in 1:3, k in eachnode(basis), j in eachnode(basis),
               i in eachnode(basis)

        result = zero(eltype(jacobian_matrix))

        for ii in eachnode(basis)
            result += basis.derivative_matrix[k, ii] *
                      node_coordinates[dim, i, j, ii, element]
        end

        jacobian_matrix[dim, 3, i, j, k, element] = result
    end

    return jacobian_matrix
end

# Calculate contravariant vectors, multiplied by the Jacobian determinant J of the transformation mapping,
# using the invariant curl form.
# These are called Ja^i in Kopriva's blue book.
function calc_contravariant_vectors!(contravariant_vectors::AbstractArray{<:Any, 6},
                                     element,
                                     jacobian_matrix, node_coordinates,
                                     basis::LobattoLegendreBasis)
    @unpack derivative_matrix = basis

    # The general form is
    # Jaⁱₙ = 0.5 * ( ∇ × (Xₘ ∇ Xₗ - Xₗ ∇ Xₘ) )ᵢ  where (n, m, l) cyclic and ∇ = (∂/∂ξ, ∂/∂η, ∂/∂ζ)ᵀ

    for n in 1:3
        # (n, m, l) cyclic
        m = (n % 3) + 1
        l = ((n + 1) % 3) + 1

        # Calculate Ja¹ₙ = 0.5 * [ (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_η - (Xₘ Xₗ_η - Xₗ Xₘ_η)_ζ ]
        # For each of these, the first and second summand are computed in separate loops
        # for performance reasons.

        # First summand 0.5 * (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_η
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to j-dimension to differentiate wrt η
                result += 0.5 * derivative_matrix[j, ii] *
                          (node_coordinates[m, i, ii, k, element] *
                           jacobian_matrix[l, 3, i, ii, k, element] -
                           node_coordinates[l, i, ii, k, element] *
                           jacobian_matrix[m, 3, i, ii, k, element])
            end

            contravariant_vectors[n, 1, i, j, k, element] = result
        end

        # Second summand -0.5 * (Xₘ Xₗ_η - Xₗ Xₘ_η)_ζ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to k-dimension to differentiate wrt ζ
                result += 0.5 * derivative_matrix[k, ii] *
                          (node_coordinates[m, i, j, ii, element] *
                           jacobian_matrix[l, 2, i, j, ii, element] -
                           node_coordinates[l, i, j, ii, element] *
                           jacobian_matrix[m, 2, i, j, ii, element])
            end

            contravariant_vectors[n, 1, i, j, k, element] -= result
        end

        # Calculate Ja²ₙ = 0.5 * [ (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_ζ - (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_ξ ]

        # First summand 0.5 * (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_ζ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to k-dimension to differentiate wrt ζ
                result += 0.5 * derivative_matrix[k, ii] *
                          (node_coordinates[m, i, j, ii, element] *
                           jacobian_matrix[l, 1, i, j, ii, element] -
                           node_coordinates[l, i, j, ii, element] *
                           jacobian_matrix[m, 1, i, j, ii, element])
            end

            contravariant_vectors[n, 2, i, j, k, element] = result
        end

        # Second summand -0.5 * (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_ξ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to i-dimension to differentiate wrt ξ
                result += 0.5 * derivative_matrix[i, ii] *
                          (node_coordinates[m, ii, j, k, element] *
                           jacobian_matrix[l, 3, ii, j, k, element] -
                           node_coordinates[l, ii, j, k, element] *
                           jacobian_matrix[m, 3, ii, j, k, element])
            end

            contravariant_vectors[n, 2, i, j, k, element] -= result
        end

        # Calculate Ja³ₙ = 0.5 * [ (Xₘ Xₗ_η - Xₗ Xₘ_η)_ξ - (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_η ]

        # First summand 0.5 * (Xₘ Xₗ_η - Xₗ Xₘ_η)_ξ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to i-dimension to differentiate wrt ξ
                result += 0.5 * derivative_matrix[i, ii] *
                          (node_coordinates[m, ii, j, k, element] *
                           jacobian_matrix[l, 2, ii, j, k, element] -
                           node_coordinates[l, ii, j, k, element] *
                           jacobian_matrix[m, 2, ii, j, k, element])
            end

            contravariant_vectors[n, 3, i, j, k, element] = result
        end

        # Second summand -0.5 * (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_η
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to j-dimension to differentiate wrt η
                result += 0.5 * derivative_matrix[j, ii] *
                          (node_coordinates[m, i, ii, k, element] *
                           jacobian_matrix[l, 1, i, ii, k, element] -
                           node_coordinates[l, i, ii, k, element] *
                           jacobian_matrix[m, 1, i, ii, k, element])
            end

            contravariant_vectors[n, 3, i, j, k, element] -= result
        end
    end

    return contravariant_vectors
end

# Calculate contravariant vectors, multiplied by the Jacobian determinant J of the transformation mapping,
# using the invariant curl form.
# These are called Ja^i in Kopriva's blue book.
function calc_contravariant_vectors_standard_curl!(contravariant_vectors::AbstractArray{<:Any, 6},
                                     element,
                                     jacobian_matrix, node_coordinates,
                                     basis::LobattoLegendreBasis)
    @unpack derivative_matrix = basis

    # The general form is
    # Jaⁱₙ = 0.5 * ( ∇ × (Xₘ ∇ Xₗ - Xₗ ∇ Xₘ) )ᵢ  where (n, m, l) cyclic and ∇ = (∂/∂ξ, ∂/∂η, ∂/∂ζ)ᵀ

    for n in 1:3
        # (n, m, l) cyclic
        m = (n % 3) + 1
        l = ((n + 1) % 3) + 1

        # Calculate Ja¹ₙ = 0.5 * [ (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_η - (Xₘ Xₗ_η - Xₗ Xₘ_η)_ζ ]
        # For each of these, the first and second summand are computed in separate loops
        # for performance reasons.

        # First summand 0.5 * (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_η
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to j-dimension to differentiate wrt η
                result += derivative_matrix[j, ii] *
                          ( - node_coordinates[l, i, ii, k, element] *
                           jacobian_matrix[m, 3, i, ii, k, element])
            end

            contravariant_vectors[n, 1, i, j, k, element] = result
        end

        # Second summand -0.5 * (Xₘ Xₗ_η - Xₗ Xₘ_η)_ζ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to k-dimension to differentiate wrt ζ
                result += derivative_matrix[k, ii] *
                          ( -node_coordinates[l, i, j, ii, element] *
                           jacobian_matrix[m, 2, i, j, ii, element])
            end

            contravariant_vectors[n, 1, i, j, k, element] -= result
        end

        # Calculate Ja²ₙ = 0.5 * [ (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_ζ - (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_ξ ]

        # First summand 0.5 * (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_ζ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to k-dimension to differentiate wrt ζ
                result += derivative_matrix[k, ii] *
                          (-node_coordinates[l, i, j, ii, element] *
                           jacobian_matrix[m, 1, i, j, ii, element])
            end

            contravariant_vectors[n, 2, i, j, k, element] = result
        end

        # Second summand -0.5 * (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_ξ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to i-dimension to differentiate wrt ξ
                result += derivative_matrix[i, ii] *
                          (-node_coordinates[l, ii, j, k, element] *
                           jacobian_matrix[m, 3, ii, j, k, element])
            end

            contravariant_vectors[n, 2, i, j, k, element] -= result
        end

        # Calculate Ja³ₙ = 0.5 * [ (Xₘ Xₗ_η - Xₗ Xₘ_η)_ξ - (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_η ]

        # First summand 0.5 * (Xₘ Xₗ_η - Xₗ Xₘ_η)_ξ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to i-dimension to differentiate wrt ξ
                result += derivative_matrix[i, ii] *
                          (-node_coordinates[l, ii, j, k, element] *
                           jacobian_matrix[m, 2, ii, j, k, element])
            end

            contravariant_vectors[n, 3, i, j, k, element] = result
        end

        # Second summand -0.5 * (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_η
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to j-dimension to differentiate wrt η
                result += derivative_matrix[j, ii] *
                          (-node_coordinates[l, i, ii, k, element] *
                           jacobian_matrix[m, 1, i, ii, k, element])
            end

            contravariant_vectors[n, 3, i, j, k, element] -= result
        end
    end

    return contravariant_vectors
end

"""
New function to compute contravariant vectors
"""
function calc_contravariant_vectors_mimetic!(contravariant_vectors::AbstractArray{<:Any, 6},
    element,
    jacobian_matrix, node_coordinates,
    basis::LobattoLegendreBasis,
    nodes)
    @unpack derivative_matrix = basis

    # Define histopolation (edge) basis functions: V[i,j] = hⱼ(ξᵢ) ... TODO: initialize beforehand...
    V = zero(MMatrix{polydeg(basis) + 1, polydeg(basis), eltype(derivative_matrix)})
    for j in 1:polydeg(basis)
        for i in 1:polydeg(basis)+1
            for k in 1:j
                V[i, j] -= derivative_matrix[i, k]
            end
        end
    end

    # Project the mapping potential \vec{g} \in H_{curl} to \vec{G} \in V_1
    Gbar = zeros(eltype(derivative_matrix), 3, 3, nnodes(basis), nnodes(basis), nnodes(basis)) # Attention: here I'm allocating N+1 nodes in each direction. We only need N in some directions!!
    for k in eachnode(basis)
        for j in eachnode(basis)
            for i in 1:polydeg(basis)
                Gbar[1, 1, i, j, k] = (nodes[3, k] * (theta(nodes[1,i+1], nodes[2,j], nodes[3,k]) - theta(nodes[1,i], nodes[2,j], nodes[3,k]) )
                                       + 0.5 * (theta(nodes[1,i+1], nodes[2,j], nodes[3,k])^2 - theta(nodes[1,i], nodes[2,j], nodes[3,k])^2 ) )
                Gbar[2, 1, i, j, k] = (nodes[1, i+1] * theta(nodes[1,i+1], nodes[2,j], nodes[3,k]) - nodes[1, i] * theta(nodes[1,i], nodes[2,j], nodes[3,k]) 
                                       + 0.5 * (theta(nodes[1,i+1], nodes[2,j], nodes[3,k])^2 - theta(nodes[1,i], nodes[2,j], nodes[3,k])^2 ) 
                                       - (theta_int1(nodes[1,i+1], nodes[2,j], nodes[3,k]) - theta_int1(nodes[1,i], nodes[2,j], nodes[3,k])))
                Gbar[3, 1, i, j, k] = ( nodes[2, j] * (nodes[1, i+1] - nodes[1, i])
                                       + nodes[2, j] * ( theta(nodes[1,i+1], nodes[2,j], nodes[3,k]) - theta(nodes[1,i], nodes[2,j], nodes[3,k]) )
                                       + 0.5 * (theta(nodes[1,i+1], nodes[2,j], nodes[3,k])^2 - theta(nodes[1,i], nodes[2,j], nodes[3,k])^2 ) 
                                       + (theta_int1(nodes[1,i+1], nodes[2,j], nodes[3,k]) - theta_int1(nodes[1,i], nodes[2,j], nodes[3,k])))
            end
        end
    end
    for k in eachnode(basis)
        for j in 1:polydeg(basis)
            for i in eachnode(basis)
                Gbar[1, 2, i, j, k] = ( nodes[3, k] * (nodes[2, j+1] - nodes[2, j])
                                        + nodes[3, k] * ( theta(nodes[1,i], nodes[2,j+1], nodes[3,k]) - theta(nodes[1,i], nodes[2,j], nodes[3,k]) )
                                        + 0.5 * (theta(nodes[1,i], nodes[2,j+1], nodes[3,k])^2 - theta(nodes[1,i], nodes[2,j], nodes[3,k])^2 ) 
                                        + (theta_int2(nodes[1,i], nodes[2,j+1], nodes[3,k]) - theta_int2(nodes[1,i], nodes[2,j], nodes[3,k])))
                Gbar[2, 2, i, j, k] = (nodes[1, i] * (theta(nodes[1,i], nodes[2,j+1], nodes[3,k]) - theta(nodes[1,i], nodes[2,j], nodes[3,k]) )
                                       + 0.5 * (theta(nodes[1,i], nodes[2,j+1], nodes[3,k])^2 - theta(nodes[1,i], nodes[2,j], nodes[3,k])^2 ) )
                Gbar[3, 2, i, j, k] = (nodes[2, j+1] * theta(nodes[1,i], nodes[2,j+1], nodes[3,k]) - nodes[2, j] * theta(nodes[1,i], nodes[2,j], nodes[3,k]) 
                                        + 0.5 * (theta(nodes[1,i], nodes[2,j+1], nodes[3,k])^2 - theta(nodes[1,i], nodes[2,j], nodes[3,k])^2 ) 
                                        - (theta_int2(nodes[1,i], nodes[2,j+1], nodes[3,k]) - theta_int2(nodes[1,i], nodes[2,j], nodes[3,k])))
            end
        end
    end
    for k in 1:polydeg(basis)
        for j in eachnode(basis)
            for i in eachnode(basis)
                Gbar[1, 3, i, j, k] = (nodes[3, k+1] * theta(nodes[1,i], nodes[2,j], nodes[3,k+1]) - nodes[3, k] * theta(nodes[1,i], nodes[2,j], nodes[3,k]) 
                                        + 0.5 * (theta(nodes[1,i], nodes[2,j], nodes[3,k+1])^2 - theta(nodes[1,i], nodes[2,j], nodes[3,k])^2 ) 
                                        - (theta_int3(nodes[1,i], nodes[2,j], nodes[3,k+1]) - theta_int3(nodes[1,i], nodes[2,j], nodes[3,k])))
                Gbar[2, 3, i, j, k] = ( nodes[1, i] * (nodes[3, k+1] - nodes[3, k])
                                        + nodes[1, i] * ( theta(nodes[1,i], nodes[2,j], nodes[3,k+1]) - theta(nodes[1,i], nodes[2,j], nodes[3,k]) )
                                        + 0.5 * (theta(nodes[1,i], nodes[2,j], nodes[3,k+1])^2 - theta(nodes[1,i], nodes[2,j], nodes[3,k])^2 ) 
                                        + (theta_int3(nodes[1,i], nodes[2,j], nodes[3,k+1]) - theta_int3(nodes[1,i], nodes[2,j], nodes[3,k])))
                Gbar[3, 3, i, j, k] = (nodes[2, j] * (theta(nodes[1,i], nodes[2,j], nodes[3,k+1]) - theta(nodes[1,i], nodes[2,j], nodes[3,k]) )
                                       + 0.5 * (theta(nodes[1,i], nodes[2,j], nodes[3,k+1])^2 - theta(nodes[1,i], nodes[2,j], nodes[3,k])^2 ) )
            end
        end
    end

    # Evaluate the mapping potential at the Lagrange points
    G = zeros(eltype(derivative_matrix), 3, 3, nnodes(basis), nnodes(basis), nnodes(basis))
    for k in eachnode(basis)
        for j in eachnode(basis)
            for i in eachnode(basis)
                for ii in 1:polydeg(basis)
                    G[:, 1, i, j, k] += Gbar[:, 1, ii, j, k] * V[i, ii]
                    G[:, 2, i, j, k] += Gbar[:, 2, i, ii, k] * V[j, ii]
                    G[:, 3, i, j, k] += Gbar[:, 3, i, j, ii] * V[k, ii]
                end
            end
        end
    end

    # Compute the contravariant vectors as the curl of the mapping potential (at the discrete level!)
    # Jaⁱₙ = -( ∇ × gₙ )ᵢ  where ∇ = (∂/∂ξ, ∂/∂η, ∂/∂ζ)ᵀ
    for n in 1:3
        # Calculate Ja¹ₙ = -(g³ₙ)_η + (g²ₙ)_ζ
        # For each of these, the first and second summand are computed in separate loops
        # for performance reasons.

        # First summand (g³ₙ)_η
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to j-dimension to differentiate wrt η
                result += derivative_matrix[j, ii] * G[n, 3, i, ii, k]
            end

            contravariant_vectors[n, 1, i, j, k, element] = -result
        end

        # Second summand -(g²ₙ)_ζ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to k-dimension to differentiate wrt ζ
                result += derivative_matrix[k, ii] * G[n, 2, i, j, ii]
            end

            contravariant_vectors[n, 1, i, j, k, element] += result
        end

        # Calculate Ja²ₙ = -(g¹ₙ)_ζ + (g³ₙ)_ξ

        # First summand (g¹ₙ)_ζ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to k-dimension to differentiate wrt ζ
                result += derivative_matrix[k, ii] * G[n, 1, i, j, ii]
            end

            contravariant_vectors[n, 2, i, j, k, element] = -result
        end

        # Second summand -(g³ₙ)_ξ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to i-dimension to differentiate wrt ξ
                result += derivative_matrix[i, ii] * G[n, 3, ii, j, k]
            end

            contravariant_vectors[n, 2, i, j, k, element] += result
        end

        # Calculate Ja³ₙ = -(g²ₙ)_ξ + (g¹ₙ)_η

        # First summand (g²ₙ)_ξ
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to i-dimension to differentiate wrt ξ
                result += derivative_matrix[i, ii] * G[n, 2, ii, j, k]
            end

            contravariant_vectors[n, 3, i, j, k, element] = -result
        end

        # Second summand -(g¹ₙ)_η
        @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
            result = zero(eltype(contravariant_vectors))

            for ii in eachnode(basis)
                # Multiply derivative_matrix to j-dimension to differentiate wrt η
                result += derivative_matrix[j, ii] * G[n, 1, i, ii, k]
            end

            contravariant_vectors[n, 3, i, j, k, element] += result
        end
    end

    return contravariant_vectors
end

theta(xi, eta, zeta) = 0.1 * cos(pi * xi) * cos(pi * eta) * cos(pi * zeta) 
theta_int1(xi, eta, zeta) = (0.1 / pi) * sin(pi * xi) * cos(pi * eta) * cos(pi * zeta) 
theta_int2(xi, eta, zeta) = (0.1 / pi) * cos(pi * xi) * sin(pi * eta) * cos(pi * zeta) 
theta_int3(xi, eta, zeta) = (0.1 / pi) * cos(pi * xi) * cos(pi * eta) * sin(pi * zeta) 

# Calculate inverse Jacobian (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 4}, element,
                                jacobian_matrix, basis)
    @turbo for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
        # Calculate Determinant by using Sarrus formula (about 100 times faster than LinearAlgebra.det())
        inverse_jacobian[i, j, k, element] = inv(jacobian_matrix[1, 1, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[2, 2, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[3, 3, i, j, k, element] +
                                                 jacobian_matrix[1, 2, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[2, 3, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[3, 1, i, j, k, element] +
                                                 jacobian_matrix[1, 3, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[2, 1, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[3, 2, i, j, k, element] -
                                                 jacobian_matrix[3, 1, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[2, 2, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[1, 3, i, j, k, element] -
                                                 jacobian_matrix[3, 2, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[2, 3, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[1, 1, i, j, k, element] -
                                                 jacobian_matrix[3, 3, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[2, 1, i, j, k,
                                                                 element] *
                                                 jacobian_matrix[1, 2, i, j, k, element])
    end

    return inverse_jacobian
end

# Save id of left neighbor of every element
function initialize_left_neighbor_connectivity!(left_neighbors, mesh::StructuredMesh{3},
                                                linear_indices)
    # Neighbors in x-direction
    for cell_z in 1:size(mesh, 3), cell_y in 1:size(mesh, 2)
        # Inner elements
        for cell_x in 2:size(mesh, 1)
            element = linear_indices[cell_x, cell_y, cell_z]
            left_neighbors[1, element] = linear_indices[cell_x - 1, cell_y, cell_z]
        end

        if isperiodic(mesh, 1)
            # Periodic boundary
            left_neighbors[1, linear_indices[1, cell_y, cell_z]] = linear_indices[end,
                                                                                  cell_y,
                                                                                  cell_z]
        else
            left_neighbors[1, linear_indices[1, cell_y, cell_z]] = 0
        end
    end

    # Neighbors in y-direction
    for cell_z in 1:size(mesh, 3), cell_x in 1:size(mesh, 1)
        # Inner elements
        for cell_y in 2:size(mesh, 2)
            element = linear_indices[cell_x, cell_y, cell_z]
            left_neighbors[2, element] = linear_indices[cell_x, cell_y - 1, cell_z]
        end

        if isperiodic(mesh, 2)
            # Periodic boundary
            left_neighbors[2, linear_indices[cell_x, 1, cell_z]] = linear_indices[cell_x,
                                                                                  end,
                                                                                  cell_z]
        else
            left_neighbors[2, linear_indices[cell_x, 1, cell_z]] = 0
        end
    end

    # Neighbors in z-direction
    for cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
        # Inner elements
        for cell_z in 2:size(mesh, 3)
            element = linear_indices[cell_x, cell_y, cell_z]
            left_neighbors[3, element] = linear_indices[cell_x, cell_y, cell_z - 1]
        end

        if isperiodic(mesh, 3)
            # Periodic boundary
            left_neighbors[3, linear_indices[cell_x, cell_y, 1]] = linear_indices[cell_x,
                                                                                  cell_y,
                                                                                  end]
        else
            left_neighbors[3, linear_indices[cell_x, cell_y, 1]] = 0
        end
    end

    return left_neighbors
end
end # @muladd
