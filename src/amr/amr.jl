module AMR

using ..Trixi
using ..Auxiliary: parameter, timer
using ..Auxiliary.Containers: append!
using ..Mesh: TreeMesh
using ..Mesh.Trees: leaf_cells, n_children_per_cell, has_parent, is_leaf
using ..Solvers: AbstractSolver, calc_amr_indicator
import ..Mesh # to use refine!
import ..Mesh.Trees # to use refine!
import ..Solvers # to use refine!

using TimerOutputs: @timeit, print_timer
using HDF5: h5open, attrs


export adapt!


function adapt!(mesh::TreeMesh, solver::AbstractSolver; only_refine=false, only_coarsen=false)
  print("Begin adaptation...")
  # Alias for convenience
  tree = mesh.tree

  # Determine indicator value
  lambda = @timeit timer() "indicator" calc_amr_indicator(solver, mesh)

  # Get list of current leaf cells
  leaf_cell_ids = leaf_cells(tree)
  @assert length(lambda) == length(leaf_cell_ids) ("Indicator and leaf cell arrays have " *
                                                   "different length")

  # Set thresholds for refinement and coarsening
  refinement_threshold = parameter("refinement_threshold",  0.5)
  coarsening_threshold = parameter("coarsening_threshold", -0.5)

  # Determine list of cells to refine or coarsen
  to_refine = leaf_cell_ids[lambda .> refinement_threshold]
  to_coarsen = leaf_cell_ids[lambda .< coarsening_threshold]

  # Start by refining cells
  @timeit timer() "refine" if !only_coarsen && !isempty(to_refine)
    # Refine cells
    refined_original_cells = @timeit timer() "mesh" Mesh.Trees.refine!(tree, to_refine)

    # Refine elements
     @timeit timer() "solver" Solvers.refine!(solver, mesh, refined_original_cells)
  else
    refined_original_cells = Int[]
  end

  # Then, coarsen cells
  @timeit timer() "coarsen" if !only_refine && !isempty(to_coarsen)
    # Since the cells may have been shifted due to refinement, first we need to
    # translate the old cell ids to the new cell ids
    if !isempty(to_coarsen)
      to_coarsen = original2refined(to_coarsen, refined_original_cells)
    end

    # Next, determine the parent cells from which the fine cells are to be
    # removed, since these are needed for the coarsen! function. However, since
    # we only want to coarsen if *all* child cells are marked for coarsening,
    # we count the coarsening indicators for each parent cell and only coarsen
    # if all children are marked as such (i.e., where the count is 2^ndim). In
    # the same time, check if a cell is marked for coarsening even though it is
    # *not* a leaf cell -> this can only happen if it was refined due to 2:1
    # smoothing during the preceding refinement operation.
    parents_to_coarsen = zeros(Int, length(tree))
    for cell_id in to_coarsen
      if !has_parent(tree, cell_id)
        continue
      end

      if !is_leaf(tree, cell_id)
        continue
      end

      parent_id = tree.parent_ids[cell_id]
      parents_to_coarsen[parent_id] += 1
    end
    to_coarsen = collect(1:length(parents_to_coarsen))[parents_to_coarsen .== 2^ndim]

    # Finally, coarsen cells
    coarsened_original_cells = @timeit timer() "mesh" Mesh.Trees.coarsen!(tree, to_coarsen)

    # Convert coarsened parent cell ids to the list of child cell ids that have been removed
    removed_child_cells = zeros(Int, n_children_per_cell(tree) * length(coarsened_original_cells))
    for (index, coarse_cell_id) in enumerate(coarsened_original_cells)
      for child in 1:n_children_per_cell(tree)
        removed_child_cells[n_children_per_cell(tree) * (index-1) + child] = coarse_cell_id + child
      end
    end

    # Coarsen elements
    @timeit timer() "solver" Solvers.coarsen!(solver, mesh, removed_child_cells)
  else
    coarsened_original_cells = Int[]
  end

  println("done (refined: $(length(refined_original_cells)), " *
                 "coarsened: $(length(coarsened_original_cells)), " *
                 "new number of cells/elements: $(length(tree))/$(solver.n_elements))")

  return !isempty(refined_original_cells) || !isempty(coarsened_original_cells)
end


# After refining cells, shift original cell ids to match new locations
# Note: Assumes sorted lists of original and refined cell ids!
function original2refined(original_cell_ids::AbstractVector{Int},
                          refined_original_cells::AbstractVector{Int})
  # Sanity check
  @assert issorted(original_cell_ids) "`original_cell_ids` not sorted"
  @assert issorted(refined_original_cells) "`refined_cell_ids` not sorted"

  # Create array with original cell ids (not yet shifted)
  shifted_cell_ids = collect(1:original_cell_ids[end])

  # Loop over refined original cells and apply shift for all following cells
  for cell_id in refined_original_cells
    # Only calculate shifts for cell ids that are relevant
    if cell_id > length(shifted_cell_ids)
      break
    end

    # Shift all subsequent cells by 2^ndim ids
    shifted_cell_ids[(cell_id + 1):end] .+= 2^ndim
  end

  # Convert original cell ids to their shifted values
  return shifted_cell_ids[original_cell_ids]
end


end # module AMR
