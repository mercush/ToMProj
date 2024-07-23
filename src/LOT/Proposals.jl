include("Prior.jl")
@gen function random_node_path(choices::ChoiceMap)::Union{Pair,Symbol}
    submaps = [addr for (addr, val) in get_submaps_shallow(choices)]
    if ({:stop} ~ bernoulli(length(submaps) == 0 ? 1.0 : 0.5))
        return :tree
    end

    sample_next ~ categorical(normalize(ones(length(submaps))))
    direction = submaps[sample_next]
    next_submap = get_submap(choices, direction)
    rest_of_path ~ random_node_path(next_submap)

    if isa(rest_of_path, Pair)
        return :tree => direction => rest_of_path[2]
    else
        return :tree => direction
    end
end

@gen function gen_subtree(node_type)
    if node_type in [MakeMove, If]
        return {*} ~ pcfg_Op()
    elseif node_type in [CustomInt, CustomZ3, CountMove, CountOppMove, CountTransition, CountOppTransition]
        return {*} ~ pcfg_Expr()
    elseif node_type in [RawMove, PrevMove, PrevOppMove, Inc, Dec, Random]
        return {*} ~ pcfg_MoveType()
    elseif node_type in [Equal, Lt, Leq]
        return {*} ~ pcfg_Bool()
    elseif node_type in [Expr, MoveType, PrevOutcome]
        return {*} ~ pcfg_BoolInput()
    end
    error("$node_type not a valid type")
end

@gen function regen_random_subtree_randomness(prev_trace::Gen.Trace)::Union{Pair,Symbol}
    choices = get_choices(prev_trace)
    path ~ random_node_path(get_submap(choices, :tree))
    subtree = get_submap(choices, path)
    new_subtree ~ gen_subtree(subtree[:node_type])
    return path
end

# @gen function regen_swap_node_randomness(prev_trace::Gen.Trace)::Union{Pair,Symbol}
#     choices = get_choices(prev_trace)
#     path ~ random_node_path(get_submap(choices, :tree))
#     subtree = get_submap(choices, path)
#     new_subtree ~ gen_subtree(subtree[:node_type])
#     return path
# end

function subtree_involution(trace::Trace, forward_choices::ChoiceMap, path_to_subtree::Union{Pair,Symbol}, proposal_args)::Tuple{Gen.Trace,Gen.ChoiceMap,Float64}
    # Need to return a new trace, backward_choices, and a weight.
    backward_choices = Gen.choicemap()
    
    # In the backward direction, the `random_node_path` function should
    # make all the same choices, so that the same exact node is reached
    # for resimulation.
    set_submap!(backward_choices, :path, get_submap(forward_choices, :path))
    
    # But in the backward direction, the `:new_subtree` generation should
    # produce the *existing* subtree.
    set_submap!(backward_choices, :new_subtree, get_submap(get_choices(trace), path_to_subtree))
    
    # The new trace should be just like the old one, but we are updating everything
    # about the new subtree.
    new_trace_choices = Gen.choicemap()
    set_submap!(new_trace_choices, path_to_subtree, get_submap(forward_choices, :new_subtree))
    # path_to_subtree is the retval of the proposal. forward_choices are the choices made by the proposal
    
    # Run update and get the new weight.
    new_trace, weight, = update(trace, get_args(trace), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end

# function swap_involution(trace::Trace, forward_choices::ChoiceMap, path_to_subtree::Union{Pair, Symbol}, proposal_args)::Tuple{Gen.Trace,Gen.ChoiceMap,Float64}
#     # Need to return a new trace, backward_choices, and a weight.
#     backward_choices = Gen.choicemap()
    
#     # In the backward direction, the `random_node_path` function should
#     # make all the same choices, so that the same exact node is reached
#     # for resimulation.
#     set_submap!(backward_choices, :path, get_submap(forward_choices, :path))
    
#     # But in the backward direction, the `:new_subtree` generation should
#     # produce the *existing* subtree.
#     set_submap!(backward_choices, :new_subtree, get_submap(get_choices(trace), path_to_subtree))
    
#     # The new trace should be just like the old one, but we are updating everything
#     # about the new subtree.
#     new_trace_choices = Gen.choicemap()
#     set_submap!(new_trace_choices, path_to_subtree, get_submap(forward_choices, :new_subtree))
#     # path_to_subtree is the retval of the proposal. forward_choices are the choices made by the proposal
    
#     # Run update and get the new weight.
#     new_trace, weight, = update(trace, get_args(trace), (NoChange(),), new_trace_choices)
#     (new_trace, backward_choices, weight)
# end

@gen function gaussian_drift(tr, addrs)
    for addr in addrs
        {addr} ~ normal(tr[addr], 0.5)
    end
end

