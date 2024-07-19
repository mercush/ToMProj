include("Grammar.jl")

@dist z3_dist() = categorical(normalize(ones(3)))-1
@dist number_prior_dist() = normal(0, 1)
@dist function int_prior_dist()
    d = Vector{Float64}()
    for i in 0:4
        append!(d, 1-(i*.04))
    end
    d = normalize(d)
    categorical(d)-1
end

Op_child = [MakeMove, If]
@dist Op_child_dist() = Op_child[categorical(normalize([2,1]))]
@gen function pcfg_Op()
    node_type ~ Op_child_dist()
    if node_type == MakeMove
        return MakeMove({:move_type} ~ pcfg_MoveType())
    elseif node_type == If
        return If({:bool} ~ pcfg_Bool(), {:op1} ~ pcfg_Op(), {:op2} ~ pcfg_Op())
    end
    error("$node_type not a valid type")
end

Expr_child = [CustomZ3, CustomInt, CountMove, CountOppMove, CountTransition, CountOppTransition]
@dist Expr_child_dist() = Expr_child[categorical(normalize([2,2,1,1,1,1]))]
@gen function pcfg_Expr()
    node_type ~ Expr_child_dist()
    if node_type == CustomZ3
        return CustomZ3({:param_z3} ~ z3_dist())
    elseif node_type == CustomInt
        return CustomInt({:param_int} ~ int_prior_dist())
    elseif node_type == CountMove
        return CountMove({:param_z3} ~ z3_dist())
    elseif node_type == CountOppMove
        return CountOppMove({:param_z3} ~ z3_dist())
    elseif node_type == CountTransition
        return CountTransition({:param_z3} ~ z3_dist())
    elseif node_type == CountOppTransition
        return CountOppTransition({:param_z3} ~ z3_dist())
    end
    error("$node_type not a valid type")
end

MoveType_child = [RawMove, PrevMove, PrevOppMove, Inc, Dec, Random]
@dist MoveType_child_dist() = MoveType_child[categorical(normalize([2,2,2,1,1,1]))]
@gen function pcfg_MoveType()
    node_type ~ MoveType_child_dist()
    if node_type == RawMove
        return RawMove({:param_z3} ~ z3_dist())
    elseif node_type == PrevMove
        return PrevMove()
    elseif node_type == PrevOppMove
        return PrevOppMove()
    elseif node_type == Inc
        return Inc({:move_type} ~ pcfg_MoveType())
    elseif node_type == Dec
        return Dec({:move_type} ~ pcfg_MoveType())
    elseif node_type == Random
        return Random(
            {:expr1} ~ pcfg_Expr(), 
            {:expr2} ~ pcfg_Expr(), 
            {:expr3} ~ pcfg_Expr())
    end
    error("$node_type not a valid type")
end

Bool_child = [Equal, Lt, Leq]
@dist Bool_child_dist() = Bool_child[categorical(normalize(ones(3)))]
@gen function pcfg_Bool()
    node_type ~ Bool_child_dist()
    bool_input1 ~ pcfg_BoolInput()
    bool_input2 ~ pcfg_BoolInput()
    return node_type(bool_input1, bool_input2)
end

BoolInput_child = [Expr, MoveType, PrevOutcome]
@dist BoolInput_child_dist() = BoolInput_child[categorical(normalize([6,6,1]))]
@gen function pcfg_BoolInput()
    node_type ~ BoolInput_child_dist()
    if node_type == Expr
        return {:expr} ~ pcfg_Expr()
    elseif node_type == MoveType
        return {:move_type} ~ pcfg_MoveType()
    elseif node_type == PrevOutcome
        return PrevOutcome()
    end
    error("$node_type not a valid type")
end

@gen function pcfg()
    return {*} ~ pcfg_Op()
    # return tree ~ pcfg_Op()
end


# @gen function gen_swap_node(node_type)
#     if node_type in [MakeMove, If]
#         return {*} ~ pcfg_Op()
#     elseif node_type in [CustomInt, CustomZ3, CountMove, CountOppMove, CountTransition, CountOppTransition]
#         return {*} ~ pcfg_Expr()
#     elseif node_type in [RawMove, PrevMove, PrevOppMove, Inc, Dec, Random]
#         return {*} ~ pcfg_MoveType()
#     elseif node_type in [Equal, Lt, Leq]
#         return {*} ~ pcfg_Bool()
#     elseif node_type in [Expr, MoveType, PrevOutcome]
#         return {*} ~ pcfg_BoolInput()
#     end
#     error("$node_type not a valid type")
# end

# @gen function gen_node(node_type)
#     if node_type in [MakeMove, If]
#         return {*} ~ Op_child_dist()
# end

@gen function init_state(state_size::Int)::State
    moves = [{(:moves, i)} ~ z3_dist() for i=1:state_size]
    oppmoves = [{(:oppmoves, i)} ~ z3_dist() for i=1:state_size]
    return State(moves, oppmoves)
end