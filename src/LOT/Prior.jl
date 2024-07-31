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

MoveType_child = [RawMove, PrevMove, PrevOppMove, Inc, Dec, Random]
@dist MoveType_child_dist() = MoveType_child[categorical(normalize([1,1,1,1,1,1]))]
@gen function pcfg_MoveType()
    node_type ~ MoveType_child_dist()
    if node_type == RawMove
        return node_type({:param_z3} ~ z3_dist())
    elseif node_type == PrevMove
        return node_type()
    elseif node_type == PrevOppMove
        return node_type()
    elseif node_type == Inc
        return node_type({:move_type} ~ pcfg_MoveType())
    elseif node_type == Dec
        return node_type({:move_type} ~ pcfg_MoveType())
    elseif node_type == Random
        return {:random} ~ pcfg_Random()
    end
    error("$node_type not a valid type")
end

Expr_child = [CustomZ3, CustomInt, CountMove, CountOppMove, CountTransition, 
    CountOppTransition, PrevMoveExpr, PrevOppMoveExpr, IncExpr, DecExpr, PrevOutcome]
@dist Expr_child_dist() = Expr_child[categorical(normalize(ones(11)))]
@gen function pcfg_Expr()
    node_type ~ Expr_child_dist()
    if node_type == CustomZ3
        return node_type({:param_z3} ~ z3_dist())
    elseif node_type == CustomInt
        return node_type({:param_int} ~ int_prior_dist())
    elseif node_type == CountMove
        return node_type({:param_z3} ~ z3_dist())
    elseif node_type == CountOppMove
        return node_type({:param_z3} ~ z3_dist())
    elseif node_type == CountTransition
        return node_type({:param_z3} ~ z3_dist())
    elseif node_type == CountOppTransition
        return node_type({:param_z3} ~ z3_dist())
    elseif node_type == PrevMoveExpr
        return node_type()
    elseif node_type == PrevOppMoveExpr
        return node_type()
    elseif node_type == IncExpr
        return node_type({:expr} ~ pcfg_Expr())
    elseif node_type == DecExpr
        return node_type({:expr} ~ pcfg_Expr())
    elseif node_type == PrevOutcome
        return node_type()
    end
    error("$node_type not a valid type")
end

Random_child = [RandomMoveFixed, RandomTransitionFixed, RandomCorrPrevMove,
    RandomInvCorrPrevMove, RandomCorrPrevTransition, RandomInvCorrPrevTransition]
@dist Random_child_dist() = Random_child[categorical(normalize([1,1,1,1,1,1]))]
@gen function pcfg_Random()
    node_type ~ Random_child_dist()
    if node_type == RandomMoveFixed
        wt = ({*} ~ dirichlet3([1,1,1]))
        return node_type(wt[1], wt[2], wt[3])
    elseif node_type == RandomTransitionFixed
        wt = ({*} ~ dirichlet3([1,1,1]))
        return node_type(wt[1], wt[2], wt[3])
    elseif node_type == RandomCorrPrevMove
        return node_type()
    elseif node_type == RandomInvCorrPrevMove
        return node_type()
    elseif node_type == RandomCorrPrevTransition
        return node_type()
    elseif node_type == RandomInvCorrPrevTransition
        return node_type()
    end
    error("$node_type not a valid type")
end

Bool_child = [Equal, Lt, Leq, Flip]
@dist Bool_child_dist() = Bool_child[categorical(normalize(ones(4)))]
@gen function pcfg_Bool()
    node_type ~ Bool_child_dist()
    if node_type in [Equal, Lt, Leq]
        return node_type({:expr1} ~ pcfg_Expr(), {:expr2} ~ pcfg_Expr())
    elseif node_type == Flip
        gamma1 ~ gamma(1,1)
        gamma2 ~ gamma(1,1)
        partition = gamma1+gamma2
        return node_type(gamma1/partition)
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