include("Grammar.jl")

@gen function eval_kern(node, state)
    if isa(node, MakeMove)
        return {*} ~ eval_kern(node.move_type, state)
    elseif isa(node, If)
        op1 ~ eval_kern(node.op1, state)
        op2 ~ eval_kern(node.op2, state)
        bool ~ eval_kern(node.bool, state)
        return bool ?  op1 : op2
    elseif isa(node, RawMove)
        return node.param_z3
    elseif isa(node, PrevMove)
        return state.moves[end]
    elseif isa(node, PrevOppMove)
        return state.oppmoves[end]
    elseif isa(node, Inc)
        return (1 + ({*} ~ eval_kern(node.move_type, state))) % 3
    elseif isa(node, Dec)
        return (2 + ({*} ~ eval_kern(node.move_type, state))) % 3
    elseif isa(node, CustomZ3)
        return node.param_z3
    elseif isa(node, CustomInt)
        return node.param_int
    elseif isa(node, CountMove)
        return sum(state.moves .== node.param_z3)
    elseif isa(node, CountOppMove)
        return sum(state.oppmoves .== node.param_z3)
    elseif isa(node, CountTransition)
        return sum(((3 .+ state.moves[2:end] .- state.moves[1:end-1]) .% 3) .== node.param_z3)
    elseif isa(node, CountOppTransition)
        return sum(((3 .+ state.oppmoves[2:end] .- state.oppmoves[1:end-1]) .% 3) .== node.param_z3)
    elseif isa(node, PrevMoveExpr)
        return state.moves[end]
    elseif isa(node, PrevOppMoveExpr)
        return state.oppmoves[end]
    elseif isa(node, IncExpr)
        return (1 + ({*} ~ eval_kern(node.expr, state))) % 3
    elseif isa(node, DecExpr)
        return (2 + ({*} ~ eval_kern(node.expr, state))) % 3
    elseif isa(node, PrevOutcome) 
        return (3 + state.moves[end] - state.oppmoves[end]) % 3
    elseif isa(node, RandomMoveFixed)
        return ({:rand_move} ~ cat0([ 
            node.gamma1, 
            node.gamma2, 
            node.gamma3]))
    elseif isa(node, RandomTransitionFixed)
        rand_transition ~ cat0([ 
            node.gamma1, 
            node.gamma2, 
            node.gamma3])
        return (state.moves[end]+rand_transition)%3
    elseif isa(node, RandomCorrPrevMove)
        return ({:rand_move} ~ cat0(normalize([eval_kern(CountMove(0), state),
            eval_kern(CountMove(1), state),
            eval_kern(CountMove(2), state)])))
    elseif isa(node, RandomInvCorrPrevMove)
        return ({:rand_move} ~ cat0(normalize([1/(eval_kern(CountMove(0), state)+EPS),
            1/(eval_kern(CountMove(1), state)+EPS),
            1/(eval_kern(CountMove(2), state)+EPS)])))
    elseif isa(node, RandomCorrPrevTransition)
        rand_transition ~ cat0(normalize([eval_kern(CountTransition(0), state),
            eval_kern(CountTransition(1), state),
            eval_kern(CountTransition(2), state)]))
            return (state.moves[end]+rand_transition)%3
    elseif isa(node, RandomInvCorrPrevTransition)
        return ({:rand_move} ~ cat0(normalize([1/(eval_kern(CountTransition(0), state)+EPS),
            1/(eval_kern(CountTransition(1), state)+EPS),
            1/(eval_kern(CountTransition(2), state)+EPS)])))
    elseif isa(node, Equal)
        return eval_kern(node.expr1, state) == eval_kern(node.expr2, state)
    elseif isa(node, Lt)
        return eval_kern(node.expr1, state) < eval_kern(node.expr2, state)
    elseif isa(node, Leq)
        return eval_kern(node.expr1, state) <= eval_kern(node.expr2, state)
    elseif isa(node, Flip)
        return {:flip} ~ bernoulli(node.gamma_bias)
    end
    error("$node_type not a valid type")
end