function eval_kern(node, state)
    if isa(node, MakeMove)
        return eval_kern(node.move_type, state)
    elseif isa(node, If)
        op1 = eval_kern(node.op1, state)
        op2 = eval_kern(node.op2, state)
        bool = eval_kern(node.bool, state)
        return bool*op1 + (1-bool)*op2
    elseif isa(node, RawMove)
        return onehot(node.param_z3)
    elseif isa(node, PrevMove)
        return onehot(state.moves[end])
    elseif isa(node, PrevOppMove)
        return onehot(state.oppmoves[end])
    elseif isa(node, Inc)
        dist = zeros(3)
        input = eval_kern(node.move_type, state)
        for i=0:2
            dist[((i+1)%3)+1] = input[i+1]
        end
        return dist
    elseif isa(node, Dec)
        dist = zeros(3)
        input = eval_kern(node.move_type, state)
        for i=0:2
            dist[((i+2)%3)+1] = input[i+1]
        end
        return dist
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
        return (1 + eval_kern(node.expr, state)) % 3
    elseif isa(node, DecExpr)
        return (2 + eval_kern(node.expr, state)) % 3
    elseif isa(node, PrevOutcome) 
        return (3 + state.moves[end] - state.oppmoves[end]) % 3
    elseif isa(node, RandomMoveFixed)
        return [node.gamma1, 
            node.gamma2, 
            node.gamma3]
    elseif isa(node, RandomTransitionFixed)
        dist = zeros(3)
        dist[state.moves[end]+1] = node.gamma1
        dist[((1+state.moves[end])%3)+1] = node.gamma2
        dist[((2+state.moves[end])%3)+1] = node.gamma3
        return dist
    elseif isa(node, RandomCorrPrevMove)
        dist = [eval_kern(CountMove(0), state),
        eval_kern(CountMove(1), state),
        eval_kern(CountMove(2), state)]
        return normalize(dist)
    elseif isa(node, RandomInvCorrPrevMove)
        dist = [1/(eval_kern(CountMove(0), state)+EPS),
        1/(eval_kern(CountMove(1), state)+EPS),
        1/(eval_kern(CountMove(2), state)+EPS)]
        return normalize(dist)
    elseif isa(node, RandomCorrPrevTransition)
        dist = zeros(3)
        for i=0:2
            dist[((i+state.moves[end])%3)+1] = eval_kern(CountTransition(i), state)
        end
        return normalize(dist)
    elseif isa(node, RandomInvCorrPrevTransition)
        dist = zeros(3)
        for i=0:2
            dist[((i+state.moves[end])%3)+1] = 1/(eval_kern(CountTransition(i), state)+EPS)
        end
        return normalize(dist)
    elseif isa(node, Equal)
        return eval_kern(node.expr1, state) == eval_kern(node.expr2, state)
    elseif isa(node, Lt)
        return eval_kern(node.expr1, state) < eval_kern(node.expr2, state)
    elseif isa(node, Leq)
        return eval_kern(node.expr1, state) <= eval_kern(node.expr2, state)
    elseif isa(node, Flip)
        return node.gamma_bias
    end
    error("$node_type not a valid type")
end

function best_pred(state)
    argmax(get_score, state.traces)    
end

function next_move_pred(state)
    weights = zeros(3)
    partition = 0
    for (idx, tr) in enumerate(state.traces)
        if length(tr[:chain]) == 0
            prev_state = tr[:init_state]
        else
            prev_state = tr[:chain][end]
        end
        wt = state.log_weights[idx]
        weights += exp(wt)*eval_kern(tr[:tree], prev_state)
        partition += exp(wt)
    end
    return weights/partition
end