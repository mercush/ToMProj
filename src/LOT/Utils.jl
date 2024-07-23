function normalize(v::Vector)
    return v/sum(v)
end

function softmax(v::Vector)
    v = exp.(v)
    v /= sum(v)
    return v
end

function new_state(prev_state, new_move, new_oppmove)
    moves = fill(new_move, 4)
    oppmoves = fill(new_oppmove, 4)
    for i=1:3
        moves[i] = prev_state.moves[i+1]
        oppmoves[i] = prev_state.oppmoves[i+1]
    end
    return State(moves, oppmoves)
end

noisy_one_hot = (invtemp, m) -> [i == m ? invtemp : 0 for i=0:2]

@dist cat0(wt) = categorical(wt)-1