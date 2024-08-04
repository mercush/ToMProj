const EPS = 0.01

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

@dist cat0(wt) = categorical(wt)-1

@gen function dirichlet3(wt)
    gamma1 ~ gamma(wt[1],1)
    gamma2 ~ gamma(wt[2],1)
    gamma3 ~ gamma(wt[3],1)
    unnormalized_weights = [gamma1, gamma2, gamma3]
    return unnormalized_weights / sum(unnormalized_weights)
end

function noisy_onehot(noise, i)
    [i==j ? 1-noise : noise/2 for j=0:2]
end

function onehot(i)
    [i==j ? 1. : 0. for j=0:2]
end