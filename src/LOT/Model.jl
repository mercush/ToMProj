using Gen
import DataFrames
using ProgressMeter

include("Proposals.jl")

@gen (static) function transition(t::Int, prev_state::State, kernel::Node, noise)::State
    new_move_proposed ~ eval_kern(kernel, prev_state)
    move ~ cat0(noisy_onehot(noise, new_move_proposed))
    opp_move ~ z3_dist()
    return new_state(prev_state, move, opp_move)
end

chain = Unfold(transition)

@gen (static) function unfold_model(T::Int)
    init_state ~ init_state(4)
    tree ~ pcfg()
    noise ~ beta(0.1,2)
    chain ~ chain(T, init_state, tree, noise)
    return (init_state, chain)
end

# @gen function transition_dynamic(t::Int, prev_statestate::State, kernel::Node)::State
#     out_move ~ eval_kern(kernel, prev_state)
#     oppmove ~ z3_dist()
#     new_moves = cat(prev_state.moves, [out_move])
#     new_oppmoves = cat(prev_state.oppmoves, [oppmove])
#     return State(new_moves[2:end], new_oppmoves[2:end])
# end

# chain_dynamic = Unfold(transition_dynamic)

# @gen function unfold_model_dynamic(T::Int)::Tuple{State,Gen.VectorTrace}
#     tree ~ pcfg()
#     init_state = State(init_move ~ cat0([1/3,1/3,1/3]), 
#         init_outcome ~ cat0([1/3,1/3,1/3]))
#     chain ~ chain_dynamic(T, init_state, tree)
#     return (init_state, chain)
# end