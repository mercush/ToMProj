using Gen
import DataFrames
using ProgressMeter

include("Proposals.jl")
include("Eval.jl")

@gen (static) function transition(
    t::Int, 
    prev_state::State, 
    kernel::Node, 
    noise)::State
    move_dist = eval_kern(kernel, prev_state)
    move ~ cat0(noise*[1/3,1/3,1/3]+(1-noise)*move_dist)
    opp_move ~ z3_dist()
    return new_state(prev_state, move, opp_move)
end

chain = Unfold(transition)

@gen (static) function unfold_model(T::Int)
    init_state ~ init_state(4)
    tree ~ pcfg()
    noise ~ beta(0.1,10)
    chain ~ chain(T, init_state, tree, noise)
    return (init_state, chain)
end