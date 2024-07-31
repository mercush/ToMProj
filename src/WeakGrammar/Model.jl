using Gen
import DataFrames
using ProgressMeter

include("Eval.jl")

kernel_types = [Move, Transition, Outcome, Plus, Times]
@dist choose_kernel_type() = kernel_types[categorical([0.2, 0.2, 0.2, 0.2, 0.2])];
@dist cat0(wt) = categorical(wt)-1

struct State 
    move::Int
    outcome::Int
end

@gen function model(moves_input::Matrix{Int64})::Kernel
    kernel = {:tree} ~ kernel_prior()
    for i=1:size(moves_input, 1)
        weights = [exp(eval_kern(kernel, m, moves_input[i,1], moves_input[i,2])) for m=0:2]
        weights = weights/sum(weights)
        {(:moves, i)} ~ cat0(weights)
    end
    return kernel
end;

@gen (static) function transition(t::Int, prev_state::State, kernel::Kernel)::State
    weights = [exp(eval_kern(kernel, m, prev_state.move, prev_state.outcome)) for m=0:2]
    weights = weights/sum(weights)
    move ~ cat0(weights)
    outcome ~ cat0([1/3,1/3,1/3])
    next_state = State(move, outcome)
    return next_state
end

chain = Unfold(transition)

@gen (static) function unfold_model(T::Int)::Tuple{State,Gen.VectorTrace}
    tree ~ kernel_prior()
    init_state = State(init_move ~ cat0([1/3,1/3,1/3]), 
        init_outcome ~ cat0([1/3,1/3,1/3]))
    chain ~ chain(T, init_state, tree)
    return (init_state, chain)
end

@gen function transition_dynamic(t::Int, prev_state::State, kernel::Kernel)::State
    weights = [exp(eval_kern(kernel, m, prev_state.move, prev_state.outcome)) for m=0:2]
    weights = weights/sum(weights)
    move ~ cat0(weights)
    outcome ~ cat0([1/3,1/3,1/3])
    next_state = State(move, outcome)
    return next_state
end

chain_dynamic = Unfold(transition_dynamic)

@gen function unfold_model_dynamic(T::Int)::Tuple{State,Gen.VectorTrace}
    tree ~ kernel_prior()
    init_state = State(init_move ~ cat0([1/3,1/3,1/3]), 
        init_outcome ~ cat0([1/3,1/3,1/3]))
    chain ~ chain_dynamic(T, init_state, tree)
    return (init_state, chain)
end