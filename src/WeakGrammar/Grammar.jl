import LinearAlgebra

include("../Utils.jl")

"""Node in a tree representing a kernel function"""
abstract type Kernel end
abstract type PrimitiveKernel <: Kernel end
abstract type CompositeKernel <: Kernel end

"""
    size(::Kernel)
Number of nodes in the tree describing this kernel.
"""
Base.size(::PrimitiveKernel) = 1
Base.size(node::CompositeKernel) = node.size

"""Move"""
struct Move <: PrimitiveKernel
    a::Real
    b::Real
    c::Real
end

function eval_kern(node::Move, move::Int64, prev_move::Int64, prev_outcome::Int64)
    if move == 0
        return node.a
    elseif move == 1
        return node.b
    else
        return node.c
    end
end

"""Transition"""
struct Transition <: PrimitiveKernel
    a::Real
    b::Real
    c::Real
end

function eval_kern(node::Transition, move::Int64, prev_move::Int64, prev_outcome::Int64)
    transition = (3+move-prev_move) % 3
    if transition == 0
        return node.a
    elseif transition == 1
        return node.b
    else
        return node.c
    end
end

"""Outcome"""
struct Outcome <: PrimitiveKernel
    a::Real
    b::Real
    c::Real
end

function eval_kern(node::Outcome, move::Int64, prev_move::Int64, prev_outcome::Int64)
    if prev_outcome == 0
        return node.a
    elseif prev_outcome == 1
        return node.b
    else
        return node.c
    end
end

"""Plus node"""
struct Plus <: CompositeKernel
    left::Kernel
    right::Kernel
    size::Int
end

Plus(left, right) = Plus(left, right, size(left) + size(right) + 1)

function eval_kern(node::Plus, move::Int64, prev_move::Int64, prev_outcome::Int64)::Real
    eval_kern(node.left, move, prev_move, prev_outcome) + eval_kern(node.right, move, prev_move, prev_outcome)
end

"""Times node"""
struct Times <: CompositeKernel
    left::Kernel
    right::Kernel
    size::Int
end

Times(left, right) = Times(left, right, size(left) + size(right) + 1)

function eval_kern(node::Times, move::Int64, prev_move::Int64, prev_outcome::Int64)::Real
    eval_kern(node.left, move, prev_move, prev_outcome) * eval_kern(node.right, move, prev_move, prev_outcome)
end

@gen function kernel_prior()::Kernel
    kernel_type ~ choose_kernel_type()

    if in(kernel_type, [Plus, Times])
        return kernel_type({:left} ~ kernel_prior(), {:right} ~ kernel_prior())
    end
    
    a ~ normal(0,1)
    b ~ normal(0,1)
    c ~ normal(0,1)
    
    kernel_args = [a, b, c]
    return kernel_type(kernel_args...)
end;