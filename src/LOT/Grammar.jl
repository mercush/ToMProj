using Gen

include("Utils.jl")

struct State 
    moves::Vector{Int}
    oppmoves::Vector{Int}
end

abstract type Node end
abstract type InternalNode <: Node end
abstract type TerminalNode <: Node end
abstract type IntermediateNode <: Node end

struct Op <: IntermediateNode end
struct MoveType <: IntermediateNode end
struct Expr <: IntermediateNode end
struct CustomBool <: IntermediateNode end
struct BoolInput <: IntermediateNode end

Base.size(node::Node) = node.size

struct MakeMove <: InternalNode
    move_type::Node
    size::Int
end
MakeMove(m) = MakeMove(m, size(m)+1)

struct If <: InternalNode
    bool::Node
    op1::Node
    op2::Node
    size::Int
end
If(b, o1, o2) = If(b, o1, o2, size(b)+size(o1)+size(o2)+1)

struct RawMove <: TerminalNode
    param_z3::Int
    size::Int
end
RawMove(p) = RawMove(p, 1)

struct PrevMove <: TerminalNode
    size::Int
end
PrevMove() = PrevMove(1)

struct PrevOppMove <: TerminalNode
    size::Int
end
PrevOppMove() = PrevOppMove(1)

struct Inc <: InternalNode
    move_type::Node
    size::Int
end
Inc(m) = Inc(m, size(m)+1)

struct Dec <: InternalNode
    move_type::Node
    size::Int
end
Dec(m) = Dec(m, size(m)+1)

struct Random <: InternalNode
    expr1::Node
    expr2::Node
    expr3::Node
    certainty::Float64
    size::Int
end
Random(e1, e2, e3, c) = Random(e1, e2, e3, c, size(e1)+size(e2)+size(e3)+1)

struct CustomInt <: TerminalNode
    param_int::Int
    size::Int
end
CustomInt(p) = CustomInt(p, 1)

struct CustomZ3 <: TerminalNode
    param_z3::Int
    size::Int
end
CustomZ3(p) = CustomZ3(p, 1)

struct CountMove <: TerminalNode
    param_z3::Int
    size::Int
end
CountMove(p) = CountMove(p, 1)

struct CountOppMove <: TerminalNode
    param_z3::Int
    size::Int
end
CountOppMove(p) = CountOppMove(p, 1)

struct CountTransition <: TerminalNode
    param_z3::Int
    size::Int
end
CountTransition(p) = CountTransition(p, 1)

struct CountOppTransition <: TerminalNode
    param_z3::Int
    size::Int
end
CountOppTransition(p) = CountOppTransition(p, 1)

struct PrevOutcome <: TerminalNode 
    size::Int
end
PrevOutcome() = PrevOutcome(1)

struct Equal <: InternalNode
    bool_input1::Node
    bool_input2::Node
    size::Int
end
Equal(b1, b2) = Equal(b1, b2, size(b1)+size(b2)+1)

struct Lt <: InternalNode
    bool_input1::Node
    bool_input2::Node
    size::Int
end
Lt(b1, b2) = Lt(b1, b2, size(b1)+size(b2)+1)

struct Leq <: InternalNode
    bool_input1::Node
    bool_input2::Node
    size::Int
end
Leq(b1, b2) = Leq(b1, b2, size(b1)+size(b2)+1)

@gen function eval_kern(node, state)
    if isa(node, MakeMove)
        return eval_kern(node.move_type, state)
    elseif isa(node, If)
        return eval_kern(node.bool, state) ? eval_kern(node.op1, state) : eval_kern(node.op2, state)
    elseif isa(node, RawMove)
        return node.param_z3
    elseif isa(node, Dec)
        return (2 + eval_kern(node.move_type, state)) % 3
    elseif isa(node, Inc)
        return (1 + eval_kern(node.move_type, state)) % 3
    elseif isa(node, PrevMove)
        return state.moves[end]
    elseif isa(node, PrevOppMove)
        return state.oppmoves[end]
    elseif isa(node, CustomInt)
        return node.param_int
    elseif isa(node, CustomZ3)
        return node.param_z3
    elseif isa(node, CountMove)
        return sum(state.moves .== node.param_z3)
    elseif isa(node, CountOppMove)
        return sum(state.oppmoves .== node.param_z3)
    elseif isa(node, CountTransition)
        return sum(((3 .+ state.moves[2:end] .- state.moves[1:end-1]) .% 3) == node.param_z3)
    elseif isa(node, CountOppTransition)
        return sum(((3 .+ state.oppmoves[2:end] .- state.oppmoves[1:end-1]) .% 3) == node.param_z3)
    elseif isa(node, PrevOutcome) 
        return (3 + state.moves[end] - state.oppmoves[end]) % 3
    elseif isa(node, Random)
        return ({:rand_move} ~ cat0(
            softmax([ 
                node.certainty*eval_kern(node.expr1, state), 
                node.certainty*eval_kern(node.expr2, state), 
                node.certainty*eval_kern(node.expr3, state)])))
    elseif isa(node, Equal)
        return eval_kern(node.bool_input1, state) == eval_kern(node.bool_input2, state)
    elseif isa(node, Lt)
        return eval_kern(node.bool_input1, state) < eval_kern(node.bool_input2, state)
    elseif isa(node, Leq)
        return eval_kern(node.bool_input1, state) <= eval_kern(node.bool_input2, state)
    end
    error("$node_type not a valid type")
end