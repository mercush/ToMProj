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
struct Random <: IntermediateNode end

Base.size(node::Node) = node.size

# Op
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

# MoveType
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

# Expr
struct CustomZ3 <: TerminalNode
    param_z3::Int
    size::Int
end
CustomZ3(p) = CustomZ3(p, 1)

struct CustomInt <: TerminalNode
    param_int::Int
    size::Int
end
CustomInt(p) = CustomInt(p, 1)

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

struct PrevMoveExpr <: TerminalNode
    size::Int
end
PrevMoveExpr() = PrevMoveExpr(1)

struct PrevOppMoveExpr <: TerminalNode
    size::Int
end
PrevOppMoveExpr() = PrevOppMoveExpr(1)


struct IncExpr <: InternalNode
    expr::Node
    size::Int
end
IncExpr(e) = IncExpr(e, 1)

struct DecExpr <: InternalNode
    expr::Node
    size::Int
end
DecExpr(e) = DecExpr(e, 1)

struct PrevOutcome <: TerminalNode 
    size::Int
end
PrevOutcome() = PrevOutcome(1)

# Random
struct RandomMoveFixed <: TerminalNode 
    gamma1::Float64
    gamma2::Float64
    gamma3::Float64
    size::Int
end
RandomMoveFixed(c1, c2, c3) = RandomMoveFixed(c1, c2, c3, 1)

struct RandomTransitionFixed <: TerminalNode 
    gamma1::Float64
    gamma2::Float64
    gamma3::Float64
    size::Int
end
RandomTransitionFixed(c1, c2, c3) = RandomTransitionFixed(c1, c2, c3, 1)

struct RandomCorrPrevMove <: TerminalNode 
    size::Int
end
RandomCorrPrevMove() = RandomCorrPrevMove(1)

struct RandomInvCorrPrevMove <: TerminalNode 
    size::Int
end
RandomInvCorrPrevMove() = RandomInvCorrPrevMove(1)

struct RandomCorrPrevTransition <: TerminalNode 
    size::Int
end
RandomCorrPrevTransition() = RandomCorrPrevTransition(1)

struct RandomInvCorrPrevTransition <: TerminalNode 
    size::Int
end
RandomInvCorrPrevTransition() = RandomInvCorrPrevTransition(1)

# Bool
struct Equal <: InternalNode
    expr1::Node
    expr2::Node
    size::Int
end
Equal(e1, e2) = Equal(e1, e2, size(e1)+size(e2)+1)

struct Lt <: InternalNode
    expr1::Node
    expr2::Node
    size::Int
end
Lt(e1, e2) = Lt(e1, e2, size(e1)+size(e2)+1)

struct Leq <: InternalNode
    expr1::Node
    expr2::Node
    size::Int
end
Leq(e1, e2) = Leq(e1, e2, size(e1)+size(e2)+1)

struct Flip <: TerminalNode
    gamma_bias::Float64
    size::Int
end
Flip(c2) = Flip(c2, 1)

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