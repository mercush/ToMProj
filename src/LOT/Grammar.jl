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
