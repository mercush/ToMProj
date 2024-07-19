# RPS

[![Build Status](https://github.com/mercush/RPS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mercush/RPS.jl/actions/workflows/CI.yml?query=branch%3Amain)

Op

Op -> MakeMove(MoveType) | If(Bool) (Op) Else (Op)
Expr -> Z3 | Int | CountMove | CountOppMove | CountTransition | CountOppTransition
MoveType -> RawMove | PrevMove | PrevOppMove | Inc(MoveType) | Dec(MoveType) | Random(Expr, Expr, Expr)
Bool -> Equals(BoolInput, BoolInput) | Lt(BoolInput, BoolInput) | Leq(BoolInput, BoolInput)
BoolInput -> Expr | MoveType | PrevOutcome

Do debugging by playing with a simulated model
Compare the performance of this model with ChatGPT
You can infer what the language model is doing

Regret matching doesn't have a theory of mind aspect

Why is what we're doing different from what algorithmic game theory people are doing? How are they different?
Connection to regret matching: we could incorporate game theoretic strategires in our setup. We would then be able to 
reliably beat a strategy like regret matching

Regret matching can only capture histogram of moves, we can capture something richer. 

Example where regret matching captures Nash and ours captures something more subtle.

Algorithmic GT: strategies that you can apply to any games. When you apply it to any specific game, it doesn't
capture the full nuance. For people, we don't have an optimal algorithmic GT, but we have some other way of playing
that works well for most environments. 

Show trajectories of rps on the screen. Will people do better.

Implementation:
    Use two language models and see what the ToM model infers.
    Read more on the algorithmic game theory stuff to get a better understanding of how they relate to each other

2 birds with one stone: despite having lots of data in the corpus, chatgpt doesn't learn to play in a good way. 
we use a bayesian method to show what they are doing.

show what chatgpt in an interpretable way (repetitive and stupid strategy despite learning from corpus)

AutoGP is so easy to implement because there is only one generation rule:


Expr -> Linear | SquaredExp | Periodic | Plus(Expr, Expr) | Times(Expr, Expr)