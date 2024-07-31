# RPS

[![Build Status](https://github.com/mercush/RPS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mercush/RPS.jl/actions/workflows/CI.yml?query=branch%3Amain)




Op -> MakeMove(MoveType) | If(Bool) (Op) Else (Op)

MoveType -> RawMove | PrevMove | PrevOppMove | Inc(MoveType) | Dec(MoveType) | Random

Expr -> Z3 | Int | CountMove | CountOppMove | CountTransition | CountOppTransition | PrevMoveExpr | PrevOppMoveExpr | IncExpr(Expr) | DecExpr(Expr) | PrevOutcome

Random -> RandomMoveFixed | RandomTransitionFixed | RandomCorrPrevMove | RandomInvCorrPrevMove | RandomCorrPrevTransition | RandomInvCorrPrevTransition

Bool -> Equals(Expr, Expr) | Lt(Expr, Expr) | Leq(Expr, Expr) | Flip





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




Most of the time, get tree with bad score. Sometimes you get something with a good score. Random move with high invtemp that sets a preference against opponent moves.
Possibly could be improved with a better inference algorithm. We could try adding swap moves and attach/detach.
Possibly could be improved if we hard code primitives involving random moves. 

Increasing the number of rejuvenation moves doesn't appear to improve inference results by much.
Increase the weight of random moves.
Increase the expressivity of random moves (choice over moves or transition).
Make Random moves more structured

Why might it be preferring random moves over conditional operations with softmax noise? Possibly because CountTransition, CountMove, etc. are important. 

See proposals that are being made.

Posteriors for small sample sizes are much less concentrated.
Maybe proposal isn't changing random subtree.