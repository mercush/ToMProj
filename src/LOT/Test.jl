import CSV
import JSON
include("Inference.jl")

# for i=1:10
#     global model, = generate(unfold_model, (2,))
#     global r, = generate(regen_random_subtree_randomness, (model,))
#     display(model[:tree])
#     display(r[:path])
#     display(r[:new_subtree])
#     global s, bc, wt = subtree_involution(model, get_choices(r), get_retval(r), ())
#     display(s[:tree])
# end

init_obs = choicemap()
n_particles = 10
state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)