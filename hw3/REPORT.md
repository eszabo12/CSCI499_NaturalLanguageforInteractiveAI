


hyperparameter choices:
I chose to do a relatively small embedding dimension of 31 because of the relatively small set of words in the vocabulary that are relevant, along with the sheer number of layers that the data needed to pass through.
I used a teacher forcing ratio of 0.5 arbitrarily.
I calculated the loss as an equally weighted sum of the action and target loss.


attention mechanism:
I used attention over the encoding inputs.

i just realized that i only used target_length number of instruction tokens in the encoding. so I skipped the rest. oops


results:
my code didnt finish running but at the 87th epoch it was val loss : 35.46153259277344 | val acc: 473.0 | val em: 0.0.
