So you can call .to_pos_tensor() to get all of the positions of the notes
you can also call .to_tensor() to get a tensor with the thing
you can also call .vocab to get a vocab object

So typically the models have > 10 layers, the ones i saw had 12 and 16. Lets start with 8
set the dimension of the model to 512
the number of heads was usually 8, so the dimension of head head is 64

self.pos_enc = nn.Embedding(context_input_length, d_model) if learned positions else PositionalEncoding(d_model)
we usually have a context input length of 150, which is essentially the size of the context vector
vocab_sz = len(data.vocab.itos)

for the encoding we can use emb = torch.nn.embedding. All passing somethign through this guy does is multiply him by
emb.weight, so emb(a) == a@emb.weight. To reverse this  we can just multiply the final embeding f_emb by the transpose
(model(a@emb.weight)@w.T).argmax(axis = 1)
for training we would use softmax instead of argmax
perhaps use logsoftmax and train with nll loss nlloss(logprobpreds,singular number)
crossentropy(input,target_as_number,label_smoothing=.0)
to sample you can use torch.multinomial(probs,1,True)

I trained on layers of 2, 6, and 12. 12 never converged, 6 did, but not to a good minimum, 2 did very well.


# ideas
we could do that thing where label smoothen