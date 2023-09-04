# Music Project
In this project I trained a Auto-Regressive transformer to generate musical scores. To do this I trained the transformer using next-token prediction over a dataset of more than 10,000 scores. The transformer was conditioned on the genre of any particular score. This was done to enable genre mixing in the inference stage.
## Analysis
While a full analysis is given in the paper, Some of our main results can be pictured here.

We generated a learned embedding for any particular genre. Pictured here is the covariance matrix for any particular genre (how similar any two genres are). Lighter colors indicate a higher similarity score. We can see that the model learned meaningful differences across genres. For example jazz and blues are relatively similar to each other in the embedding space, but different from most of the other genres. This is somewhat consistent to how we view blues and jazz in the real world.

## Examples
Examples of the output of the Music Transformer can be found in the examples_base and examples_cherry_picked. Where the cherry picked examples are some of the best output we heard come from the transformer, while the base exampes are just regular examples of music generation for each of the classes. Note, the musical genre interpolation example is found in the cherry picked folder. The Datasets and full models are not revealed here due to size restrictions. Also we'd like to thank music-autobot for the help converting midi files to readable format.
Note that each of the examples in the cherry picked folder have themes consistent with their genre. EDM has repetition of similar pitches with varying rhythm, Jazz has a lot of seventh and ninth chords, and Classical has more complicated melodic structures such as arpegios.
