# REPORT

## Implementation choices preprocessing for both cbow and skipgram

I noticed that the sentences were looking like this:
[   1 1442   19   13  181   22    3    3    2    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0

    ... ]
    so most of the context and targets were all zeroes.

 - I threw away samples with context windows that were all 0s, even if the target word itself wasn't pad. this amounted to what I estimated to be a very large portion of the samples, given what I observed from printing the samples. it seemed that the recommended_pad number was much higher than the length of most of the sentences- this would make sense, given it's just a max over all lengths.
 - I did an 80/20 split using random numbers for splitting into bins during the processing. I chose to split based on sample and not sentence, so a sample in the train could come from the exact same sentence and even a neighboring word as the train set. the model might have performed worse had I not done this.

## CBOW:
### Implementation choices
 - I set embed dim to 128. 
 - i specified a padding index, not sure how this affects performance, but it essentially ensures that the padding index is the same as it is in embed space; basically no projection for the pad token
 - I chose a batch size of 256 and dropped the last samples that didn't fit in the last batch- the program crashed when I didn't do this. 256 was an arbitrary power of two
 - I set the context window to 1, i.e. 1 word on either side of the target word
### In vitro
The performance was measured by accuracy. this is a direct measure of the proportion of times the model produced the target word which fits within the context. 
### In vivo
Evaluating downstream performance on analogy task over 1309 analogies...
...Total performance across all 1309 analogies: 0.0008 (Exact); 0.0051 (MRR); 198 (MR)
...Analogy performance across 969 "sem" relation types: 0.0000 (Exact); 0.0039 (MRR); 256 (MR)
	relation	N	exact	MRR	MR
	capitals	1	0.0000	0.0000	inf
	binary_gender	12	0.0000	0.0427	23
	antonym	54	0.0000	0.0054	187
	member	4	0.0000	0.0058	171
	hypernomy	542	0.0000	0.0043	235
	similar	117	0.0000	0.0020	491
	partof	29	0.0000	0.0010	1030
	instanceof	9	0.0000	0.0008	1281
	derivedfrom	133	0.0000	0.0020	490
	hascontext	32	0.0000	0.0005	1836
	relatedto	10	0.0000	0.0015	674
	attributeof	11	0.0000	0.0023	428
	causes	6	0.0000	0.0050	201
	entails	9	0.0000	0.0025	400
...Analogy performance across 340 "syn" relation types: 0.0029 (Exact); 0.0083 (MRR); 120 (MR)
	relation	N	exact	MRR	MR
	adj_adv	22	0.0000	0.0013	767
	comparative	7	0.0000	0.0016	634
	superlative	3	0.0000	0.0008	1319
	present_participle	62	0.0000	0.0079	127
	denonym	2	0.0000	0.0030	332
	past_tense	64	0.0156	0.0224	45
	plural_nouns	107	0.0000	0.0065	153
	plural_verbs	73	0.0000	0.0021	466

### Metrics:
 - Exact: this reflects how often the average analogy task produced *exactly* the correct word out of the 3000 vocabulary tokens. It got the correct word at a rate of 0.0008 which is more than twice what it would do randomly. as I will repeat, comparing to random is the only way to get insight into these metrics, and a better way to assess the model's performance would be a ratio of the metric divided by the expected random performance. with large vocab size, exact underestimates the model's performance because it's only looking at the first word produced.
 - MRR: MRR and MR are similar values. MRR is assessed this way: the model produces a ranked list of how confident it thinks each word in the vocabulary fits into the slot of the analogy. The correct word is found, and its place on the list (Rank) is divided by 1. it's the average (1/Rank) for each word tested. This metric assesses cases where the model didn't identify the correct word as its first guess, but maybe close to first. It also is somewhat deceptively low, as the model uniformly sampling from the vocab would yield 0.000333... MRR. I think a more insightful way to evaluate this would be to divide the true MRR by the "random" MRR. in this case it would be 15.3. this indicates that the reciprocal rank is 15.3 times higher than random. again, with large vocabulary size, MRR underestimates the model's performance, but is more accurate than "Exact".
 - MR is the average rank in the results list itself. this value is strangely high and hard to interpret without the vocab size. So the average word produced was 198 down on the list of words produced. I would interpret this as the model has a fairly good placement of the words in embed space.
 - overall, if the vocabulary in embed space around the correct word is quite dense, each metric would underestimate the model's performance- i.e. if the cosine distance to many target words is very small and similar. The ranking the model produces itself would hold less meaning.

### Code analysis:
 - In the preprocessing code, it joins all lines in a book and asks the spacy language library to identify sentences. while it makes this call before removing punctuation, if it incorrectly identifies the start and end of a sentence, then in the dataset the start and end tokens will be put in the wrong places.
 - 


## Bonus, SKIPGRAM:
### Implementation choices
 - I used the same train/test split method as for cbow.
 - I also set the context window to 1, and used the same embed dim

### In vitro
 - I needed to decide whether to evaluate the in-vitro accuracy with the actual tokens or the one-hot-encoded, 3000 more dimensional versions of the target words. I decided to use the one-hot encoded versions and pytorch's bitwise operators to compare the values at each index.
 - I got an accuracy score of 0.272372... in vitro for the task. this metric seems pretty high, but can be explained by the fact that my training set is very large and the model learned the features of the train set well by the time this was assessed.
### In vivo
 - after the model was trained, this was how it performed on the analogy tasks:
Evaluating downstream performance on analogy task over 1309 analogies...
...Total performance across all 1309 analogies: 0.0000 (Exact); 0.0016 (MRR); 628 (MR)
...Analogy performance across 969 "sem" relation types: 0.0000 (Exact); 0.0016 (MRR); 610 (MR)
	relation	N	exact	MRR	MR
	capitals	1	0.0000	0.0027	364
	binary_gender	12	0.0000	0.0005	2010
	antonym	54	0.0000	0.0008	1309
	member	4	0.0000	0.0029	344
	hypernomy	542	0.0000	0.0017	588
	similar	117	0.0000	0.0019	537
	partof	29	0.0000	0.0035	285
	instanceof	9	0.0000	0.0003	3429
	derivedfrom	133	0.0000	0.0017	572
	hascontext	32	0.0000	0.0005	1910
	relatedto	10	0.0000	0.0011	896
	attributeof	11	0.0000	0.0002	4279
	causes	6	0.0000	0.0012	803
	entails	9	0.0000	0.0012	829
...Analogy performance across 340 "syn" relation types: 0.0000 (Exact); 0.0015 (MRR); 685 (MR)
	relation	N	exact	MRR	MR
	adj_adv	22	0.0000	0.0021	480
	comparative	7	0.0000	0.0000	inf
	superlative	3	0.0000	0.0000	inf
	present_participle	62	0.0000	0.0011	915
	denonym	2	0.0000	0.0006	1808
	past_tense	64	0.0000	0.0030	335
	plural_nouns	107	0.0000	0.0011	930
	plural_verbs	73	0.0000	0.0010	962

### Metrics:
 - Exact: this reflects how often the average analogy task produced *exactly* the correct word out of the 3000 vocabulary tokens. because of the high number of trials, the model might have gotten it exactly right once or twice, but is not reflected in the floating portion of the printed value. In any case, this metric with a large vocabulary size would underestimate the performance of the model.
 - MRR: MRR and MR are similar values. MRR is assessed this way: the model produces a ranked list of how confident it thinks each word in the vocabulary fits into the slot of the analogy. The correct word is found, and its place on the list (Rank) is divided by 1. it's the average (1/Rank) for each word tested. This metric assesses cases where the model didn't identify the correct word as its first guess, but maybe close to first. It also is somewhat deceptively low, as the model uniformly sampling from the vocab would yield 0.000333... MRR. I think a more insightful way to evaluate this would be to divide the true MRR by the "random" MRR. in this case it would be 4.8. again, with large vocabulary size, MRR underestimates the model's performance, but is more accurate than "Exact".
 - MR is the average rank in the results list itself. this value is strangely high and hard to interpret without the vocab size. It seems that the correct word being 685 down on the list is rather poor performance, but still better than average.
 - the metrics were worse for skipgram than cbow- perhaps this was because the one-hot encoding passed into the BCELoss produced very small gradients due to the sparsity of the tensors. if I were to rerun it again, I would probably increase the learning rate from its default.

