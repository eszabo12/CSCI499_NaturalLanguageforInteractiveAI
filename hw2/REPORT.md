# REPORT

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
    so most of the context and targets were literally all zeroes.

i threw away context windows that were all 0s.

I started with setting the max_per_book = 200 as a trial run on my cpu.

### CBOW:

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


### SKIPGRAM:
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

exact vs correct: 
exact:
MRR:
MR: the ratio of the num tested / inverse rank of correct. 
there's only 1 correct word, so if the denominator is 1/1 or 1/2 for all examples, then it will be really good. it's saying on average, which rank is the word which the model guesses?