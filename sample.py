from POSTagger.HMMPOSTagger import HMMPOSTagger

hmm_brill = HMMPOSTagger('Data/train.txt')
hmm_brill.train()
prob_brill, labels_brill, precision_brill = hmm_brill.tagger('Data/test.txt')

hmm_swj = HMMPOSTagger('Data/train.txt')
hmm_swj.setLabelType('swj')
hmm_swj.train()
prob_swj, labels_swj, precision_swj = hmm_swj.tagger('Data/test.txt')

print(precision_brill, precision_swj)
