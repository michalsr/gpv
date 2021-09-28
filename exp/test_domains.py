import os
import dbm
from nltk.corpus import wordnet
from collections import defaultdict
from nltk.corpus import wordnet as wn

# Loading the Wordnet domains.
domain2synsets = defaultdict(list)
synset2domains = defaultdict(list)
domains = set()
for i in open('wn-domains-3.2-20070223', 'r'):
    ssid, doms = i.strip().split('\t')

    #doms = doms.split()
    domains.add(doms)
print(len(domains))

# # Gets domains given synset.
# for ss in wn.all_synsets():
  
#     ssid = str(ss.offset).zfill(8) + "-" + ss.pos()

#     if synset2domains[ssid]: # not all synsets are in WordNet Domain.
#         print(ss, ssid, synset2domains[ssid])

# # Gets synsets given domain.        
# for dom in sorted(domain2synsets):
#     print(dom, domain2synsets[dom][:3])

# sn = wn.synsets('beaver')
# print(type(list(synset2domains.keys())[0]))
# for i in sn:
#     ssid = str(i.offset()).zfill(8) + "-" + i.pos()
#     print(ssid in synset2domains.keys())
# print(ssid)
# print(synset2domains[ssid])