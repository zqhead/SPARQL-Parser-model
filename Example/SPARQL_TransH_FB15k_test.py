import sys
sys.path.append("..")
import SPARQLPARSER as SP
import codecs
import numpy as np

prepareQuery = "PREFIX m: <http://rdf.freebase.com/ns/m/> SELECT ?a { "
s = SP.sparqlParser(model="TransH")

s.TransH_dataloader("..//Data//TransH//TransH_entity_vector.data",
                    "..//Data//TransH//TransH_relation_norm_vector.data",
                    "..//Data//TransH//TransH_relation_hyper_vector.data",
                    "..//Data_preprocess//FB15k//FB15k_test_triples.rdf")
with codecs.open("..///Data_preprocess//FB15k//FB15k_test_ntrips.nt", 'r') as f1:
    lines1 = f1.readlines()
    hit_target = 0
    hit_10 = 0
    wmr_sum = 0
    mmr_sum = 0
    num = 0
    for line in lines1:
        num += 1

        s.resetQuery(model="TransH")
        line = line.strip().split(' ')
        # if len(line) != 3:
        #     continue
        pr = np.random.random(1)[0]
        if pr > 0.5:
            line[0] = "?a"
        else:
            line[2] = "?a"
        query = prepareQuery + line[0] + " " + line[1] + " " + line[2] + " . }"
        s.select_parse(query)
        hit, hit10, wmr, mmr = s.getHitTargetAndMeanRank()
        if hit:
            hit_target += 1
        if hit10:
            hit_10 += 1
        wmr_sum += wmr
        mmr_sum += mmr
        print(num, line, str(hit_target / num), str(hit_10 / num), str(wmr_sum / num), str(mmr_sum / num))
