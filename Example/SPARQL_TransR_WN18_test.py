import sys
sys.path.append("..")
import SPARQLPARSER as SP
import codecs
import numpy as np

prepareQuery = "PREFIX m: <http://rdf.freebase.com/ns/m/> SELECT ?a { "
s = SP.sparqlParser(model="TransR")
s.TransR_dataloader("..//Data//TransR_WN18//TransR_entity_vector.data",
                    "..//Data//TransR_WN18//TransR_relation_vector.data",
                    "..//Data//TransR_WN18//TransR_relation_matrix.data",
                    "..//Data_preprocess//WN18//WN18_test_triples.rdf")
with codecs.open("..//Data_preprocess//WN18//WN18_test_ntrips.nt", 'r') as f1:
    lines1 = f1.readlines()
    hit_target = 0
    wmr_sum = 0
    mmr_sum = 0
    num = 0
    hit_10 = 0
    for line in lines1:
        num += 1

        s.resetQuery(model="TransR")
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
    print("hit_target: " + str(hit_target / len(lines1)))
    print("hit_10: " + str(hit_10 / len(lines1)))
    print("weight_mean_rank: " + str(wmr_sum / len(lines1)))
    print("mean_mean_rank: " + str(mmr_sum / len(lines1)))