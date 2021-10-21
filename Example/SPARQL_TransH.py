import sys
sys.path.append("..")
import SPARQLPARSER as SP

prepareQuery = '''
    PREFIX m: <http://rdf.freebase.com/ns/m/>
    PREFIX ns60: <http://rdf.freebase.com/relation/music/musical_group/member./music/group_membership/> 
    SELECT ?a
    WHERE {
       ?a ns60:member m:01vs4ff .
       ?a ns60:role m:05148p4 .
    }
    '''


s = SP.sparqlParser(model="TransH")
s.TransH_dataloader("..//Data//TransH//TransH_entity_vector.data",
                    "..//Data//TransH//TransH_relation_norm_vector.data",
                    "..//Data//TransH//TransH_relation_hyper_vector.data",
                    "..//Data_preprocess//FB15k//FB15k_test_triples.rdf")

s.select_parse(prepareQuery)
hit, hit_10, wmr, mmr = s.getHitTargetAndMeanRank()
print(hit, hit_10, wmr, mmr)
result = s.getRecommendSolution()
for item in result:
    print(item)
answer, answer_rank = s.getCurrentSolution()
for key in answer_rank.keys():
    print(key, answer_rank[key])