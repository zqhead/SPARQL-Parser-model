import sys
sys.path.append("..")
import SPARQLPARSER as SP

prepareQuery = '''
    PREFIX m: <http://rdf.freebase.com/ns/m/>
    PREFIX ns60: <http://rdf.freebase.com/relation/music/musical_group/member./music/group_membership/> 
    SELECT ?a
    WHERE {
       ?a ns60:member m:01vs4ff .
       ?a ns60:role m:064t9 .
    }
    '''



s = SP.sparqlParser(model="TransE")
s.resetQuery(model="TransR")
s.TransR_dataloader("..//Data//TransR//TransR_entity_vector.data",
                    "..//Data//TransR//TransR_relation_vector.data",
                    "..//Data//TransR//TransR_relation_matrix.data",
                    "..//Data_preprocess//FB15k//FB15k_test_triples.rdf")

s.select_parse(prepareQuery)
hit,hit10, wmr, mmr = s.getHitTargetAndMeanRank()
print(hit,hit10, wmr, mmr)
result = s.getRecommendSolution()
for item in result:
    print(item)
answer, answer_rank = s.getCurrentSolution()
for key in answer_rank.keys():
    print(key, answer_rank[key])