import sys
sys.path.append("..")
import SPARQLPARSER as SP
import time

#Q1

prepareQuery = '''
    PREFIX m: <http://rdf.wordnet.com/ns/>
    PREFIX r: <http://rdf.wordnet.com/relation/>
    SELECT ?a
    WHERE {
       m:00040962 r:_derivationally_related_form ?a .
       ?a r:_derivationally_related_form m:00031921 .
    }
    '''

# Q2
# prepareQuery = '''
#     PREFIX m: <http://rdf.wordnet.com/ns/>
#     PREFIX r: <http://rdf.wordnet.com/relation/>
#     SELECT ?a ?b
#     WHERE {
#             ?a r:_derivationally_related_form m:06806469 .
#             m:06806469 r:_hyponym ?b .
#     }
#     '''

s = SP.sparqlParser(model="TransE")
s.TransE_dataloader("..//Data//TransE_WN18//TransE_entity_vector.data",
                    "..//Data//TransE_WN18//TransE_relation_vector.data",
                    "..//Data_preprocess//WN18//WN18_test_triples.rdf")

start = time.time()
s.select_parse(prepareQuery)
hit, hit_10, wmr, mmr = s.getHitTargetAndMeanRank()
end = time.time()
print("TranE cost time: %s" % (round((end - start), 3)))
print(hit, hit_10, wmr, mmr)
result = s.getRecommendSolution()
for item in result:
    print(item)
answer, answer_rank = s.getCurrentSolution()
for key in answer_rank.keys():
    print(key, answer_rank[key])

s.resetQuery(model="TransH")
s.TransH_dataloader("..//Data//TransH_WN18//TransH_entity_vector.data",
                    "..//Data//TransH_WN18//TransH_relation_norm_vector.data",
                    "..//Data//TransH_WN18//TransH_relation_hyper_vector.data",
                    "..//Data_preprocess//WN18//WN18_test_triples.rdf")

start = time.time()
s.select_parse(prepareQuery)
hit, hit_10, wmr, mmr = s.getHitTargetAndMeanRank()
end = time.time()
print("TranH cost time: %s" % (round((end - start), 3)))
print(hit, hit_10, wmr, mmr)
result = s.getRecommendSolution()
for item in result:
    print(item)
answer, answer_rank = s.getCurrentSolution()
for key in answer_rank.keys():
    print(key, answer_rank[key])


s.resetQuery(model="TransR")
s.TransR_dataloader("..//Data//TransR_WN18//TransR_entity_vector.data",
                    "..//Data//TransR_WN18//TransR_relation_vector.data",
                    "..//Data//TransR_WN18//TransR_relation_matrix.data",
                    "..//Data_preprocess//WN18//WN18_test_triples.rdf")

start = time.time()
s.select_parse(prepareQuery)
hit, hit_10, wmr, mmr = s.getHitTargetAndMeanRank()
end = time.time()
print("TranR cost time: %s" % (round((end - start), 3)))
print(hit, hit_10, wmr, mmr)
result = s.getRecommendSolution()
for item in result:
    print(item)
answer, answer_rank = s.getCurrentSolution()
for key in answer_rank.keys():
    print(key, answer_rank[key])