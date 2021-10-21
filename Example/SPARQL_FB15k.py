import sys
sys.path.append("..")
import SPARQLPARSER as SP
import time
# Q3
# prepareQuery = '''
#     PREFIX m: <http://rdf.freebase.com/ns/m/>
#     PREFIX ns89: <http://rdf.freebase.com/relation/music/musical_group/member./music/group_membership/>
#     PREFIX ns35: <http://rdf.freebase.com/relation/music/artist/>
#     SELECT ?a
#     WHERE {
#        ?a ns35:genre m:064t9 .
#     }
#     '''

# # # Q5
prepareQuery = '''
    PREFIX m: <http://rdf.freebase.com/ns/m/>
    PREFIX ns8: <http://rdf.freebase.com/relation/award/award_nominee/award_nominations./award/award_nomination/>
    PREFIX ns18: <http://rdf.freebase.com/relation/influence/influence_node/>
    SELECT ?a
    WHERE {
        ?a ns8:award m:01by1l .
        ?a ns18:influenced m:046lt .
    }
    '''

# # Q6
# prepareQuery = '''
#     PREFIX m: <http://rdf.freebase.com/ns/m/>
#     PREFIX ns89: <http://rdf.freebase.com/relation/music/musical_group/member./music/group_membership/>
#     SELECT ?a
#     WHERE {
#        ?a ns89:member m:01vs4ff .
#        ?a ns89:role m:05148p4 .
#     }
#     '''


# # Q6
# prepareQuery = '''
#     PREFIX m: <http://rdf.freebase.com/ns/m/>
#     PREFIX ns25: <http://rdf.freebase.com/relation/award/award_winner/awards_won./award/award_honor/>
#     PREFIX ns125: <http://rdf.freebase.com/relation/award/hall_of_fame_inductee/hall_of_fame_inductions./award/hall_of_fame_induction/>
#     PREFIX ns274: <http://rdf.freebase.com/relation/base/activism/activist/>
#     PREFIX ns35: <http://rdf.freebase.com/relation/music/artist/>
#     SELECT ?a
#     WHERE {
#         ?a ns25:ceremony m:01mh_q .
#         ?a ns125:hall_of_fame m:0g2c8 .
#         ?a ns274:area_of_activism m:097s4 .
#         ?a ns35:label m:03vv61 .
#     }
#     '''




# # Q7
# prepareQuery = '''
#     PREFIX m: <http://rdf.freebase.com/ns/m/>
#     PREFIX ns28: <http://rdf.freebase.com/relation/award/award_nominated_work/award_nominations./award/award_nomination/>
#     PREFIX ns5: <http://rdf.freebase.com/relation/film/film/>
#
#     SELECT ?a ?b
#     WHERE {
#         m:03bzyn4 ns28:award ?a .
#         m:03bzyn4 ns5:country ?b .
#     }
#     '''

# # Q8
# prepareQuery = '''
#     PREFIX m: <http://rdf.freebase.com/ns/m/>
#     PREFIX ns5: <http://rdf.freebase.com/relation/film/film/>
#     PREFIX ns13: <http://rdf.freebase.com/relation/film/film/other_crew./film/film_crew_gig/>
#
#     SELECT ?a ?b
#     WHERE {
#         m:0d6_s ns5:country ?a .
#         m:0d6_s ns13:film_crew_role ?b .
#     }
#     '''

#9
# prepareQuery = '''
#     PREFIX m: <http://rdf.freebase.com/ns/m/>
#     PREFIX ns25: <http://rdf.freebase.com/relation/award/award_winner/awards_won./award/award_honor/>
#     PREFIX ns125: <http://rdf.freebase.com/relation/award/hall_of_fame_inductee/hall_of_fame_inductions./award/hall_of_fame_induction/>
#     SELECT ?a ?b
#     WHERE {
#
#     }
#     '''

#10
# prepareQuery = '''
#     PREFIX m: <http://rdf.freebase.com/ns/m/>
#     PREFIX ns25: <http://rdf.freebase.com/relation/award/award_winner/awards_won./award/award_honor/>
#     PREFIX ns125: <http://rdf.freebase.com/relation/award/hall_of_fame_inductee/hall_of_fame_inductions./award/hall_of_fame_induction/>
#     SELECT ?a ?b
#     WHERE {
#         ?a ns25:ceremony m:01mh_q .
#         ?a ns125:hall_of_fame ?b .
#         m:04k05 ns125:hall_of_fame ?b .
#     }
#     '''



s = SP.sparqlParser(model="TransE")
s.TransE_dataloader("..//Data//TransE//TransE_entity_vector.data",
                    "..//Data//TransE//TransE_relation_vector.data",
                    "..//Data_preprocess//FB15k//FB15k_test_triples.rdf")
start = time.time()
s.select_parse(prepareQuery)
hit, hit_10, wmr, mmr = s.getHitTargetAndMeanRank()
end = time.time()
print("TranE cost time: %s" % (round((end - start), 3)))
print("hit, hit_10, wmr, mmr:",hit, hit_10, wmr, mmr)
result = s.getRecommendSolution()
for item in result:
    print("TransE Recommend answers:", item)
answer, answer_rank = s.getCurrentSolution()
for key in answer_rank.keys():
    print(key, answer_rank[key])

s.resetQuery(model="TransH")
s.TransH_dataloader("..//Data//TransH//TransH_entity_vector.data",
                    "..//Data//TransH//TransH_relation_norm_vector.data",
                    "..//Data//TransH//TransH_relation_hyper_vector.data",
                    "..//Data_preprocess//FB15k//FB15k_test_triples.rdf")


start = time.time()
s.select_parse(prepareQuery)
hit, hit_10, wmr, mmr = s.getHitTargetAndMeanRank()
end = time.time()
print("TranH cost time: %s" % (round((end - start), 3)))
print("hit, hit_10, wmr, mmr:",hit, hit_10, wmr, mmr)
result = s.getRecommendSolution()
for item in result:
    print(item)
answer, answer_rank = s.getCurrentSolution()
for key in answer_rank.keys():
    print(key, answer_rank[key])

s.resetQuery(model="TransR")
s.TransR_dataloader("..//Data//TransR//TransR_entity_vector.data",
                    "..//Data//TransR//TransR_relation_vector.data",
                    "..//Data//TransR//TransR_relation_matrix.data",
                    "..//Data_preprocess//FB15k//FB15k_test_triples.rdf")

start = time.time()
s.select_parse(prepareQuery)
hit, hit_10, wmr, mmr = s.getHitTargetAndMeanRank()
end = time.time()
print("TranR cost time: %s" % (round((end - start), 3)))
print("hit, hit_10, wmr, mmr:", hit, hit_10, wmr, mmr)
result = s.getRecommendSolution()
for item in result:
    print(item)
answer, answer_rank = s.getCurrentSolution()
for key in answer_rank.keys():
    print(key, answer_rank[key])