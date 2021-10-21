import sys
sys.path.append("..")
import SPARQLPARSER as SP

# prepareQuery = '''
#     PREFIX m: <http://rdf.freebase.com/ns/m/>
#     PREFIX ns60: <http://rdf.freebase.com/relation/music/musical_group/member./music/group_membership/>
#     SELECT ?a
#     WHERE {
#        ?a ns60:member m:01vs4ff .
#        ?a ns60:role m:05148p4 .
#     }
#     '''

prepareQuery = '''
    PREFIX m: <http://rdf.freebase.com/ns/m/>
    PREFIX ns25: <http://rdf.freebase.com/relation/award/award_winner/awards_won./award/award_honor/>
    PREFIX ns125: <http://rdf.freebase.com/relation/award/hall_of_fame_inductee/hall_of_fame_inductions./award/hall_of_fame_induction/>
    SELECT ?a ?b
    WHERE {
        m:011vx3 ns25:ceremony ?a .
        m:011vx3 ns125:hall_of_fame ?b .
    }
    '''


s = SP.sparqlParser(model="TransE")
s.TransE_dataloader("..//Data//TransE//TransE_entity_vector.data",
                    "..//Data//TransE//TransE_relation_vector.data",
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