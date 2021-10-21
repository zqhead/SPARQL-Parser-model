import sys
sys.path.append("..")
import Text2RDF as T2R


file = "..\\..\\FB15k\\freebase_mtr100_mte100-test.txt"
entIds = "..\\..\\FB15k\\entity2id.txt"
relIds = "..\\..\\FB15k\\relation2id.txt"

entUri = "http://rdf.freebase.com/ns"
relUri = "http://rdf.freebase.com/relation"

T2R.text2NTriple(file,entIds, relIds,entUri,relUri, out_triple_nt="FB15k_test_ntrips.nt",
                 out_ent_data="FB15k_test_entity.data",
                 out_rel_data="FB15k_test_relation.data")

namespace = {"http://rdf.freebase.com/ns/m/": "m", "http://rdf.freebase.com/relation/": "ns"}

T2R.NTriple2RDFGraph(input_file="FB15k_test_ntrips.nt", namespace=namespace, output_file="FB15k_test_triples.rdf")