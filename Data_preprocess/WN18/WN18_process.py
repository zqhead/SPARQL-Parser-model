import sys
sys.path.append("..")
import Text2RDF as T2R


file = "..\\..\\WN18\\wordnet-mlj12-test.txt"
entIds = "..\\..\\WN18\\entity2id.txt"
relIds = "..\\..\\WN18\\relation2id.txt"

entUri = "http://rdf.wordnet.com/ns/"
relUri = "http://rdf.wordnet.com/relation/"

T2R.text2NTriple(file,entIds,relIds,entUri,relUri, out_triple_nt="WN18_test_ntrips.nt",
                 out_ent_data="WN18_test_entity.data",
                 out_rel_data="WN18_test_relation.data")

namespace = {"http://rdf.wordnet.com/ns/": "m", "http://rdf.wordnet.com/relation/": "r"}

T2R.NTriple2RDFGraph(input_file="WN18_test_ntrips.nt", namespace=namespace, output_file="WN18_test_triples.rdf")
