import sys
sys.path.append("..")
import Text2RDF as T2R

entIds = "..\\..\\FB15k\\entity2id.txt"
relIds = "..\\..\\FB15k\\relation2id.txt"
entData = "..\\..\\Data_preprocess\\FB15k\\FB15k_test_entity.data"
relData = "..\\..\\Data_preprocess\\FB15k\\FB15k_test_relation.data"

T2R.embeddingNodeTranslate(entData,
                            "KB15k_torch_TransE_entity_50dim_batch9600",
                             entIds,
                             relData,
                            "KB15k_torch_TransE_relation_50dim_batch9600" ,
                             relIds,)