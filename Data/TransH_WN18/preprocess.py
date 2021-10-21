import sys
sys.path.append("..")
import Text2RDF as T2R

entIds = "..\\..\\WN18\\entity2id.txt"
relIds = "..\\..\\WN18\\relation2id.txt"
entData = "..\\..\\Data_preprocess\\WN18\\WN18_test_entity.data"
relData = "..\\..\\Data_preprocess\\WN18\\WN18_test_relation.data"

T2R.TransHembeddingNodeTranslate(entData,
                            "WN18_1epoch_TransH_pytorch_entity_50dim_batch4800",
                             entIds,
                             relData,
                            "WN18_1epoch_TransH_pytorch_norm_relations_50dim_batch4800" ,
                             relIds,
                           "WN18_1epoch_TransH_pytorch_hyper_relations_50dim_batch4800")
