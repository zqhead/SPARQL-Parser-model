import sys
sys.path.append("..")
import Text2RDF as T2R

entIds = "..\\..\\FB15k\\entity2id.txt"
relIds = "..\\..\\FB15k\\relation2id.txt"
entData = "..\\..\\Data_preprocess\\FB15k\\FB15k_test_entity.data"
relData = "..\\..\\Data_preprocess\\FB15k\\FB15k_test_relation.data"

T2R.TransHembeddingNodeTranslate(entData,
                            "FB15k_2epoch_TransH_pytorch_entity_200dim_batch9600",
                             entIds,
                             relData,
                            "FB15k_2epoch_TransH_pytorch_norm_relations_200dim_batch9600" ,
                             relIds,
                             "FB15k_2epoch_TransH_pytorch_hyper_relations_200dim_batch9600")

# T2R.TransHembeddingNodeTranslate(entData,
#                             "TransH_entity_50dim_batch1207",
#                              entIds,
#                              relData,
#                             "TransH_relation_norm_50dim_batch1207" ,
#                              relIds,
#                              "TransH_relation_hyper_50dim_batch1207")