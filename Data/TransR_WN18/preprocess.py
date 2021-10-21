import sys
sys.path.append("..")
import Text2RDF as T2R

entIds = "..\\..\\WN18\\entity2id.txt"
relIds = "..\\..\\WN18\\relation2id.txt"
entData = "..\\..\\Data_preprocess\\WN18\\WN18_test_entity.data"
relData = "..\\..\\Data_preprocess\\WN18\\WN18_test_relation.data"

T2R.TransRembeddingNodeTranslate(entData,
                                 "WN18_1torch_TransR_pytorch_entity_50dim_batch4800",
                                 entIds,
                                 relData,
                                "WN18_1torch_TransR_pytorch_reltion_50dim_batch4800" ,
                                 relIds,
                                 "WN18_1torch_TransR_pytorch_rel_matrix_50_50dim_batch4800")