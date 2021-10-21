import sys
sys.path.append("..")
import Text2RDF as T2R

file2 = "..\\..\\FB15k\\entity2id.txt"
file3 = "..\\..\\FB15k\\relation2id.txt"

T2R.TransRembeddingNodeTranslate("entity.data", "FB15k_torch_TransR_pytorch_entity_50dim_batch9600", file2, "relation.data",
                             "FB15k_torch_TransR_pytorch_reltion_50dim_batch9600" ,file3,  "FB15k_torch_TransR_pytorch_rel_matrix_50_50dim_batch9600")