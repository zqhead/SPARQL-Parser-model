import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransR_Train_Test


file1 = "..\\..\\FB15k\\freebase_mtr100_mte100-train.txt"
file2 = "..\\..\\FB15k\\entity2id.txt"
file3 = "..\\..\\FB15k\\relation2id.txt"
file4 = "..\\..\\FB15k\\freebase_mtr100_mte100-valid.txt"

# file5 = "FB15k_torch_TransE_entity_50dim_batch9600"
# file6 = "FB15k_torch_TransE_relation_50dim_batch9600"


entity_set, relation_set, triple_list, valid_triple_list = TransR_Train_Test.dataloader(file1, file2, file3, file4)


transR =  TransR_Train_Test.TransR_Training_Testing(entity_set, relation_set, triple_list, ent_dim=50, rel_dim=50, lr=0.001, margin=6.0,
                             norm=1, C=0.25,  valid_triples=valid_triple_list)
transR.data_initialise()
# transR.data_initialise(file5, file6)
transR.training_run(epochs=100, batch_size=4800, out_file_title="FB15k_1torch_")
