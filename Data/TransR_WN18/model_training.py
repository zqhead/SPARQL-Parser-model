import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransR_Train_Test



file1 = "..\\..\\WN18\\wordnet-mlj12-train.txt"
file2 = "..\\..\\WN18\\entity2id.txt"
file3 = "..\\..\\WN18\\relation2id.txt"
file4 = "..\\..\\WN18\\wordnet-mlj12-valid.txt"
entity_set, relation_set, triple_list, valid_triple_list = TransR_Train_Test.dataloader(file1, file2, file3, file4)

# file5 = "WN18_torch_TransE_entity_50dim_batch9600"
# file6 = "WN18_torch_TransE_relation_50dim_batch9600"


transR = TransR_Train_Test.TransR_Training_Testing(entity_set, relation_set, triple_list, ent_dim=50, rel_dim = 50,
                                                   lr=0.001, margin=4.0, norm=1, C=1.0, valid_triples=valid_triple_list)
transR.data_initialise()
# transR.data_initialise(file5, file6)
transR.training_run(epochs=500, batch_size=4800, out_file_title="WN18_1epoch_")

