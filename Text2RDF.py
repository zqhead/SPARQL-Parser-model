import rdflib
import codecs
import numpy
from rdflib import Graph
from rdflib import Namespace
import sys

def text2NTriple(file, file2, file3, entityURI, relationURI, out_triple_nt, out_ent_data, out_rel_data):
    entities = {}
    relations = {}
    triple_list = []

    with codecs.open(file, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue
            head = "<" + entityURI + triple[0] + ">"
            tail = "<" + entityURI + triple[2] + ">"
            relation = "<" + relationURI + triple[1] + ">"
            triple_list.append([head, relation, tail])


    with codecs.open(file2, 'r') as f, codecs.open(file3, 'r') as f1:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            ent = "<" + entityURI + triple[0] + ">"
            entities[triple[0]] = ent


        content1 = f1.readlines()
        for line in content1:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            rel = "<" + relationURI + triple[0] + ">"
            relations[triple[0]] = rel

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entities), len(relations), len(triple_list)))

    with codecs.open(out_triple_nt, "w") as f1:

        for triple in triple_list:
            for e in triple:
                f1.write(e+' ')
            f1.write(".\n")

    with codecs.open(out_ent_data, "w") as f2:

        for k in entities.keys():
            f2.write(k+' '+entities[k])
            f2.write("\n")

    with codecs.open(out_rel_data, "w") as f3:

        for k in relations.keys():
            f3.write(k + ' ' +relations[k])
            f3.write("\n")



def NTriple2RDFGraph(input_file:str, namespace:dict ,output_file:str):
    g = Graph()
    g.parse(input_file, format=rdflib.util.guess_format(input_file))
    for prefix in namespace:
        n = Namespace(prefix)
        g.bind(namespace[prefix], n)
    g.serialize(output_file, format="turtle")

    print(g.serialize(format="turtle").decode("utf-8"))



def embeddingNodeTranslate(entity_file, entity_embedding ,entities_id, relation_file, relation_embedding, relations_id):
    entities = {}
    embeddings = {}
    entity_id = {}
    new_entity_embedding = {}

    # with codecs.open(entity_file, 'r') as f1, codecs.open(entity_embedding, 'r') as f2, codecs.open(entities_id, 'r') as f3:
    #     content = f1.readlines()
    #     embedding = f2.readlines()
    #     id = f3.readlines()
    #     for line in content:
    #         triple = line.strip().split(" ")
    #         if len(triple) != 2:
    #             continue
    #         triple[1] = triple[1].replace("<", "")
    #         triple[1] = triple[1].replace(">", "")
    #         entities[triple[0]] = triple[1]
    #
    #     for line in embedding:
    #         triple = line.strip().split("\t")
    #         if len(triple) != 2:
    #             continue
    #         embeddings[triple[0]] = triple[1]
    #
    #     for line in id:
    #         triple = line.strip().split("\t")
    #         if len(triple) != 2:
    #             continue
    #         entity_id[triple[0]] = triple[1]

    with codecs.open(entity_file, 'r') as f1, codecs.open(entity_embedding, 'r') as f2, codecs.open(entities_id, 'r') as f3:
        content = f1.readlines()
        embedding = f2.readlines()
        id = f3.readlines()
        for line in content:
            triple = line.strip().split(" ")
            if len(triple) != 2:
                continue
            triple[1] = triple[1].replace("<", "")
            triple[1] = triple[1].replace(">", "")
            entities[triple[0]] = triple[1]

        for line in embedding:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            embeddings[triple[0]] = triple[1]

        for line in id:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            entity_id[triple[0]] = triple[1]

    for item in entity_id.keys():
        new_entity_embedding[entities[item]] = embeddings[entity_id[item]]

    with codecs.open("TransE_entity_vector.data", "w") as f4:

        for k in new_entity_embedding.keys():
            f4.write(k + '\t' +new_entity_embedding[k])
            f4.write("\n")

    relations = {}
    relations_embeddings = {}
    relation_id = {}
    new_relations_embedding = {}

    with codecs.open(relation_file, 'r') as f1, codecs.open(relation_embedding, 'r') as f2, codecs.open(relations_id, 'r') as f3:
        content = f1.readlines()
        embedding = f2.readlines()
        id = f3.readlines()
        for line in content:
            triple = line.strip().split(" ")
            if len(triple) != 2:
                continue
            triple[1] = triple[1].replace("<", "")
            triple[1] = triple[1].replace(">", "")
            relations[triple[0]] = triple[1]

        for line in embedding:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relations_embeddings[triple[0]] = triple[1]

        for line in id:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relation_id[triple[0]] = triple[1]

    for item in relation_id.keys():
        new_relations_embedding[relations[item]] = relations_embeddings[relation_id[item]]

    with codecs.open("TransE_relation_vector.data", "w") as f4:

        for k in new_relations_embedding.keys():
            f4.write(k + '\t' +new_relations_embedding[k])
            f4.write("\n")

def TransHembeddingNodeTranslate(entity_file, entity_embedding ,entities_id, relation_file, relation_norm_embedding, relations_id, relation_hyper_embedding):
    entities = {}
    embeddings = {}
    entity_id = {}
    new_entity_embedding = {}

    with codecs.open(entity_file, 'r') as f1, codecs.open(entity_embedding, 'r') as f2, codecs.open(entities_id, 'r') as f3:
        content = f1.readlines()
        embedding = f2.readlines()
        id = f3.readlines()
        for line in content:
            triple = line.strip().split(" ")
            if len(triple) != 2:
                continue
            triple[1] = triple[1].replace("<", "")
            triple[1] = triple[1].replace(">", "")
            entities[triple[0]] = triple[1]

        for line in embedding:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            embeddings[triple[0]] = triple[1]

        for line in id:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            entity_id[triple[0]] = triple[1]

    for item in entity_id.keys():
        new_entity_embedding[entities[item]] = embeddings[entity_id[item]]

    with codecs.open("TransH_entity_vector.data", "w") as f4:

        for k in new_entity_embedding.keys():
            f4.write(k + '\t' +new_entity_embedding[k])
            f4.write("\n")

    relations = {}
    relations_norm_embeddings = {}
    relation_id = {}
    new_relations_norm_embedding = {}

    with codecs.open(relation_file, 'r') as f1, codecs.open(relation_norm_embedding, 'r') as f2, codecs.open(relations_id, 'r') as f3:
        content = f1.readlines()
        embedding = f2.readlines()
        id = f3.readlines()
        for line in content:
            triple = line.strip().split(" ")
            if len(triple) != 2:
                continue
            triple[1] = triple[1].replace("<", "")
            triple[1] = triple[1].replace(">", "")
            relations[triple[0]] = triple[1]

        for line in embedding:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relations_norm_embeddings[triple[0]] = triple[1]

        for line in id:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relation_id[triple[0]] = triple[1]

    for item in relation_id.keys():
        new_relations_norm_embedding[relations[item]] = relations_norm_embeddings[relation_id[item]]

    with codecs.open("TransH_relation_norm_vector.data", "w") as f4:

        for k in new_relations_norm_embedding.keys():
            f4.write(k + '\t' +new_relations_norm_embedding[k])
            f4.write("\n")

    relations = {}
    relations_hyper_embeddings = {}
    relation_id = {}
    new_relations_hyper_embedding = {}

    with codecs.open(relation_file, 'r') as f1, codecs.open(relation_hyper_embedding, 'r') as f2, codecs.open(
            relations_id, 'r') as f3:
        content = f1.readlines()
        embedding = f2.readlines()
        id = f3.readlines()
        for line in content:
            triple = line.strip().split(" ")
            if len(triple) != 2:
                continue
            triple[1] = triple[1].replace("<", "")
            triple[1] = triple[1].replace(">", "")
            relations[triple[0]] = triple[1]

        for line in embedding:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relations_hyper_embeddings[triple[0]] = triple[1]

        for line in id:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relation_id[triple[0]] = triple[1]

    for item in relation_id.keys():
        new_relations_hyper_embedding[relations[item]] = relations_hyper_embeddings[relation_id[item]]

    with codecs.open("TransH_relation_hyper_vector.data", "w") as f4:

        for k in new_relations_hyper_embedding.keys():
            f4.write(k + '\t' + new_relations_hyper_embedding[k])
            f4.write("\n")

def TransRembeddingNodeTranslate(entity_file, entity_embedding ,entities_id, relation_file, relation_embedding, relations_id, relation_matrix_embedding):
    entities = {}
    embeddings = {}
    entity_id = {}
    new_entity_embedding = {}

    with codecs.open(entity_file, 'r') as f1, codecs.open(entity_embedding, 'r') as f2, codecs.open(entities_id, 'r') as f3:
        content = f1.readlines()
        embedding = f2.readlines()
        id = f3.readlines()
        for line in content:
            triple = line.strip().split(" ")
            if len(triple) != 2:
                continue
            triple[1] = triple[1].replace("<", "")
            triple[1] = triple[1].replace(">", "")
            entities[triple[0]] = triple[1]

        for line in embedding:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            embeddings[triple[0]] = triple[1]

        for line in id:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            entity_id[triple[0]] = triple[1]

    for item in entities.keys():
        new_entity_embedding[entities[item]] = embeddings[entity_id[item]]

    with codecs.open("TransR_entity_vector.data", "w") as f4:

        for k in new_entity_embedding.keys():
            f4.write(k + '\t' +new_entity_embedding[k])
            f4.write("\n")

    relations = {}
    relations_embeddings = {}
    relation_id = {}
    new_relations_embedding = {}
    relations_matrix_embeddings = {}
    new_relations_matrix_embedding = {}

    with codecs.open(relation_file, 'r') as f1, codecs.open(relation_embedding, 'r') as f2, \
            codecs.open(relations_id, 'r') as f3, codecs.open(relation_matrix_embedding, 'r') as f4:
        content = f1.readlines()
        embedding = f2.readlines()
        id = f3.readlines()
        matrix = f4.readlines()
        for line in content:
            triple = line.strip().split(" ")
            if len(triple) != 2:
                continue
            triple[1] = triple[1].replace("<", "")
            triple[1] = triple[1].replace(">", "")
            relations[triple[0]] = triple[1]

        for line in embedding:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relations_embeddings[triple[0]] = triple[1]

        for line in id:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relation_id[triple[0]] = triple[1]

        for line in matrix:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relations_matrix_embeddings[triple[0]] = triple[1]

    for item in relations.keys():
        new_relations_embedding[relations[item]] = relations_embeddings[relation_id[item]]
        new_relations_matrix_embedding[relations[item]] = relations_matrix_embeddings[relation_id[item]]

    with codecs.open("TransR_relation_vector.data", "w") as f4:

        for k in new_relations_embedding.keys():
            f4.write(k + '\t' +new_relations_embedding[k])
            f4.write("\n")

    with codecs.open("TransR_relation_matrix.data", "w") as f4:

        for k in new_relations_matrix_embedding.keys():
            f4.write(k + '\t' + new_relations_matrix_embedding[k])
            f4.write("\n")

# file1 = "FB15k\\train.txt"
# file2 = "FB15k\\entity2id.txt"
# file3 = "FB15k\\relation2id.txt"
# file4 = "FB15k\\test.txt"
# NTriple2RDFGraph("ntrips.nt", "train_triples.rdf")

# embeddingNodeTranslate("entity.data", "entity_vector_50dim", file2, "relation.data", "relation_vector_50dim",file3)


# entity_set, relation_set, triple_list = text2NTriple(file4, "test_ntrips.nt", "test_entity.data", "test_relation.data")
# NTriple2RDFGraph("test_ntrips.nt", "test_triples.rdf")

# TransHembeddingNodeTranslate("entity.data", "TransH_entity_50dim_batch1207", file2, "relation.data",
#                              "TransH_relation_norm_50dim_batch1207" ,file3,  "TransH_relation_hyper_50dim_batch1207")
# TransRembeddingNodeTranslate("entity.data", "TransR_pytorch_entity_50dim_batch4831", file2, "relation.data",
#                              "TransR_pytorch_reltion_50dim_batch4831" ,file3,  "TransR_pytorch_rel_matrix_50dim_batch4831")