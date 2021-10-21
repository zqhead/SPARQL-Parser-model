import rdflib
from rdflib import Graph
import codecs
import json
import numpy as np
import operator
import time
# import module.TransR as TransR
from module import TransR, TransE, TransH
import torch

class sparqlParser:
    def __init__(self, model="TransE"):
        self.query = """SELECT * WHERE{ ?s ?p ?o. }"""
        self.entity_dic = {}
        self.relation_dic = {}

        self.relation_norm_dic = {}
        self.relation_hyper_dic = {}

        self.rel_matrix_dic = {}

        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}

        self.g = Graph()

        self.triples = []
        self.prefix_dic = {}
        self.variables = []
        self.entities = {}
        self.relatioins = {}

        self.model = model
        self.result = {}
        self.query_answer = []
        self.query_answer_rank = {}
        self.isQueried = False



    def TransE_dataloader(self, entity_file, relation_file, rdf_graph):
        if self.model != "TransE":
            raise RuntimeError("Current model is not transE. self.model is %s" % self.model)
        print("embedding Data Loading...")
        self.entity_dic = {}
        self.relation_dic = {}

        with codecs.open(entity_file, 'r') as f1, codecs.open(relation_file, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            num = 0
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.entity2id[line[0]] = num
                self.id2entity[num] = line[0]
                self.entity_dic[num] = json.loads(line[1])
                num += 1

            num = 0
            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.relation2id[line[0]] = num
                self.id2relation[num] = line[0]
                self.relation_dic[num] = json.loads(line[1])
                num += 1


        self.module = TransE.TransE(len(self.entity_dic), len(self.relation_dic), len(self.entity_dic[0]),
                                    1.0, 1, 1.0)
        self.module.input_pre_transe(self.entity_dic, self.relation_dic)

        print("RDF Graph Loading...")
        self.g = Graph()
        self.g.parse(rdf_graph, format="turtle")

    def TransH_dataloader(self, entity_file, relation_norm_file, relation_hyper_file, rdf_graph):
        if self.model != "TransH":
            raise RuntimeError("Current model is not TransH. self.model is %s" % self.model)
        print("embedding Data Loading...")
        self.entity_dic = {}

        self.relation_norm_dic = {}
        self.relation_hyper_dic = {}
        with codecs.open(entity_file, 'r') as f1, codecs.open(relation_norm_file, 'r') as f2\
                , codecs.open(relation_hyper_file, 'r') as f3:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            lines3 = f3.readlines()
            num = 0
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.entity2id[line[0]] = num
                self.id2entity[num] = line[0]
                self.entity_dic[num] = json.loads(line[1])
                num += 1

            num = 0
            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.relation2id[line[0]] = num
                self.id2relation[num] = line[0]
                self.relation_norm_dic[num] = json.loads(line[1])
                num += 1

            num = 0
            for line in lines3:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.relation_hyper_dic[num] = json.loads(line[1])
                num += 1


        self.module = TransH.TransH(len(self.entity_dic), len(self.relation_norm_dic), len(self.entity_dic[0]),
                                    1.0, 1, 1e-3, 1.0)
        self.module.input_pre_transh(self.entity_dic, self.relation_hyper_dic, self.relation_norm_dic)

        print("RDF Graph Loading...")
        self.g = Graph()
        self.g.parse(rdf_graph, format="turtle")

    def TransR_dataloader(self, entity_file, relation_file, relation_matrix_file, rdf_graph):
        if self.model != "TransR":
            raise RuntimeError("Current model is not TransR. self.model is %s" % self.model)
        print("embedding Data Loading...")
        self.entity_dic = {}
        self.relation_dic = {}

        self.rel_matrix_dic = {}
        with codecs.open(entity_file, 'r') as f1, codecs.open(relation_file, 'r') as f2\
                , codecs.open(relation_matrix_file, 'r') as f3:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            lines3 = f3.readlines()
            num = 0
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.entity2id[line[0]] = num
                self.id2entity[num] = line[0]
                self.entity_dic[num] = json.loads(line[1])
                num += 1

            num = 0
            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.relation2id[line[0]] = num
                self.id2relation[num] = line[0]
                self.relation_dic[num] = json.loads(line[1])
                num += 1

            num = 0
            for line in lines3:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.rel_matrix_dic[num] = json.loads(line[1])
                num += 1
        self.module = TransR.TransR(len(self.entity_dic), len(self.relation_dic), len(self.entity_dic[0]), len(self.relation_dic[0]), 1.0, 1, 1.0)
        self.module.input_pre_transr(self.entity_dic, self.relation_dic, self.rel_matrix_dic)


        print("RDF Graph Loading...")
        self.g = Graph()
        self.g.parse(rdf_graph, format="turtle")


    def select_parse(self, query):
        """
        parse need to delete Literals and their predicate in the SPARQL
        and translate the BGPs to the URI.

        :param query:
        :return:
        """

        if isinstance(query, str):
            pass
        elif isinstance(query, bytes):
            query = query.decode('utf-8')
        else:
            raise TypeError('query takes either unicode-strings or utf-8 encoded byte-strings')

        qres = self.g.query(query)
        if len(qres) == 0:
            raise ValueError("The query can't return less than one result")
        for q in qres:
            self.query_answer.append(q)

        self.query = query
        select_pos = self.query.find("SELECT")
        if select_pos == -1:
            raise ValueError('query should have the "SELECT" keyword.')

        prefix = self.query[:select_pos]
        select = self.query[select_pos:]

        # parse the prefix word
        prefix_dic = {}

        while prefix != "":
            start = prefix.find("PREFIX")
            if start == -1:
                start = prefix.find("@prefix")
                if start == -1:
                    break

            end = prefix.find(">")
            if end == -1:
                break

            p = prefix[start:end+1]
            prefix_word, path = self.__prefixWrapper(p)
            prefix_dic[prefix_word] = path
            prefix = prefix[end+1:]

        # parse the Query Variables
        variables = []
        brace_pos = select.find("{")
        if brace_pos == -1:
            raise ValueError('query format is not right.')

        brace = select[:brace_pos]
        select = select[brace_pos:]

        while brace != "":
            start1 = brace.find("?")
            start2 = brace.find("$")

            if start1 == -1 and start2 != -1:
                start = start1
            elif start2 == -1 and start1 != -1:
                start = start1
            elif start1 < start2 and start1 != -1 and start2 != -1:
                start = start1
            elif start2 < start1 and start1 != -1 and start2 != -1:
                start = start2
            else:
                break

            end = brace.find(" ", start)
            if end == -1:
                break

            variable = brace[start:end]
            variable = variable.replace("\n", "")
            variables.append(variable)
            brace = brace[end:]

        # parse the BGPS
        triples = []
        brace_start = select.find("{")
        if brace_start == -1:
            raise ValueError('query format is not right.')
        brace_end = select.find("}", brace_start)
        if brace_end == -1:
            raise ValueError('query format is not right.')

        select = select[brace_start+1:brace_end]

        triples = self.__BGPsWrapper(select)

        # translate the triples to IRIS
        triples_IRIS = []
        entity = []
        relation = []
        for triple in triples:
            valid = True
            iris = []
            for element in triple:
                if element in variables:
                    iris.append(element)
                    continue

                if element[0] == '<' and element[-1] == '>':
                    iris.append(element[1:-1])
                    continue

                if element[0] == '"' and element[-1] == '"':
                    valid = False
                    break

                pos = element.find(":")
                if pos == -1:
                    raise ValueError('query syntax format is not right.')
                pre = element[:pos]
                if pre not in prefix_dic:
                    raise ValueError('Invalid prefix word "%s"' % pre)
                word = element[pos+1:]
                iris.append(prefix_dic[pre]+word)
            if valid:
                triples_IRIS.append(iris)
                if iris[0] not in variables and iris[0] not in entity:
                    entity.append(iris[0])
                if iris[1] not in variables and iris[1] not in relation:
                    relation.append(iris[1])
                if iris[2] not in variables and iris[2] not in relation:
                    entity.append(iris[2])

        # if triples_IRIS:
        #     raise ValueError('This query syntax is not suitable to use embedding model.')



        self.prefix_dic, self.variables, self.triples, self.triple_iris = prefix_dic, variables, triples, triples_IRIS


    def __prefixWrapper(self, source):
        start_str = "PREFIX "
        end_str = ":"
        start = source.find(start_str)
        if start >= 0:
            start += len(start_str)
            end = source.find(end_str, start)
            if end >= 0:
                prefix_word = source[start:end].strip()
            else:
                raise ValueError('prefix format is not right.')
        else:
             raise ValueError('prefix format is not right.')
        path_start = "<"
        path_end = ">"

        start = source.find(path_start)
        if start >= 0:
            start += len(path_start)
            end = source.find(path_end, start)
            if end >= 0:
                path = source[start:end].strip()
            else:
                raise ValueError('prefix format is not right.')
        else:
            raise ValueError('prefix format is not right.')


        return prefix_word, path

    def __BGPsWrapper(self, source):
        bgps = []
        sample = source.replace("\n", "")

        while len(sample) != 0:
            sample = sample.strip()

            sub_pos = sample.find(" ")
            subject = sample[:sub_pos]
            sample = sample[sub_pos+1:].strip()

            pred_pos = sample.find(" ")
            predicate = sample[:pred_pos]
            sample = sample[pred_pos+1:].strip()

            obj_pos = sample.find(" ")
            object = sample[:obj_pos]
            sample = sample[obj_pos+1:].strip()

            spot = sample.find(".")
            if spot == -1:
                raise ValueError('query syntax format is not right.')
            else:
                sample = sample[spot+1:]

            bgps.append([subject, predicate, object])

        return bgps

    def getRecommendSolution(self):
        '''

        :return:
        '''
        if self.isQueried == False:
            self.getHitTargetAndMeanRank()

        recommendSolution = []
        for i in range(len(self.query_answer)):
            recommendSolution.append(self.result[i][0])
        return recommendSolution

    def getCurrentSolution(self):
        """

        :return:
        """
        if self.isQueried == False:
            self.getHitTargetAndMeanRank()
        return self.query_answer, self.query_answer_rank

    def getHitTargetAndMeanRank(self):
        '''
        return the solution of the query

        :param model: the model type
        :return:
        '''
        weight = self.getWeightOfEdge()


        if len(self.variables) == 1:

            result = {}
            for index in range(len(self.triple_iris)):
                score_triple = self.getScoceOfTriple(self.triple_iris[index])
                for item in score_triple:
                    score_triple[item] *= weight[index]
                    if item in result:
                        result[item] += score_triple[item]
                    else:
                        result[item] = score_triple[item]

            rank_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
            self.result = rank_result

            rank_sum = 0
            hit_range = False
            hit_10 = False
            for item in self.query_answer:
                for i in range(len(self.result)):
                    if str(item[0]) == self.result[i][0]:
                        rank_sum += i + 1
                        if i < len(self.query_answer):
                            hit_range = True
                        if i < 10:
                            hit_10 = True
                        self.query_answer_rank[item] = i + 1
                        break

            sum = 0
            for i in range(len(self.query_answer) + 1):
                sum += i

            weighted_mean_rank = rank_sum/sum
            self.isQueried = True
            mean_mean_rank = rank_sum/len(self.query_answer)
            return hit_range, hit_10, weighted_mean_rank, mean_mean_rank
        else:
            threshold = 100000
            for answer in self.query_answer:
                s = 0
                head = []
                tail = []
                relation = []
                for triple in self.triple_iris:
                    triple = [str(answer[0]) if self.variables[0] == i else i for i in triple]
                    triple = [str(answer[1]) if self.variables[1] == i else i for i in triple]
                    head.append(self.entity2id[triple[0]])
                    relation.append(self.relation2id[triple[1]])
                    tail.append(self.entity2id[triple[2]])

                head = torch.from_numpy(np.array(head)).long()
                relation = torch.from_numpy(np.array(relation)).long()
                tail = torch.from_numpy(np.array(tail)).long()

                distance = self.module.test_distance(head, relation, tail)

                for i, id in enumerate(distance):
                    s += id * weight[i]

                if s <= threshold:
                    threshold = s

            result1 = {}
            for index in range(len(self.triple_iris)):
                if self.variables[0] in self.triple_iris[index] and self.variables[1] not in self.triple_iris[index]:
                    score_triple = self.getScoceOfTriple(self.triple_iris[index])
                    for item in score_triple:
                        score_triple[item] *= weight[index]
                        if item in result1:
                            result1[item] += score_triple[item]
                        else:
                            result1[item] = score_triple[item]

            if not result1:
                for ent1 in self.entity_dic:
                    result1[self.id2entity[ent1]] = 0




            result2 = {}
            for index in range(len(self.triple_iris)):
                if self.variables[1] in self.triple_iris[index] and self.variables[0] not in self.triple_iris[index]:
                    score_triple = self.getScoceOfTriple(self.triple_iris[index])
                    for item in score_triple:
                        score_triple[item] *= weight[index]
                        if item in result2:
                            result2[item] += score_triple[item]
                        else:
                            result2[item] = score_triple[item]

            if not result2:
                for ent1 in self.entity_dic:
                    result2[self.id2entity[ent1]] = 0

            result = {}
            for triple in self.triple_iris:
                if self.variables[0] in triple and self.variables[1] in triple:
                    for ent1 in self.entity_dic:
                        score = 0
                        head = []
                        tail = []
                        relation = []
                        trip = [self.id2entity[ent1] if self.variables[0] == i else i for i in triple]
                        for ent2 in self.entity_dic:
                            triples = [self.id2entity[ent2] if self.variables[1] == i else i for i in trip]
                            head.append(self.entity2id[triples[0]])
                            relation.append(self.relation2id[triples[1]])
                            tail.append(self.entity2id[triples[2]])

                        head = torch.from_numpy(np.array(head)).long()
                        relation = torch.from_numpy(np.array(relation)).long()
                        tail = torch.from_numpy(np.array(tail)).long()

                        distance = self.module.test_distance(head, relation, tail)

                        for i, id in enumerate(distance):
                            score = distance[i] + result1[self.id2entity[ent1]] + result2[self.id2entity[i]]
                            if score >= threshold:
                                result[tuple((self.id2entity[ent1], self.id2entity[i]))] = score

            if not result:
                num = 0
                for k1 in result1:
                    # print(num , k1)
                    for k2 in result2:
                        s = result1[k1] + result2[k2]
                        if s >= threshold:
                            num += 1
                            result[tuple((k1, k2))] = s


            rank_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
            self.result = rank_result

            rank_sum = 0
            hit_range = False
            hit_10 = False
            for item in self.query_answer:
                a = tuple((str(item[0]), str(item[1])))
                for i in range(len(self.result)):

                    if a == self.result[i][0]:
                        rank_sum += i + 1
                        if i < len(self.query_answer):
                            hit_range = True
                        if i < 10:
                            hit_10 = True
                        self.query_answer_rank[item] = i + 1
                        break

            sum = 0
            for i in range(len(self.query_answer) + 1):
                sum += i

            weighted_mean_rank = rank_sum / sum
            self.isQueried = True
            mean_mean_rank = rank_sum / len(self.query_answer)
            return hit_range, hit_10, weighted_mean_rank, mean_mean_rank





    def resetQuery(self, model = "TransE"):
        self.query = """SELECT * WHERE{ ?s ?p ?o. }"""
        self.triples = []
        self.prefix_dic = {}
        self.variables = []
        self.entities = {}
        self.relatioins = {}

        self.result = {}
        self.query_answer = []
        self.query_answer_rank = {}
        self.model = model
        self.isQueried = False

    def getWeightOfEdge(self):
        """
        to generate the weight of the Edge。These weight will be used to calculate the score of the candidate solution.

        :return:
        """
        weight = []

        prefix = ""
        for pre in self.prefix_dic:
            prefix += "PREFIX " + pre +  ":<" + self.prefix_dic[pre] +"> "
        sum = 0
        for triple in self.triple_iris:
            count = 0
            index = 0

            for item in triple:
                select = "SELECT (COUNT("
                pos = item.find("?")
                if pos == -1:
                    continue
                else:
                    index += 1
                    select += item + ") AS ?count) WHERE{ "
                    select += "<" + triple[0] + "> " if triple[0].find("?") == -1 else triple[0] + " "
                    select += "<" + triple[1] + "> " if triple[1].find("?") == -1 else triple[1] + " "
                    select += "<" + triple[2] + "> " if triple[2].find("?") == -1 else triple[2] + " "
                    select += ". } GROUP BY " + item


                query = prefix + select
                qres = self.g.query(query)
                count += len(qres)
            count /= index
            sum += count
            weight.append(count)

        for i in range(len(weight)):
            if weight[i] != 0:
                w = sum / weight[i]
                weight[i] = w
        return weight

    def getScoceOfTriple(self, triple):
        """
        get the score of the candidate solution of the BGPS triple:

        :param triple:
        :return:
        """
        variable = []
        pos = triple[0].find("?")
        if pos != -1:
            variable.append(triple[0])

        pos = triple[2].find("?")
        if pos != -1:
            variable.append(triple[2])

        head = []
        tail = []
        relation = []

        score = {}
        if len(variable) == 1:
            if triple[0] == variable[0]:
                for entity in self.entity_dic:
                    head.append(entity)
                    relation.append(self.relation2id[triple[1]])
                    tail.append(self.entity2id[triple[2]])

                head = torch.from_numpy(np.array(head)).long()
                relation = torch.from_numpy(np.array(relation)).long()
                tail = torch.from_numpy(np.array(tail)).long()

                distance = self.module.test_distance(head, relation, tail)

                # if len(self.variables)  == 1 :
                for i, id in enumerate(self.entity_dic):
                    score[self.id2entity[id]] = distance[i]
                # else:
                #     if variable[0] == self.variables[0]:
                #         for i1, id1 in enumerate(self.entity_dic):
                #             for i2, id2 in enumerate(self.entity_dic):
                #                 score[tuple((self.id2entity[id1], self.id2entity[id2]))] = distance[i1]
                #
                #     elif variable[0] == self.variables[1]:
                #         for i1, id1 in enumerate(self.entity_dic):
                #             for i2, id2 in enumerate(self.entity_dic):
                #                 score[tuple((self.id2entity[id1], self.id2entity[id2]))] = distance[i1]
            elif triple[2] == variable[0]:
                for entity in self.entity_dic:
                    head.append(self.entity2id[triple[0]])
                    relation.append(self.relation2id[triple[1]])
                    tail.append(entity)

                head = torch.from_numpy(np.array(head)).long()
                relation = torch.from_numpy(np.array(relation)).long()
                tail = torch.from_numpy(np.array(tail)).long()

                distance = self.module.test_distance(head, relation, tail)
                # if len(self.variables) == 1:
                for i, id in enumerate(self.entity_dic):
                    score[self.id2entity[id]] = distance[i]
                # else:
                #     if variable[0] == self.variables[0]:
                #         for i1, id1 in enumerate(self.entity_dic):
                #             for i2, id2 in enumerate(self.entity_dic):
                #                 score[tuple((self.id2entity[id2], self.id2entity[id1]))] = distance[i1]
                #
                #     elif variable[0] == self.variables[1]:
                #         for i1, id1 in enumerate(self.entity_dic):
                #             for i2, id2 in enumerate(self.entity_dic):
                #                 score[tuple((self.id2entity[id2], self.id2entity[id1]))] = distance[i1]
        elif len(variable) == 2:
            index = 0
            for h in self.entity_dic:
                head = []
                tail = []
                relation = []
                for t in self.entity_dic:
                    head.append(h)
                    relation.append(self.relation2id[triple[1]])
                    tail.append(t)

                head = torch.from_numpy(np.array(head)).long()
                relation = torch.from_numpy(np.array(relation)).long()
                tail = torch.from_numpy(np.array(tail)).long()

                distance = self.module.test_distance(head, relation, tail)
                for i, id in enumerate(self.entity_dic):
                    score[tuple([self.id2entity[head], self.id2entity[id]])]= distance[i]
                index+=1
                print(index)


        return score

    # def distance(self, h, r, t):
    #     d = 0
    #     if self.model == "TransE":
    #         head = np.array(self.entity_dic[h])
    #         relation = np.array(self.relation_dic[r])
    #         tail = np.array(self.entity_dic[t])
    #         d = np.sum(np.square(head + relation - tail))
    #     elif self.model == "TransH":
    #         head = np.array(self.entity_dic[h])
    #         relation_norm = np.array(self.relation_norm_dic[r])
    #         relation_hyper = np.array(self.relation_hyper_dic[r])
    #         tail = np.array(self.entity_dic[t])
    #         h_hyper = head - np.dot(relation_norm, head) * relation_norm
    #         t_hyper = tail - np.dot(relation_norm, tail) * relation_norm
    #         d = np.sum(np.square(h_hyper + relation_hyper - t_hyper))
    #     elif self.model == "TransR":
    #         head = np.array(self.entity_dic[h])
    #         relation = np.array(self.relation_dic[r])
    #         relation_matrix = np.array(self.rel_matrix_dic[r])
    #         tail = np.array(self.entity_dic[t])
    #         relation_matrix = np.reshape(relation_matrix, (50, 50))
    #         h_r = np.dot(head, relation_matrix)
    #         t_r = np.dot(tail, relation_matrix)
    #         d = np.sum(np.square(h_r + relation - t_r))
    #     return 1.0/(1.0+d)





# if __name__ == '__main__':
#     prepareQuery = '''
#         PREFIX m: <http://rdf.freebase.com/ns/m/>
#         PREFIX ns60: <http://rdf.freebase.com/relation/organization/organization/headquarters./location/mailing_address/>
#         PREFIX ns101:<http://rdf.freebase.com/relation/media_common/netflix_genre/>
#         SELECT ?a
#         WHERE {
#             ?a ns60:citytown m:02_286 .
#             ?a ns60:state_province_region m:059rby .
#         }
#         '''
#
#
#
#
#     s = sparqlParser(model = "TransE")
#     s.TransE_dataloader("entity_vector.data", "relation_vector.data", "train_triples.rdf")
#
#     #s.TransH_dataloader("TransH_entity_vector.data", "TransH_relation_norm_vector.data", "TransH_relation_hyper_vector.data", "train_triples.rdf")
#     # a, p = s.prefixWrapper("PREFIX  m: <http://rdf.freebase.com/ns/m/>")
#     # print(a, p)
#
#     s.select_parse(prepareQuery)
#     hit, mr = s.getHitTargetAndMeanRank()
#     print(hit, mr)
#     result = s.getRecommendSolution()
#     for item in result:
#         print(item)
#     answer, answer_rank = s.getCurrentSolution()
#     for key in answer_rank.keys():
#         print(key, answer_rank[key])
#
#     s.resetQuery(model="TransH")
#     s.TransH_dataloader("TransH_entity_vector.data", "TransH_relation_norm_vector.data", "TransH_relation_hyper_vector.data", "train_triples.rdf")
#     # a, p = s.prefixWrapper("PREFIX  m: <http://rdf.freebase.com/ns/m/>")
#     # print(a, p)
#
#     s.select_parse(prepareQuery)
#     hit, mr = s.getHitTargetAndMeanRank()
#     print(hit, mr)
#     result = s.getRecommendSolution()
#     for item in result:
#         print(item)
#     answer, answer_rank = s.getCurrentSolution()
#     for key in answer_rank.keys():
#         print(key, answer_rank[key])
#
#     # prepareQuery = "PREFIX m: <http://rdf.freebase.com/ns/m/> SELECT ?a { "
#     # s = sparqlParser()
#     # s.dataloader("entity_vector.data", "relation_vector.data", "test_triples.rdf")
#     # with codecs.open("test_ntrips.nt", 'r') as f1:
#     #     lines1 = f1.readlines()
#     #     hit_target = 0
#     #     rank_sum = 0
#     #     for line in lines1:
#     #         s.resetQuery()
#     #         line = line.strip().split(' ')
#     #         # if len(line) != 3:
#     #         #     continue
#     #         pr = np.random.random(1)[0]
#     #         if pr > 0.5:
#     #             line[0] = "?a"
#     #         else:
#     #             line[2] = "?a"
#     #         query = prepareQuery + line[0] + " " + line[1] + " " + line[2] + " . }"
#     #         s.select_parse(query)
#     #         hit, rank, num = s.getRecommendSolution()
#     #         if hit:
#     #             hit_target += 1
#     #         rank_sum += rank
#     #     print("hit_target: " + str(hit_target / len(lines1)))
#     #     print("mean_rank: " + str(rank_sum / len(lines1)))




# if __name__ == '__main__':
    # prepareQuery = '''
    #     PREFIX m: <http://rdf.freebase.com/ns/m/>
    #     PREFIX ns60: <http://rdf.freebase.com/relation/organization/organization/headquarters./location/mailing_address/>
    #     PREFIX ns101:<http://rdf.freebase.com/relation/media_common/netflix_genre/>
    #     SELECT ?a
    #     WHERE {
    #         ?a ns60:citytown m:02_286 .
    #         ?a ns60:state_province_region m:059rby .
    #     }
    #     '''
    #
    #
    #
    #
    # s = sparqlParser(model = "TransE")
#     s.TransE_dataloader("entity_vector.data", "relation_vector.data", "train_triples.rdf")

    #s.TransH_dataloader("TransH_entity_vector.data", "TransH_relation_norm_vector.data", "TransH_relation_hyper_vector.data", "train_triples.rdf")
    # a, p = s.prefixWrapper("PREFIX  m: <http://rdf.freebase.com/ns/m/>")
    # print(a, p)

#     s.select_parse(prepareQuery)
#     hit, mr = s.getHitTargetAndMeanRank()
#     print(hit, mr)
#     result = s.getRecommendSolution()
#     for item in result:
#         print(item)
#     answer, answer_rank = s.getCurrentSolution()
#     for key in answer_rank.keys():
#         print(key, answer_rank[key])

    # s.resetQuery(model="TransH")
    # s.TransH_dataloader("TransH_entity_vector.data", "TransH_relation_norm_vector.data", "TransH_relation_hyper_vector.data", "train_triples.rdf")
    # # a, p = s.prefixWrapper("PREFIX  m: <http://rdf.freebase.com/ns/m/>")
    # # print(a, p)
    #
    # s.select_parse(prepareQuery)
    # hit, mr = s.getHitTargetAndMeanRank()
    # print(hit, mr)
    # result = s.getRecommendSolution()
    # for item in result:
    #     print(item)
    # answer, answer_rank = s.getCurrentSolution()
    # for key in answer_rank.keys():
    #     print(key, answer_rank[key])

    # s.resetQuery(model="TransR")
    # s.TransR_dataloader("TransR_entity_vector.data", "TransR_relation_vector.data", "TransR_relation_matrix.data", "train_triples.rdf")
    #
    # s.select_parse(prepareQuery)
    # hit, mr = s.getHitTargetAndMeanRank()
    # print(hit, mr)
    # result = s.getRecommendSolution()
    # for item in result:
    #     print(item)
    # answer, answer_rank = s.getCurrentSolution()
    # for key in answer_rank.keys():
    #     print(key, answer_rank[key])

    # prepareQuery = "PREFIX m: <http://rdf.freebase.com/ns/m/> SELECT ?a { "
    # s = sparqlParser(model="TransR")
    # s.TransR_dataloader("TransR_entity_vector.data", "TransR_relation_vector.data", "TransR_relation_matrix.data", "test_triples.rdf")
    # with codecs.open("test_ntrips.nt", 'r') as f1:
    #     lines1 = f1.readlines()
    #     hit_target = 0
    #     rank_sum = 0
    #     num = 0
    #     for line in lines1:
    #         num += 1
    #
    #         s.resetQuery(model="TransR")
    #         line = line.strip().split(' ')
    #         # if len(line) != 3:
    #         #     continue
    #         pr = np.random.random(1)[0]
    #         if pr > 0.5:
    #             line[0] = "?a"
    #         else:
    #             line[2] = "?a"
    #         query = prepareQuery + line[0] + " " + line[1] + " " + line[2] + " . }"
    #         s.select_parse(query)
    #         hit, rank = s.getHitTargetAndMeanRank()
    #         if hit:
    #             hit_target += 1
    #         rank_sum += rank
    #         print(num, line, str(hit_target / num), str(rank_sum / num))
    #     print("hit_target: " + str(hit_target / len(lines1)))
    #     print("mean_rank: " + str(rank_sum / len(lines1)))








