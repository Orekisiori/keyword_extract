import os
from collections import Counter
import networkx as nx
import operator
import numpy as np
import model.rakun.pre_data as pd


class RakunDetector:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.keyword_graph = None
        self.input_type = 'text'

    def corpus_graph(self,
                     language_file,
                     limit_range=3000000,
                     verbose=False,
                     stopwords_path=os.path.normpath(os.path.join(os.getcwd(), 'data/biao_stopwords.txt'))):

        G = nx.DiGraph()
        ctx = 0
        reps = False
        dictionary_with_counts_of_pairs = {}
        self.whole_document = []

        # get stopwords
        stop = []
        with open(stopwords_path, "r",encoding='utf-8') as f:
            for fline in f.readlines():
                stop.append(fline.strip())

        def process_line(line):
            nonlocal G
            nonlocal ctx
            nonlocal reps
            nonlocal dictionary_with_counts_of_pairs

            # 去掉停止词
            line = line.strip()
            temp_line = line.split(" ")
            re_line = [i for i in temp_line if i not in stop]
            self.whole_document += re_line

            if len(re_line) > 1:
                ctx += 1
                if ctx % limit_range == 0:
                    return True
                for enx, el in enumerate(re_line):
                    if enx > 0:
                        edge_directed = (re_line[enx - 1], el)
                        if edge_directed[0] != edge_directed[1]:
                            G.add_edge(edge_directed[0], edge_directed[1])
                        else:
                            edge_directed = None
                    if enx < len(re_line) - 1:
                        edge_directed = (el, re_line[enx + 1])
                        if edge_directed[0] != edge_directed[1]:
                            G.add_edge(edge_directed[0], edge_directed[1])
                        else:
                            edge_directed = None
                    if edge_directed:
                        if edge_directed in dictionary_with_counts_of_pairs:
                            dictionary_with_counts_of_pairs[edge_directed] += 1
                            reps = True
                        else:
                            dictionary_with_counts_of_pairs[edge_directed] = 1
            return False

        input_type = self.input_type
        if input_type == "file":
            with open(language_file, encoding='utf-8') as lf:
                for line in lf:
                    breakBool = process_line(line)
                    if breakBool:
                        break

        elif input_type == "text":
            lines = language_file.split("\n")
            for line in lines:
                breakBool = process_line(line)
                if breakBool:
                    break

        ## assign edge properties.
        for edge in G.edges(data=True):
            try:
                edge[2]['weight'] = dictionary_with_counts_of_pairs[(edge[0], edge[1])]
            except Exception as es:
                raise (es)
        if verbose:
            print(nx.info(G))

        return (G, reps)

    def find_keywords(self, document, limit_num_keywords, num_tokens=[1], double_weight_threshold=2, max_occurrence=2,
                      max_similar=2, connectives=True, stopwords_path=os.path.normpath(os.path.join(os.getcwd(), 'data/biao_stopwords.txt'))):


        all_terms = set()

        input_type = self.input_type
        if input_type == "file":
            self.raw_text = open(document, encoding='utf_8').read().split(" ")

        else:
            self.raw_text = document.split()

        weighted_graph, reps = self.corpus_graph(document,
                                                 stopwords_path=stopwords_path)
        # node number
        nn = len(list(weighted_graph.nodes()))

        self.initial_tokens = nn

        pgx = nx.load_centrality(weighted_graph)

        # global vars
        self.keyword_graph = weighted_graph
        self.centrality = pgx

        # sort by scores
        keywords_with_scores = sorted(pgx.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)
        # 重新转化成dict
        kw_map = dict(keywords_with_scores)

        if reps and 2 in num_tokens or 3 in num_tokens:

            higher_order_1 = []
            higher_order_2 = []
            frequent_pairs = []

            # Check potential edges
            for edge in weighted_graph.edges(data=True):
                if edge[0] != edge[1]:
                    if "weight" in edge[2]:
                        if edge[2]['weight'] > double_weight_threshold:
                            frequent_pairs.append(edge[0:2])

            # Traverse the frequent pairs
            for pair in frequent_pairs:
                w1 = pair[0]
                w2 = pair[1]
                if w1 in kw_map and w2 in kw_map:
                    score = np.mean([kw_map[w1], kw_map[w2]])
                    if not w1 + " " + w2 in all_terms:
                        higher_order_1.append((w1 + " " + w2, score))
                        all_terms.add(w1 + " " + w2)

            # Three word keywords are directed paths.
            three_gram_candidates = []
            for pair in frequent_pairs:
                for edge in weighted_graph.in_edges(pair[0]):
                    if edge[0] in kw_map:
                        trip_score = [
                            kw_map[edge[0]], kw_map[pair[0]], kw_map[pair[1]]
                        ]
                        term = edge[0] + " " + pair[0] + " " + pair[1]
                        score = np.mean(trip_score)
                        if not term in all_terms:
                            higher_order_2.append((term, score))
                            all_terms.add(term)

                for edge in weighted_graph.out_edges(pair[1]):
                    if edge[1] in kw_map:
                        trip_score = [
                            kw_map[edge[1]], kw_map[pair[0]], kw_map[pair[1]]
                        ]
                        term = pair[0] + " " + pair[1] + " " + edge[1]
                        score = np.mean(trip_score)
                        if not term in all_terms:
                            higher_order_2.append((term, score))
                            all_terms.add(term)
        else:
            higher_order_1 = []
            higher_order_2 = []

        total_keywords = []

        if 1 in num_tokens:
            total_keywords += keywords_with_scores

        if 2 in num_tokens:
            total_keywords += higher_order_1

        if 3 in num_tokens:
            total_keywords += higher_order_2
        total_kws = sorted(set(total_keywords),
                           key=operator.itemgetter(1),
                           reverse=True)
        # remove some noise
        tokensets = []
        for keyword in total_kws:
            ltx = keyword[0].split(" ")
            if len(ltx) > 1:
                tokensets += ltx

        # 通用操作
        penalty = set([
            x[0] for x in Counter(tokensets).most_common(max_occurrence)
        ])

        tmp = []
        pnx = 0
        for keyword in total_kws:
            parts = set(keyword[0].split(" "))
            if len(penalty.intersection(parts)) > 0:
                pnx += 1
                if pnx < max_similar:
                    tmp.append(keyword)
            else:
                tmp.append(keyword)
        total_kws = tmp

        # missing connectives
        if connectives:
            final_keywords = []
            for kw in total_kws:
                parts = kw[0].split(" ")
                joint = False
                if len(parts) > 1:
                    if len(parts) == 2:
                        p1 = parts[0]
                        p2 = parts[1]
                        i1_indexes = [
                            n for n, x in enumerate(self.raw_text) if x == p1
                        ]
                        i2_indexes = [
                            n for n, x in enumerate(self.raw_text) if x == p2
                        ]
                        i2_indexes_map = set([
                            n for n, x in enumerate(self.raw_text) if x == p2
                        ])
                        for ind in i1_indexes:
                            if ind + 2 in i2_indexes_map:
                                joint_kw = " ".join([
                                    p1, self.raw_text[ind + 1],
                                    self.raw_text[ind + 2]
                                ])
                                final_keywords.append((joint_kw, kw[1]))
                                joint = True
                if not joint:
                    final_keywords.append((kw[0], kw[1]))
            total_kws = final_keywords
        # 结尾需要去重处理。目前重复的原因未知，若关键词完全重复出现只是一个bug而没有实际的意义，这样处理没有问题
        result_kws = []
        part_head = len(result_kws)
        part_tail = limit_num_keywords
        while len(result_kws) < limit_num_keywords:
            result_kws += set(total_kws[part_head:part_tail])
            result_kws = sorted(list(result_kws), key=lambda x:x[1],reverse=True)
            add_len = len(result_kws) - part_head
            part_head = part_tail
            part_tail = part_head + add_len
        # 去掉评分，只输出关键词
        final_result = [i[0] for i in result_kws]
        return final_result


def rakun(text, input_type):
    keyword_detector = RakunDetector()
    keyword_detector.input_type = input_type
    example_data = text
    # 预处理
    done = pd.precess_data(text, input_type)
    # 关键词提取
    keywords = keyword_detector.find_keywords(document=done, limit_num_keywords=10, num_tokens=[1],
                                              max_occurrence=2, max_similar=2)
    return keywords
