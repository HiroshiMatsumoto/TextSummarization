#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pathlib
import dotenv
import os
import json
import MeCab as mcb
from collections import Counter
import math
import copy

tagger = mcb.Tagger("-Owakati")
tagger.parse("")


def load_doc():
    dotenv.load_dotenv()
    doc_dir_path = pathlib.Path(os.environ['DOCUMENT_DIR_PATH'])
    json_dir = doc_dir_path / "data" / "json"

    docs = dict() # doc : sentence : count
    org_docs = dict()
    for f in json_dir.iterdir():
        file_json = json.load(f.open("r", encoding="utf-8"))

        if file_json["original_title"] not in docs:
            docs[file_json["original_title"]] = dict()
            org_docs[file_json["original_title"]] = dict()

        sentence_idx = 0
        for content in file_json['contents']:
            for subcontent in content['part_contents']:
                article = subcontent["article"]
                for sentence in article.split("。"):
                    if not sentence:
                        continue
                    sentence = sentence + "。"
                    tokens = tagger.parse(sentence).split()
                    docs[file_json["original_title"]][sentence_idx] = Counter(tokens)
                    org_docs[file_json["original_title"]][sentence_idx] = sentence
                    sentence_idx += 1
        break
    return docs, org_docs


def cosine(m: dict, n: dict):
    denom = math.sqrt(sum(m[i]**2 for i in m.keys())) * math.sqrt(sum(n[i]**2 for i in n.keys()))
    if 0 == denom:
        return 0
    return sum(m[i] * n[i] for i in (set(m.keys()) & set(n.keys()))) / denom


def Cover(i, Set) -> float:
    """
    similarity of i to Set
    eqn. (4)
    C:2**V -> R

    :param i: vector
    :param Set: collection of vectors
    :return:
    """
    return sum(cosine(i, Set[j]) for j in Set)


def L(S, V, alpha) -> float:
    """
    similarity of summary set S to the document to be summarized
    or coverage of V

    monotone submodular function

    :param S: subset
    :param V: ground set
    :param alpha: parameter
    :return:
    """
    return sum(min(Cover(V[i], S), alpha * Cover(V[i], V)) for i in V)


def R(S):
    # Diversity Rewards
    return 0


def F(S, V, alpha, lmbd):
    return L(S, V, alpha)  # + lmbd * R(S)


def main():
    docs, orig_docs = load_doc()
    doc = docs[list(docs.keys())[0]]
    orig_doc = orig_docs[list(docs.keys())[0]]


    # subset = set(itr.combinations(range(set_size), subset_size))
    # print(subset)
    V = doc
    S = dict()

    # S: subset
    # V: all
    # S \in argmax F(S) s.t sum

    # parameters
    constrain_size = 5
    alpha = 0.5
    lmbd = 0.1

    # Greedy Search
    cost = 1
    budget = 5
    # summary = dict()
    while sum(cost for i in S) < budget:
        # print("S: ", len(S))
        # create tmp-subset
        tmp_subset = copy.deepcopy(S)
        max_score = 0
        max_i = 0
        for i in V:
            if i in S:
                continue
            # select highest score Item
            tmp_subset[i] = V[i]
            score = F(tmp_subset, V, alpha, lmbd)
            if max_score < score:
                max_score = score
                max_i = i
        S[max_i] = V[max_i]
        print("arg_max:", max_i, ":", orig_doc[max_i])

    print("summary")
    for i in S:
        print(i,  orig_doc[i])
    print(len(S))


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


