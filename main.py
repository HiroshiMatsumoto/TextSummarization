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
import random

tagger = mcb.Tagger("-Owakati")
tagger.parse("")


def load_doc():
    dotenv.load_dotenv()
    doc_dir_path = pathlib.Path(os.environ['DOCUMENT_DIR_PATH'])
    json_dir = doc_dir_path / "data" / "json"

    docs = dict()
    org_docs = dict()
    vocab = Counter()
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
                    vocab += Counter(tokens)
                    org_docs[file_json["original_title"]][sentence_idx] = sentence
                    sentence_idx += 1
        break
    return docs, org_docs, vocab


def cosine(m: dict, n: dict) -> float:
    """
    cosine similarity of two vectors, m and n

    :param m: dict {feature:scalar}
    :param n: dict {feature:scalar}
    :return: float
    """
    denom = math.sqrt(sum(m[i]**2 for i in m.keys())) * math.sqrt(sum(n[i]**2 for i in n.keys()))
    if 0 == denom:
        return 0
    return sum(m[i] * n[i] for i in (set(m.keys()) & set(n.keys()))) / denom


def euclid(m: dict, n: dict) -> float:
    """
    euclid distance between m and n
    :param m: dict {feature_key: scalar}
    :param n: dict {feature_key: scalar}
    :return: float
    """
    return math.sqrt(sum(((m[k] if k in m else 0) - (n[k] if k in n else 0))**2 for k in set(m.keys()) | set(n.keys())))


def Cover(i, S, w=cosine) -> float:
    """
    similarity of i to Set
    eqn. (4)
    C:2**V -> R

    :param i: vector
    :param S: collection of vectors
    :param w: similarity measure
    :return: float
    """
    return sum(w(i, S[j]) for j in S)


def L(S, V, alpha) -> float:
    """
    similarity of summary set S to the document to be summarized
    or coverage of V

    monotone submodular function

    :param S: subset
    :param V: ground set
    :param alpha: parameter
    :return: float
    """
    return sum(min(Cover(V[i], S), alpha * Cover(V[i], V)) for i in V)


def R(S, V, P) -> float:
    """
    Diveristy Reward function

    eqn. (5) in Section 4.2 -->  eqn. (7)

    P: a paration of V
    r: singleton reward of unit i, estimation of the importance of i to the summary
    "the average similarity of sentence i to the rest of the document"

    :param S: subset
    :param V: ground set
    :param P: partions (clusterings)
    :return: float
    """
    N = len(V)
    # summation = 0
    # for k in P:
    #     if not k:
    #         continue
    #     if not set(k.keys() & set(S.keys())):
    #         continue
    #     summation += math.sqrt(sum(sum(cosine(V[i], S[j]) for i in V) / N for j in set(k.keys()) & set(S.keys())))
    return sum(math.sqrt(sum(sum(cosine(V[i], S[j]) for i in V) / N for j in set(k.keys()) & set(S.keys()))) for k in P)


def F(S, V, alpha, lmbd, P):
    return L(S, V, alpha) + lmbd * R(S, V, P)


def kmeans(targets, k, features, threashold=0.1):
    pair_distance_list = []
    # centroid index is same as cluster id
    centroids = [{fkey:random.random() for fkey in features} for _ in range(k)]
    # each cluster is a colletion of target keys (document-id)
    clusters = [[] for _ in range(k)]

    max_distance = 1
    while threashold < max_distance:
        # initialize
        clusters = [[] for _ in range(k)]
        # calculate centroid-element distance
        for centroid_idx, centroid in enumerate(centroids):
            for target_key, target in targets.items():
                distance = euclid(centroid, target)
                pair_distance_list.append((distance, centroid_idx, target_key))
        # cluster distribution
        clustered_ids = set()
        for distance, centroid_id, target_key in sorted(pair_distance_list):
            if target_key in clustered_ids:
                continue
            clusters[centroid_id].append(target_key)
            clustered_ids.add(target_key)
        # update centroids
        new_centroids = [{} for _ in range(k)]
        for cluster_id, cluster in enumerate(clusters):
            new_centroids[cluster_id] = {
                feature:
                    sum(targets[target_id][feature] if feature in targets[target_id] else 0 for target_id in cluster)
                    / len(cluster) if len(cluster) else 0
                for feature in features}
        # centroid move distance
        max_distance = 0
        for cluster_id, centroid in enumerate(centroids):
            if not new_centroids[cluster_id] or not centroids[cluster_id]:
                continue
            distance = euclid(new_centroids[cluster_id], centroids[cluster_id])
            if max_distance < distance:
                max_distance = distance
        centroids = new_centroids
    return clusters


def main():
    docs, orig_docs, vocab = load_doc()
    doc = docs[list(docs.keys())[0]]
    orig_doc = orig_docs[list(docs.keys())[0]]


    clusters = kmeans(doc, k=5, features=vocab)

    # V: ground set
    V = doc
    # S: subset
    S = dict()
    # P: document clusters
    P = [{doc_id: doc[doc_id] for doc_id in cluster} for cluster in clusters]

    # Problem 1.
    # S \in argmax F(S) s.t sum < budget

    # parameters
    alpha = 0.5
    lmbd = 0.1
    cost = 1
    budget = 5
    
    # Greedy Search
    while sum(cost for i in S) < budget:
        # create tmp-subset
        tmp_subset = copy.deepcopy(S)
        max_score = 0
        max_i = 0
        for i in V:
            if i in S:
                continue
            # select highest score Item
            tmp_subset[i] = V[i]
            score = F(tmp_subset, V, alpha, lmbd, P)
            if max_score < score:
                max_score = score
                max_i = i
        S[max_i] = V[max_i]

    print("summary")
    for i in S:
        print(i,  orig_doc[i])
    print(len(S))


if __name__ == '__main__':
    main()
