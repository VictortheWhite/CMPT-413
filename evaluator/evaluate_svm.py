#!/usr/bin/env python
import argparse # optparse is deprecated
import sys
import string
import nltk
import os
import pickle
from sklearn import svm
from itertools import islice # slicing for iterators


parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref', help='input file (default data/hyp1-hyp2-ref)')
parser.add_argument('-n', '--num_sentences', default=None, type=int, help='Number of hypothesis pairs to evaluate')

opts = parser.parse_args()


def word_matches(h, ref):
    return sum(1 for w in h if w in ref)

def classify():
    pass


# we create a generator and avoid loading all sentences into a list
def sentences(input_file):
    with open(input_file) as f:
        for pair in f:
            yield map(preprocess, [sentence.strip().split() for sentence in pair.split(' ||| ')])

def preprocess(sentence):
    # lower all words and seprerate punctuations from words
    res = []
    puncs = string.punctuation
    for w in sentence:
        start = 0
        for i, l in enumerate(w):
            if l in string.punctuation:
                if i > start:
                    res.append(w[start:i].lower())
                res.append(l)
                start = i + 1
            i += 1
        if start < len(w):
            res.append(w[start:].lower())
    return res

def train():
    feat_vecs = []
    train_set = 'data/train-test.hyp1-hyp2-ref'

    # compute feat_vecs
    i = 0
    if not os.path.isfile("data/feat_vec_train"):
        sys.stderr.write("computing feature vec\n")
        for h1, h2, ref in islice(sentences(train_set), opts.num_sentences):
            if i % 1000 == 0:
                sys.stderr.write("%d\n" % (i,))
            i += 1
            feat_vecs.append((computeFeatVec(h1, ref), computeFeatVec(h2, ref)));
            # use the diff of 2 vecs?
        with open('data/feat_vec_train', 'wb') as f:
            sys.stderr.write('writting feat_vec to file\n')
            pickle.dump(feat_vecs, f)
            f.close()
    else:
        with open('data/feat_vec_train', 'rb') as f:
            sys.stderr.write('reading feat_vec from file\n')
            feat_vecs = pickle.load(f)
            f.close()

    # read labels
    true_labels = []
    with open('data/train.gold', 'r') as f:
        for label in f:
            true_labels.append(int(label))

    feat_vecs = map(lambda (f1, f2): [ f1[i]-f2[i] for i in range(len(f1))], feat_vecs[:len(true_labels)])
    assert len(feat_vecs) == len(true_labels)

    lin_clf = svm.LinearSVC()
    sys.stderr.write('trainning svm\n')
    lin_clf.fit(feat_vecs, true_labels)

    return lin_clf



def computeFeatVec(h, ref):
    feat_vec = [0.0] * 32

    uniCount = uniMatch(h, ref)
    biCount = biMatch(h, ref)
    triCount = triMatch(h, ref)
    fourCount = fourMatch(h, ref)

    l_h, l_ref = float(len(h)), float(len(ref))
    # 0 - 3 : n-gram precision
    feat_vec[0] = uniCount / l_ref if l_ref > 0 else 1.0
    feat_vec[1] = biCount / l_ref-1 if l_ref > 1 else 1.0
    feat_vec[2] = triCount / l_ref-2 if l_ref > 2 else 1.0
    feat_vec[3] = fourCount / l_ref-3 if l_ref > 3 else 1.0

    # 4 - 7 : n-gram recall
    feat_vec[4] = uniCount / l_h if l_h > 0 else 1.0
    feat_vec[5] = biCount / (l_h-1)  if l_h > 1 else 1.0
    feat_vec[6] = triCount / (l_h-2)  if l_h > 2 else 1.0
    feat_vec[7] = fourCount / (l_h-3)  if l_h > 3 else 1.0

    # 8 - 11 : n-gram f-measure
    feat_vec[8] = 2 * uniCount / (l_h + l_ref) if l_h + l_ref > 0 else 1.0
    feat_vec[9] = 2 * biCount / (l_h + l_ref - 2) if l_h + l_ref > 2 else 1.0
    feat_vec[10] = 2 * triCount / (l_h + l_ref - 4) if l_h + l_ref > 4 else 1.0
    feat_vec[11] = 2 * fourCount / (l_h + l_ref - 6) if l_h + l_ref > 6 else 1.0

    # 13 : average n-gram precison
    feat_vec[13] = (feat_vec[0] + feat_vec[1] + feat_vec[2] + feat_vec[3]) / 4

    # 14 : word count
    feat_vec[14] = (l_h - l_ref) / l_ref

    # 15 : punctuation count
    feat_vec[15] = (float(countPunctuation(h) - countPunctuation(ref))) / l_ref


    # pos tags
    h_w_tag = nltk.pos_tag(h)
    ref_w_tag = nltk.pos_tag(ref)

    h_pos_tag = map(lambda (w, tag): tag, h_w_tag)
    ref_pos_tag = map(lambda (w, tag): tag, ref_w_tag)
    h_w_tag = map(lambda (w, tag): w+'/'+tag, h_w_tag)
    ref_w_tag = map(lambda (w, tag): w+'/'+tag, ref_w_tag)

    uniPosCount = uniMatch(h_pos_tag, ref_pos_tag)
    biPosCount = biMatch(h_pos_tag, ref_pos_tag)
    triPosCount = triMatch(h_pos_tag, ref_pos_tag)
    fourPosCount = fourMatch(h_pos_tag, ref_pos_tag)

    # 16 - 19 : n-gram pos precision
    feat_vec[16] = uniPosCount / l_ref if l_ref > 0 else 1.0
    feat_vec[17] = biPosCount / (l_ref-1) if l_ref > 1 else 1.0
    feat_vec[18] = triPosCount / (l_ref-2) if l_ref > 2 else 1.0
    feat_vec[19] = fourPosCount / (l_ref-3) if l_ref > 3 else 1.0

    # 20 - 23 : n-gram pos recall
    feat_vec[20] = uniPosCount / (l_h) if l_h > 0 else 1.0
    feat_vec[21] = biPosCount / (l_h-1) if l_h > 1 else 1.0
    feat_vec[22] = triPosCount / (l_h-2) if l_h > 2 else 1.0
    feat_vec[23] = fourPosCount / (l_h-3) if l_h > 3 else 1.0

    # 24 - 27 : n-gram pos f-measure
    feat_vec[24] = 2 * uniPosCount / (l_h + l_ref) if l_h + l_ref > 0 else 1.0
    feat_vec[25] = 2 * biPosCount / (l_h + l_ref - 2) if l_h + l_ref > 2 else 1.0
    feat_vec[26] = 2 * triPosCount / (l_h + l_ref - 4) if l_h + l_ref > 4 else 1.0
    feat_vec[27] = 2 * fourPosCount / (l_h + l_ref - 6) if l_h + l_ref > 6 else 1.0

    # 28 - 31 : pos string mixed precision
    feat_vec[28] = uniMatch(h_w_tag, ref_w_tag) / l_ref  if l_ref > 0 else 1.0
    feat_vec[29] = biMatch(h_w_tag, ref_w_tag) / (l_ref-1)  if l_ref > 1 else 1.0
    feat_vec[30] = triMatch(h_w_tag, ref_w_tag) / (l_ref-2)  if l_ref > 2 else 1.0
    feat_vec[31] = fourMatch(h_w_tag, ref_w_tag) / (l_ref-3)  if l_ref > 3 else 1.0

    return feat_vec

def uniMatch(h, ref):
    return sum(1 for w in h if w in ref)

def biMatch(h, ref):
    ref_scentence = ' '.join(ref)
    count = 0
    for i in range(len(h)-1):
        if ' '.join(h[i:i+2]) in ref_scentence:
            count += 1
    return count

def triMatch(h, ref):
    ref_scentence = ' '.join(ref)
    count = 0
    for i in range(len(h)-2):
        if ' '.join(h[i:i+3]) in ref_scentence:
            count += 1
    return count

def fourMatch(h, ref):
    ref_scentence = ' '.join(ref)
    count = 0
    for i in range(len(h)-3):
        if ' '.join(h[i:i+4]) in ref_scentence:
            count += 1
    return count

def countPunctuation(s):
    return sum(1 for w in s if w in string.punctuation)


def read_test_data():
    feat_vecs = []

    sys.stderr.write('reading test data\n')
    # compute feat_vecs
    i = 0
    if not os.path.isfile("data/feat_vec_test"):
        sys.stderr.write("computing feature vec\n")
        for h1, h2, ref in islice(sentences(opts.input), opts.num_sentences):
            if i % 1000 == 0:
                sys.stderr.write("%d\n" % (i,))
            i += 1
            feat_vecs.append((computeFeatVec(h1, ref), computeFeatVec(h2, ref)));
            # use the diff of 2 vecs?
        with open('data/feat_vec_test', 'wb') as f:
            sys.stderr.write('writting feat_vec_test to file\n')
            pickle.dump(feat_vecs, f)
            f.close()
    else:
        with open('data/feat_vec_test', 'rb') as f:
            sys.stderr.write('reading feat_vec_test from file\n')
            feat_vecs = pickle.load(f)
            f.close()

    return map(lambda (f1, f2): [ f1[i]-f2[i] for i in range(len(f1)) ], feat_vecs)

# convention to allow import of this file as a module
if __name__ == '__main__':
    # read test data
    test_data = read_test_data()
    # train classifier
    lin_clf = train()

    sys.stderr.write('classifiying\n')
    for l in lin_clf.predict(test_data):
        print l
        #print lin_clf.predict([feat_vec])[0]
