from typing import List, Union

from nltk.corpus import wordnet as wn
import spacy
from pyinflect import getInflection

from common import ActionPred, BBT_PEOPLE

ROOT_TAG = 'ROOT'
VERB_TAG = 'VERB'
NOUN_TAG = 'NOUN'
PRON_TAG = 'PRON'
PROPN_TAG = 'PROPN'

N_SUBJ = 'nsubj'
D_OBJ = 'dobj'
NPADV_MOD = 'npadvmod'

PLACE_HOLDER = 'place_holder'

nlp = spacy.load('en_core_web_sm')

SYNONYMS = 'SYNONYMS'
HYPONYMS = 'HYPONYMS'
HYPERNYMS = 'HYPERNYMS'
NYMS_FUNC = [SYNONYMS, HYPONYMS, HYPERNYMS]


def _similar_lemmas(synset1, synset2, s1: str, s2: str) -> bool:
    for ss in synset1:
        for l in ss.lemmas():
            if l.name() == s2:
                return True
    for ss in synset2:
        for l in ss.lemmas():
            if l.name() == s1:
                return True
    return False


def check_synonyms(s1: str, s2: str, pos: Union[str, None] = None) -> bool:
    if s1 == s2:
        return True

    synset1, synset2 = get_synset(s1, s2, pos)
    for ss in synset1:
        for l in ss.lemmas():
            if l.name() == s2:
                return True
    for ss in synset2:
        for l in ss.lemmas():
            if l.name() == s1:
                return True
    return False


def check_hyponyms(s1: str, s2: str, pos: Union[str, None] = None) -> bool:
    # Check if s1 is s2's hyponym
    synset1, synset2 = get_synset(s1, s2, pos)
    for ss2 in synset2:
        hyponyms = ss2.hyponyms()
        for h in hyponyms:
            for l in h.lemmas():
                if l.name() == s1:
                    return True
    return False


def check_hypernyms(s1: str, s2: str, pos=None) -> bool:
    # Check if s1 is s2's hypernym
    synset1, synset2 = get_synset(s1, s2, pos)
    for ss2 in synset2:
        hyponyms = ss2.hypernyms()
        for h in hyponyms:
            for l in h.lemmas():
                if l.name() == s1:
                    return True
    return False


def get_synset(s1: str, s2: str, pos: Union[str, None] = None):
    if pos is not None:
        pos_to_wn_pos = {
            'VERB': wn.VERB,
            'NOUN': wn.NOUN
        }
        assert pos in pos_to_wn_pos

        wn_pos = pos_to_wn_pos[pos]
        ss1 = wn.synsets(s1, pos=wn_pos)
        ss2 = wn.synsets(s2, pos=wn_pos)
    else:
        ss1 = wn.synsets(s1)
        ss2 = wn.synsets(s2)
    return ss1, ss2


def _check_nyms(s1: str, s2: str, func: str,
                pos: Union[str, None] = None) -> bool:
    assert func in NYMS_FUNC

    if s1 == s2:
        return True

    ss1, ss2 = get_synset(s1, s2, pos)

    if func == SYNONYMS:
        return _similar_lemmas(ss1, ss2, s1, s2)

    for ss in ss1:
        hyp = ss.hyponyms() if func == HYPONYMS else ss.hypernyms()
        if _similar_lemmas(hyp, ss2, s1, s2):
            return True
    for ss in ss2:
        hyp = ss.hyponyms() if func == HYPONYMS else ss.hypernyms()
        if _similar_lemmas(hyp, ss1, s1, s2):
            return True
    return False


def valid_obj_word(t) -> bool:
    pos = t.pos_
    dep = t.dep_
    return pos in [NOUN_TAG, PRON_TAG] and dep == D_OBJ


def valid_subj_word(t) -> bool:
    if t.text.lower() in BBT_PEOPLE:
        return True

    pos = t.pos_
    dep = t.dep_
    return pos in [PROPN_TAG] and dep == N_SUBJ


def _get_action_pred(doc) -> Union[ActionPred, None]:
    for token in doc:
        lemma = token.lemma_
        dep = token.dep_
        pos = token.pos_

        # Process the ROOT of the dependency tree
        if dep == ROOT_TAG and pos == VERB_TAG:
            pred_subj = None

            for l in token.lefts:
                if pred_subj is None and valid_subj_word(l):
                    pred_subj = l.lemma_

            if pred_subj is not None:
                action_g = getInflection(lemma, 'VBG')[0]
                return ActionPred(action_g, pred_subj, PLACE_HOLDER)
            else:
                return None


def get_action_pred(text: str) -> Union[ActionPred, None]:
    return _get_action_pred(nlp(text))


def get_action_preds(texts: List[str]) -> List[Union[ActionPred, None]]:
    return list(map(lambda x: _get_action_pred(x), nlp.pipe(texts)))


def get_root_obj_token(text: str):
    doc = nlp(text)
    for token in doc:
        if token.dep_ == ROOT_TAG and token.pos_ in [NOUN_TAG, PROPN_TAG]:
            return token
    return None


def get_action_obj_token(text: str):
    doc = nlp(text)
    for token in doc:
        if token.dep_ == ROOT_TAG:
            for r in token.rights:
                if r.dep_ == D_OBJ and r.pos_ == NOUN_TAG:
                    return r

    return None


def valid_hold_questions(text: str):
    doc = nlp(text)
    if doc[0].lemma_ != 'what':
        return False
    for token in doc:
        if token.dep_ == ROOT_TAG and token.lemma_ == 'hold':
            return True
    return False
