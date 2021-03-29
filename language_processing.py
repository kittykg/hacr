from typing import List, Union

from nltk.corpus import wordnet
import spacy
from pyinflect import getInflection

from common import ActionPred

ROOT_TAG = 'ROOT'
VERB_TAG = 'VERB'
NOUN_TAG = 'NOUN'
PRON_TAG = 'PRON'
PROPN_TAG = 'PROPN'

N_SUBJ = 'nsubj'
D_OBJ = 'dobj'

PLACE_HOLDER = 'place_holder'

nlp = spacy.load('en_core_web_sm')


def check_synonyms(s1: str, s2: str) -> bool:
    if s1 == s2:
        return True

    for ss in wordnet.synsets(s1):
        for l in ss.lemmas():
            if l.name() == s2:
                return True
    for ss in wordnet.synsets(s2):
        for l in ss.lemmas():
            if l.name() == s1:
                return True
    return False


def valid_obj_word(t) -> bool:
    pos = t.pos_
    dep = t.dep_
    return pos in [NOUN_TAG, PRON_TAG] and (dep == N_SUBJ or dep == D_OBJ)


def valid_subj_word(t) -> bool:
    pos = t.pos_
    dep = t.dep_
    return pos in [PROPN_TAG] and (dep == N_SUBJ or dep == D_OBJ)


def _get_action_pred(doc) -> Union[ActionPred, None]:
    for token in doc:
        lemma = token.lemma_
        dep = token.dep_
        pos = token.pos_

        # Process the ROOT of the dependency tree
        if dep == ROOT_TAG and pos == VERB_TAG:
            pred_subj = None
            pred_obj = None

            for l in token.lefts:
                if pred_obj is None and valid_obj_word(l):
                    pred_obj = PLACE_HOLDER
                elif pred_subj is None and valid_subj_word(l):
                    pred_subj = l.lemma_

            for r in token.rights:
                if pred_obj is None and valid_obj_word(r):
                    pred_obj = PLACE_HOLDER
                elif pred_subj is None and valid_subj_word(r):
                    pred_subj = r.lemma_

            if pred_subj is not None and pred_obj is not None:
                action_g = getInflection(lemma, 'VBG')[0]
                return ActionPred(action_g, pred_subj, pred_obj)
            else:
                return None


def get_action_pred(text: str) -> Union[ActionPred, None]:
    return _get_action_pred(nlp(text))


def get_action_preds(texts: List[str]) -> List[Union[ActionPred, None]]:
    return list(map(lambda x: _get_action_pred(x), nlp.pipe(texts)))
