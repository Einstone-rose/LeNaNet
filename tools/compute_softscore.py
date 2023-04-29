import os
import sys
import json
import _pickle
import numpy as np
sys.path.append(os.getcwd())
sys.path.append('.')
sys.path.append('..')
import utils as utils
import config
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval


def get_file(train=False, val=False, test=False, question=False, answer=False):
    """ Get the correct question or answer file."""
    _file = utils.path_for(train=train, val=val, test=test,
                            question=question, answer=answer)
    with open(_file, 'r') as fd:
        _object = json.load(fd)
    return _object


def get_score(occurences):
    """ Average over all 10 choose 9 sets. """
    score_soft = occurences * 0.3
    score = score_soft if score_soft < 1.0 else 1.0
    return score


def preprocess_answer(answer):
    """ Mimicing the answer pre-processing with evaluation server. """
    dummy_vqa = lambda: None
    dummy_vqa.getQuesIds = lambda: None
    vqa_eval = VQAEval(dummy_vqa, None)

    answer = vqa_eval.processDigitArticle(
            vqa_eval.processPunctuation(answer))
    answer = answer.replace(',', '')
    return answer


def filter_answers(answers_dset, min_occurence):
    """ Filtering answers whose frequency is less than min_occurence. """
    occurence = {}
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        gtruth = ans_entry['multiple_choice_answer']
        gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= {} times: {}'.format(
                                min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurence, name, cache_root):
    """ Map answers to label. """
    label, label2ans, ans2label = 0, [], {}
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root)

    cache_file = os.path.join(cache_root, name+'_ans2label.pkl')
    _pickle.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name+'_label2ans.pkl')
    _pickle.dump(label2ans, open(cache_file, 'wb'))
    return ans2label


def compute_target(answers_dset, ans2label, name, cache_root):
    """ Augment answers_dset with soft score as label. """
    target = []
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels, scores = [], []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        target.append({
            'question_id': ans_entry['question_id'],
            'image_id': ans_entry['image_id'],
            'labels': labels,
            'scores': scores
        })

    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, name+'_target.pkl')
    _pickle.dump(target, open(cache_file, 'wb'))
    return target


if __name__ == '__main__':
    train_answers = get_file(train=True, answer=True)
    val_answers = get_file(val=True, answer=True)
    if not config.cp_data:
        train_answers = train_answers['annotations']
        val_answers = val_answers['annotations']

    answers = train_answers + val_answers
    print("filtering answers less than minmum occurence...")
    occurence = filter_answers(answers, config.min_occurence)
    print("create answers to integer labels...")
    ans2label = create_ans2label(occurence, 'trainval', config.cache_root)

    print("converting target for train and val answers...")
    compute_target(train_answers, ans2label, 'train', config.cache_root)
    compute_target(val_answers, ans2label, 'val', config.cache_root)
