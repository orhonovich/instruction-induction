import argparse
import json
import string
from collections import Counter
import re

TASK_TO_METRIC = {'common_concept': 'f1', 'informal_to_formal': 'f1', 'orthography_starts_with': 'es',
                  'taxonomy_animal': 'es', 'synonyms': 'contains'}


INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']


def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(str.maketrans('', '', string.punctuation))

    return prediction


def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_em_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(ground_truth, lowercase=True)
    return prediction_normalized == ground_truth_normalized


def get_exact_set_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True).split()
    ground_truth_normalized = normalize_prediction(ground_truth, lowercase=True).split()
    return int(set(prediction_normalized) == set(ground_truth_normalized))


def get_contains_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(ground_truth, lowercase=True)
    if re.search(r'\b({0})\b'.format(ground_truth_normalized), prediction_normalized):
        return 1


def get_multi_answer_em(prediction, answers):
    for answer in answers:
        if get_em_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_f1(prediction, answers):
    f1_scores = []
    for answer in answers:
        f1_scores.append(get_f1_score(prediction, answer))
    return max(f1_scores)


def get_multi_answer_exact_set(prediction, answers):
    for answer in answers:
        if get_exact_set_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_contains(prediction, answers):
    for answer in answers:
        if get_contains_score(prediction, answer) == 1:
            return 1
    return 0


def get_weighted_task_score(scored_predictions):
    """Get the task overall score, weighted according to the instructions prediction frequencies."""
    id_to_counter = {}
    id_to_score = {}
    for instruction_id, instruction_data in scored_predictions.items():
        id_to_counter[instruction_id] = instruction_data['prediction_counter']
        id_to_score[instruction_id] = instruction_data['average_score']
    num_instructions = sum(list(id_to_counter.values()))
    weighted_score = 0
    for id_, count in id_to_counter.items():
        weighted_score += (id_to_score[id_] * count) / num_instructions
    return weighted_score


def save_predictions_execution_accuracy(instruction_generation_model, task_name, execution_input_dir, predictions_dir):
    # load examples
    with open(f'{execution_input_dir}/{instruction_generation_model}/{task_name}.json', encoding='utf-8') as f_task:
        examples = json.load(f_task)

    # load predictions
    with open(f'{predictions_dir}/{instruction_generation_model}/{task_name}_execution.json', encoding='utf-8') \
            as f_predictions:
        predictions = json.load(f_predictions)

    # score predictions
    for instruction_id, instruction_data in examples.items():
        instruction_outputs = predictions[instruction_id]['instruction_outputs']
        instruction_scores = []
        for input_id, input_ in instruction_data['test_inputs'].items():
            answers = input_['answers']
            prediction = instruction_outputs[input_id]['prediction']
            task_metric = TASK_TO_METRIC.get(task_name, 'em')
            if task_metric == 'f1':
                score = get_multi_answer_f1(prediction=prediction, answers=answers)
            elif task_metric == 'es':
                score = get_multi_answer_exact_set(prediction=prediction, answers=answers)
            elif task_metric == 'contains':
                score = get_multi_answer_contains(prediction=prediction, answers=answers)
            else:  # EM
                score = get_multi_answer_em(prediction=prediction, answers=answers)
            predictions[instruction_id]['instruction_outputs'][input_id]['answers'] = answers
            predictions[instruction_id]['instruction_outputs'][input_id]['score'] = score
            instruction_scores.append(score)
        avg_score = sum(instruction_scores) / len(instruction_scores)
        predictions[instruction_id]['average_score'] = avg_score

    predictions['weighted_task_score'] = get_weighted_task_score(predictions)

    # save the scored predictions
    with open(f'{predictions_dir}/{instruction_generation_model}/{task_name}_with_scores.json', 'w', encoding='utf-8') \
            as f_scored_predictions:
        json.dump(predictions, f_scored_predictions, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    INDUCTION_TASKS_STR = ','.join(INDUCTION_TASKS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction_generation_model", type=str, default='text-davinci-002',
                        help='The model used to generate the instruciton, i.e, the evaluated model.')
    parser.add_argument('--execution_input_dir', type=str, required=True,
                        help='Path of the input execution accuracy data.')
    parser.add_argument('--predictions_dir', type=str, default='', required=True,
                        help='Path of the execution predictions.')
    parser.add_argument('--tasks', type=str, default=INDUCTION_TASKS_STR,
                        help='Tasks for execution accuracy evaluation.')
    args = parser.parse_args()

    task_list = args.tasks.split(',')

    for induction_task in task_list:
        save_predictions_execution_accuracy(instruction_generation_model=args.instruction_generation_model,
                                            task_name=induction_task,
                                            execution_input_dir=args.execution_input_dir,
                                            predictions_dir=args.predictions_dir)