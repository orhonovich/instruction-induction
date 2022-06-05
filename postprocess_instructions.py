import argparse
import json
import re

from collections import Counter


INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']


def normalize_predicted_instruction(prediction):
    """Heuristic for postprocessing predictions.

    This includes a basic string cleanup (e.g., removing redundant spaces), as well as removing some common predicted
    prefixes that precede the instructions themselves, such as
    'probably to translate the input from English to French' -> 'translate the input from English to French'.
    """
    if prediction.startswith(':'):
        prediction = prediction.replace(':', '', 1)
    prediction = prediction.strip()
    for common_prefix in ['probably:', 'most likely:', 'probably', 'most likely', 'to']:
        if prediction.startswith(common_prefix):
            prediction = prediction.replace(f'{common_prefix}', '', 1).strip()
    prediction = prediction.strip().split('\n')[0]
    in_quotes = re.findall(r'\"([a-zA-Z\.,\'\\/]+ [a-zA-Z \.,\'\\/]+)\"', prediction)
    if in_quotes and len(in_quotes[0].split()) > 3:
        # Heuristic for extracting instructions that are fully inside quotation marks.
        # Short quotes are usually part of the generated instruction (i.e., if the instruction is
        # 'write 'positive' for if the sentiment of the input was positive and 'negative' otherwise')
        # rather than the full instruction.
        prediction = in_quotes[0]
    prediction = prediction.replace('\"', '').strip()
    return prediction


def group_instructions(predictions_dir, engine, task_name):
    """Post-process a given task's predictions and group identical predictions.

    The instructions are grouped for efficiency: it saves running the execution accuracy experiments multiple times
    over the same instruction.
    The processed instructions are saved to a json file under the provided predictions_dir.
    """
    with open(f'{predictions_dir}/{task_name}/{engine}_predictions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_predictions = {}
    predictions_list = []

    for id_, vals in data.items():
        prediction = vals['prediction']
        prediction = normalize_predicted_instruction(prediction)
        data[id_]['normalized_prediction'] = prediction
        predictions_list.append(prediction)
        if prediction_id := unique_predictions.get(prediction):
            data[id_]['unique_prediction_id'] = prediction_id
        else:
            prediction_id = len(unique_predictions) + 1
            data[id_]['unique_prediction_id'] = prediction_id
            unique_predictions[prediction] = prediction_id

    unique_predictions_counter = dict(Counter(predictions_list).most_common())
    groups_metadata = {'num_unique_predictions': len(unique_predictions),
                'unique_predictions_counter': unique_predictions_counter}
    with_groups_metadata = {'metadata': groups_metadata, 'examples': data}

    with open(f'{predictions_dir}/{task_name}/{engine}_prediction_groups.json', 'w', encoding='utf-8') as out_f:
        json.dump(with_groups_metadata, out_f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    INDUCTION_TASKS_STR = ','.join(INDUCTION_TASKS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default='text-davinci-002',
                        help='The OpenAI model that was used to generate instructions.')
    parser.add_argument('--predictions_dir', type=str, default='', required=True,
                        help='Path of the predicted instructions.')
    parser.add_argument('--tasks', type=str, default=INDUCTION_TASKS_STR, help='Tasks to postprocess')
    args = parser.parse_args()

    task_list = args.tasks.split(',')

    for induction_task in task_list:
        group_instructions(predictions_dir=args.predictions_dir,
                                     engine=args.engine,
                                     task_name=induction_task)