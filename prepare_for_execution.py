import argparse
import json
from pathlib import Path

TASK_TO_ANSWERS = {'rhymes': 'other_rhymes', 'translation_en-de': 'possible_translations',
                   'translation_en-es': 'possible_translations', 'translation_en-fr': 'possible_translations',
                   'sentence_similarity': 'possible_outputs', 'word_in_context': 'possible_outputs'}


INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']


def create_task_examples(model_name, task_name, execute_data_dir, predictions_dir, out_dir, answers_key=None):
    predictions_path = f'{predictions_dir}/{task_name}/{model_name}_prediction_groups.json'
    with open(predictions_path, 'r', encoding='utf-8') as predictions_f:
        predictions = json.load(predictions_f)
    predictions_counter = predictions['metadata']['unique_predictions_counter']

    with open(f'{execute_data_dir}/{task_name}.json', 'r', encoding='utf-8') as test_f:
        test_data = json.load(test_f)['examples']

    examples = {}
    for prediction, counter in predictions_counter.items():
        d = {}
        d['instruction'] = prediction
        d['prediction_counter'] = counter
        prediction_prompts = {}
        for id_, example in test_data.items():
            input_ = example['input']
            if answers_key:  # multiple possible answers
                answers = example[answers_key]
            else:  # only one answer
                answers = [example['output']]
            prompt = f'Instruction: {prediction}\nInput: {input_}\nOutput:'
            prediction_prompts[id_] = {'prompt': prompt, 'test_input': input_, 'answers': answers, 'test_id': id_}
        d['test_inputs'] = prediction_prompts
        examples[str(len(examples) + 1)] = d

    output_path = Path(f'{out_dir}/{model_name}')
    output_path.mkdir(exist_ok=True)

    with open(output_path.joinpath(f'{task_name}.json'), 'w', encoding='utf-8') as out_f:
        json.dump(examples, out_f, indent=2, ensure_ascii=False)


def create_cause_and_effect_examples(model_name, execute_data_dir, predictions_dir, out_dir):
    predictions_path = f'{predictions_dir}/cause_and_effect/{model_name}_prediction_groups.json'
    with open(predictions_path, 'r', encoding='utf-8') as predictions_f:
        predictions = json.load(predictions_f)
    predictions_counter = predictions['metadata']['unique_predictions_counter']

    with open(f'{execute_data_dir}/cause_and_effect.json', 'r', encoding='utf-8') as test_f:
        test_data = json.load(test_f)['examples']

    examples = {}
    for prediction, counter in predictions_counter.items():
        d = {}
        d['instruction'] = prediction
        d['prediction_counter'] = counter
        prediction_prompts = {}
        for id_, example in test_data.items():
            cause = example['cause']
            effect = example['effect']
            option1 = f'Sentence 1: {cause} Sentence 2: {effect}'
            option2 = f'Sentence 1: {effect} Sentence 2: {cause}'
            answers = [cause]
            for input_ in [option1, option2]:
                prompt = f'Instruction: {prediction}\nInput: {input_}\nOutput:'
                prediction_prompts[str(len(prediction_prompts)+1)] = \
                    {'prompt': prompt, 'test_input': input_, 'answers': answers, 'test_id': id_}
        d['test_inputs'] = prediction_prompts
        examples[str(len(examples) + 1)] = d

    output_path = Path(f'{out_dir}/{model_name}')
    output_path.mkdir(exist_ok=True)

    with open(output_path.joinpath('cause_and_effect.json'), 'w', encoding='utf-8') as out_f:
        json.dump(examples, out_f, indent=2, ensure_ascii=False)


def create_common_concept(model_name, execute_data_dir, predictions_dir, out_dir):
    predictions_path = f'{predictions_dir}/common_concept/{model_name}_prediction_groups.json'
    with open(predictions_path, 'r', encoding='utf-8') as predictions_f:
        predictions = json.load(predictions_f)
    predictions_counter = predictions['metadata']['unique_predictions_counter']

    with open(f'{execute_data_dir}/common_concept.json', 'r', encoding='utf-8') as test_f:
        test_data = json.load(test_f)['examples']

    examples = {}
    for prediction, counter in predictions_counter.items():
        d = {}
        d['instruction'] = prediction
        d['prediction_counter'] = counter
        prediction_prompts = {}
        for id_, example in test_data.items():
            input_ = ', '.join(example['items'])
            answers = example['all_common_concepts']
            prompt = f'Instruction: {prediction}\nInput: {input_}\nOutput:'
            prediction_prompts[str(len(prediction_prompts)+1)] = \
                {'prompt': prompt, 'test_input': input_, 'answers': answers, 'test_id': id_}
        d['test_inputs'] = prediction_prompts
        examples[str(len(examples) + 1)] = d

    output_path = Path(f'{out_dir}/{model_name}')
    output_path.mkdir(exist_ok=True)

    with open(output_path.joinpath('common_concept.json'), 'w', encoding='utf-8') as out_f:
        json.dump(examples, out_f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    INDUCTION_TASKS_STR = ','.join(INDUCTION_TASKS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='text-davinci-002',
                        help='The OpenAI model that was used to generate instructions.')
    parser.add_argument('--execute_data_dir', type=str, default='data/raw/execute',
                        help='Path of the raw (without instructions) execution set.')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Path of the predicted instructions to evaluate.')
    parser.add_argument('--out_dir', type=str, default='', required=True,
                        help='Path for saving the created execution experiment inputs.')
    parser.add_argument('--tasks', type=str, default=INDUCTION_TASKS_STR, help='Tasks for evaluation.')
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)
    task_list = args.tasks.split(',')

    for induction_task in task_list:
        if induction_task not in ['cause_and_effect', 'common_concept']:
            task_answers_key = TASK_TO_ANSWERS.get(induction_task)
            create_task_examples(model_name=args.model_name,
                                 task_name=induction_task,
                                 execute_data_dir=args.execute_data_dir,
                                 predictions_dir=args.predictions_dir,
                                 out_dir=args.out_dir,
                                 answers_key=task_answers_key)
        elif induction_task == 'cause_and_effect':
            create_cause_and_effect_examples(model_name=args.model_name,
                                             execute_data_dir=args.execute_data_dir,
                                             predictions_dir=args.predictions_dir,
                                             out_dir=args.out_dir)
        elif induction_task == 'common_concept':
            create_common_concept(model_name=args.model_name,
                                  execute_data_dir=args.execute_data_dir,
                                  predictions_dir=args.predictions_dir,
                                  out_dir=args.out_dir)