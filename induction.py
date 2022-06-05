import argparse
import json
import openai
from pathlib import Path

INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']


def generate_instructions(engine, task_name, openai_organization, openai_api_key, data_dir, out_dir, max_tokens=50):
    with open(f'{data_dir}/{task_name}.json', encoding='utf-8') as f_examples:
        data = json.load(f_examples)
    examples = data['examples']

    openai.organization = openai_organization
    openai.api_key = openai_api_key

    output = dict()

    parameters = {
        'max_tokens': max_tokens,
        'top_p': 0,  # greedy
        'temperature': 1,
        'logprobs': 5,
        'engine': engine
    }

    for id_, example in examples.items():
        prompt = example['input']
        parameters['prompt'] = example['input']

        response = openai.Completion.create(**parameters)

        output[id_] = dict()
        output[id_]['input'] = prompt
        output[id_]['prediction'] = response.choices[0].text

        metadata = dict()
        metadata['logprobs'] = response.choices[0]['logprobs']
        metadata['finish_reason'] = response.choices[0]['finish_reason']
        output[id_]['metadata'] = metadata

        if int(id_) % 100 == 0:
            print(f'generated {id_} predictions with engine {engine}')

    output_path = f'{out_dir}/{task_name}'
    Path(output_path).mkdir(exist_ok=True)

    output_with_metadata_path = f'{output_path}/{engine}_predictions_with_metadata.json'
    with open(output_with_metadata_path, 'w', encoding='utf-8') as f_predictions_with_metadata:
        json.dump(output, f_predictions_with_metadata, indent=2, ensure_ascii=False)

    for id_ in output:
        del output[id_]['metadata']

    output_no_metadata_path = f'{output_path}/{engine}_predictions.json'
    with open(output_no_metadata_path, 'w', encoding='utf-8') as f_predictions:
        json.dump(output, f_predictions, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    INDUCTION_TASKS_STR = ','.join(INDUCTION_TASKS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default='text-davinci-002',
                        help='The OpenAI model that will be used to generate instructions.')
    parser.add_argument('--organization', type=str, required=True, help='Organization for the OpenAI API.')
    parser.add_argument('--api_key', type=str, required=True, help='API key for the OpenAI API')
    parser.add_argument('--data_dir', type=str, default='data/induction_input', help='Path of the input data.')
    parser.add_argument('--out_dir', type=str, default='', required=True, help='Path for saving the predictions.')
    parser.add_argument('--max_tokens', type=int, default=50, help='Max number of tokens to generate.')
    parser.add_argument('--tasks', type=str, default=INDUCTION_TASKS_STR, help='Tasks for instructions generation')
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)
    task_list = args.tasks.split(',')

    for induction_task in task_list:
        generate_instructions(engine=args.engine,
                              task_name=induction_task,
                              openai_organization=args.organization,
                              openai_api_key=args.api_key,
                              data_dir=args.data_dir,
                              out_dir=args.out_dir,
                              max_tokens=args.max_tokens)