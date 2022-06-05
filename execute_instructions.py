import argparse
import json
import openai
from pathlib import Path


INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']


def run_execution_accuracy(execution_engine, instruction_generation_model, task_name, openai_organization,
                        openai_api_key, input_dir, out_dir, max_tokens=30):
    with open(f'{input_dir}/{instruction_generation_model}/{task_name}.json', encoding='utf-8') as f_examples:
        data = json.load(f_examples)

    openai.organization = openai_organization
    openai.api_key = openai_api_key

    output_ = dict()

    parameters = {
        'max_tokens': max_tokens,
        'top_p': 0,
        'temperature': 1,
        'logprobs': 5,
        'engine': execution_engine
    }
    for instruction_id, instruction_data in data.items():
        d = {}
        d['instruction'] = instruction_data['instruction']
        d['prediction_counter'] = instruction_data['prediction_counter']
        instruction_outputs = {}
        test_examples = instruction_data['test_inputs']
        for id_, example in test_examples.items():
            prompt = example['prompt']
            parameters['prompt'] = prompt

            response = openai.Completion.create(**parameters)

            instruction_outputs[id_] = dict()
            instruction_outputs[id_]['prompt'] = prompt
            instruction_outputs[id_]['prediction'] = response.choices[0].text

            if int(id_) % 100 == 0:
                print(f'generated {id_} predictions with OpenAI {execution_engine}')

        d['instruction_outputs'] = instruction_outputs
        output_[instruction_id] = d

    output_path = f'{out_dir}/{instruction_generation_model}'
    Path(output_path).mkdir(exist_ok=True)

    with open(f'{output_path}/{task_name}_execution.json', 'w', encoding='utf-8') as f_predictions:
        json.dump(output_, f_predictions, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    INDUCTION_TASKS_STR = ','.join(INDUCTION_TASKS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution_engine", type=str, default='text-davinci-002', help='The execution engine.')
    parser.add_argument("--instruction_generation_model", type=str, default='text-davinci-002',
                        help='The model used to generate the instruciton, i.e, the evaluated model.')
    parser.add_argument('--organization', type=str, required=True, help='Organization for the OpenAI API.')
    parser.add_argument('--api_key', type=str, required=True, help='API key for the OpenAI API')
    parser.add_argument('--input_dir', type=str, required=True, help='Path of the input execution accuracy data.')
    parser.add_argument('--out_dir', type=str, default='', required=True, help='Path for saving the predictions.')
    parser.add_argument('--max_tokens', type=int, default=30, help='Max number of tokens to generate.')
    parser.add_argument('--tasks', type=str, default=INDUCTION_TASKS_STR,
                        help='Tasks for execution accuracy evaluation.')
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)
    task_list = args.tasks.split(',')

    for induction_task in task_list:
        run_execution_accuracy(execution_engine=args.execution_engine,
                               instruction_generation_model=args.instruction_generation_model,
                               task_name=induction_task,
                               openai_organization=args.organization,
                               openai_api_key=args.api_key,
                               input_dir=args.input_dir,
                               out_dir=args.out_dir,
                               max_tokens=args.max_tokens)