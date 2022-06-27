# Instruction Induction
This repository contains the code and data of the paper:
"[Instruction Induction: From Few Examples to Natural Language Task Descriptions](https://arxiv.org/abs/2205.10782)"

## Data
The data for the instruction induction experiments, as well as for the execution accuracy evaluation,
are available in the `data` folder.

## Usage
### Setup
Install the required packages using `pip install -r requirements.txt`.

### Instructions Prediction
To run the instruction induction experiments, run the following command:
```
python induction.py \
--engine $OPENAI_ENGINE \
--organization $OPENAI_ORGANIZATION \
--api_key $OPENAI_API_KEY \
--data_dir $INPUT_DATA_DIR \
--out_dir $OUTPUT_DIR \
--max_tokens $MAX_TOKENS \
--tasks $TASK_LIST
```

where
- `$OPENAI_ENGINE` is the model used for inducing instructions (default: text-davinci-002).
- `$OPENAI_ORGANIZATION` is your OpenAI API organization.
- `$OPENAI_API_KEY` is your OpenAI API key.
- `$INPUT_DATA_DIR` is a path to the input data, should be in the format specified in `data/induction_input` 
(default: data/induction_input)
- `$OUTPUT_DIR` is the output dir path, will contain the predictions.
- `$MAX_TOKENS` is an upper bound on how many tokens the model can generate - `max_tokens` in the OpenAI API 
(default: 50).
- `$TASK_LIST` is a list of all tested tasks. Task names should correspond to the input files in `$INPUT_DATA_DIR`. 
Defaults to all tasks under `data/induction_input`.

#### Postprocessing
We apply a postprocessing protocol, which includes a basic cleanup for the generated instructions as well as grouping 
identical instructions, to speedup and reduce the cost of the execution accuracy experiments. 
To postprocess the generated instructions, run
```
python postprocess_instructions.py \
--engine $OPENAI_ENGINE \
--predictions_dir $PREDICTIONS_DIR \
--tasks $TASK_LIST
```

where
- `$OPENAI_ENGINE` is the name of the model that was used for inducing instructions (default: text-davinci-002).
- `$PREDICTIONS_DIR` is a path to a directory containing the predictions (the `out_dir`) passed to the induction script.
- `$TASK_LIST` is a list of all tested tasks. Task names should correspond to the input files in `$PREDICTIONS_DIR`. 
Defaults to all the instruction induction tasks.

### Evaluation
To measure the execution accuracy of the generated instructions, first run the following command:
```
python prepare_for_execution.py \
--model_name $OPENAI_ENGINE \
--execute_data_dir $EXECUTE_DATA_DIR \
--predictions_dir $PREDICTIONS_DIR \
--out_dir $OUTPUT_DIR \
--tasks $TASK_LIST
```

where
- `$OPENAI_ENGINE` is the name of the model that was used for inducing instructions (default: text-davinci-002).
- `$EXECUTE_DATA_DIR` is the path of the (without instructions) execution set (default: data/raw/execute).
- `$PREDICTIONS_DIR` is a path of a directory containing the predictions (after postprocessing).
- `$OUTPUT_DIR` will contain the execution accuracy experiment inputs.
- `$TASK_LIST` is a list of all evaluated tasks. Task names should correspond to the input files in `$INPUT_DATA_DIR`. 
Defaults to all tasks under `data/induction_input`.

Next, to execute the instructions, run
```
python execute_instructions.py \
--execution_engine $OPENAI_EXECUTION_ENGINE \
--instruction_generation_model $INSTRUCTION_GENERATION_MODEL \
--organization $OPENAI_ORGANIZATION \
--api_key $OPENAI_API_KEY \
--input_dir $INPUT_DATA_DIR \
--out_dir $OUTPUT_DIR \
--max_tokens $MAX_TOKENS \
--tasks $TASK_LIST
```

where
- `$OPENAI_EXECUTION_ENGINE` is the model that will be used for executing the instructions 
(default: text-davinci-002).
- `$INSTRUCTION_GENERATION_MODEL` is the evaluated model - the model that was used to generate instructions
(default: text-davinci-002).
- `$OPENAI_ORGANIZATION` is your OpenAI API organization.
- `$OPENAI_API_KEY` is your OpenAI API key.
- `$INPUT_DATA_DIR` is a path of the input execution accuracy data.
- `$OUTPUT_DIR` is the output dir path, will contain the execution accuracy predictions.
- `$MAX_TOKENS` is an upper bound on how many tokens the model can generate - `max_tokens` in the OpenAI API 
(default: 30).
- `$TASK_LIST` is a list of all tested tasks. Task names should correspond to the input files in `$INPUT_DATA_DIR`. 
Defaults to all tasks under `data/induction_input`.

Finally, to obtain the execution accuracy scores, run the following command:
```
python evaluate.py \
--instruction_generation_model $INSTRUCTION_GENERATION_MODEL \
--execution_input_dir $INPUT_DATA_DIR \
--predictions_dir $PREDICTIONS_DIR \
--tasks $TASK_LIST
```

where
- `$INSTRUCTION_GENERATION_MODEL` is the evaluated model - the model that was used to generate instructions
(default: text-davinci-002).
- `$INPUT_DATA_DIR` is a path of the input execution accuracy data.
- `$PREDICTIONS_DIR` is a path containing the instructions execution outputs.
- `$TASK_LIST` is a list of all tested tasks. Task names should correspond to the input files in `$INPUT_DATA_DIR`. 
Defaults to all tasks under `data/induction_input`.

## Citation

```
@misc{honovich2022induction,
      title={Instruction Induction: From Few Examples to Natural Language Task Descriptions},
      author={Honovich, Or and Shaham, Uri and Bowman, Samuel R. and Levy, Omer},
      year={2022},
      eprint={2205.10782},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```