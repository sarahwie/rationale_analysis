Suggested use is to use Anaconda and do `pip install -r requirements.txt` .

1. `data` : Folder to store datasets
2. `Rationale_Analysis/models` : Folder to store allennlp models
    1. `classifiers` : Models that do actually learning 
    2. `saliency_scorer` : Takes a trained model and return saliency scorers for inputs
    3. `rationale_extractors` : Models that take saliency scores and generate rationales in form readable by `rationale_reader.py`
    4. `base_predictor.py` : Simple predictor to use with allennlp predict command as needed
3. `Rationale_Analysis/subcommands` : Subcommands to run saliency and rationale extractors since allennlp existingcommand semantics doesn't map quite as well to what we wanna do.
4. `Rationale_Analysis/training_config` : Contains jsonnet training configs to use with allennlp for each of the three types of models above.
5. `Rationale_Analysis/commands` : Actual bash scripts to run stuff.
6. `Rationale_Analysis/data/dataset_readers` : Contains dataset readers to work with Allennlp.
    1. `rationale_reader.py` : Code to load actual datasets (jsonl with 3 fields - document, query, label)
    2. `saliency_reader.py` : Read output of Saliency scorer to pass into rationale extractors.

Current setup is following  : 
1. First we run `commands/bert_train_script.sh` to train model A. This generates some output folder.
2. Next we run `commands/bert_saliency_script.sh` to get saliency scores on inputs. The output is stored in a subfolder of output_folder above.
3. Next we run `commands/bert_rationale_script.sh` to get rationales on saliency scores. The output is stored in a subsubfolder.
4. We train Model B on output generated in step 3 using `commands/bert_train_script.sh` . Here we need to pass and create a subsubsubfolder of Step 4.

Example Run to extract rationales for SST dataset using [CLS] attentions and min_attention heuristic - 

1. `CUDA_DEVICE=0 DATASET_NAME=SST DATA_BASE_PATH=Datasets/SST/data EXP_NAME=bert_base bash Rationale_Analysis/commands/bert_train_script.sh`
    Output generated in `outputs/bert_classification/SST/bert_base/` .

2. `CUDA_DEVICE=0 DATASET_NAME=SST DATA_BASE_PATH=Datasets/SST/data EXP_NAME=bert_base SALIENCY=wrapper bash Rationale_Analysis/commands/bert_saliency_script.sh`
    Output generate in `outputs/bert_classification/SST/bert_base/wrapper_saliency` .

3. `CUDA_DEVICE=0 DATASET_NAME=SST EXP_NAME=bert_base SALIENCY=wrapper RATIONALE=min_attention bash Rationale_Analysis/commands/bert_rationale_script.sh`
    Output generated in `outputs/bert_classification/SST/bert_base/wrapper_saliency/min_attention_rationale` .

4. `CUDA_DEVICE=0 DATASET_NAME=SST DATA_BASE_PATH=outputs/bert_classification/SST/bert_base/wrapper_saliency/min_attention_rationale EXP_NAME=bert_base/wrapper_saliency/min_attention_rationale/model_b bash Rationale_Analysis/commands/bert_train_script.sh`

You can also combine steps 3 and 4 in single script :

5. `CUDA_DEVICE=0 DATASET_NAME=SST EXP_NAME=bert_base SALIENCY=wrapper RATIONALE=min_attention bash Rationale_Analysis/commands/bert_rationale_and_train_model_b_script.sh`