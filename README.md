1. `data` : Folder to store datasets
2. `models` : Folder to store allennlp models
    1. `classifiers` : Models that do actually learning 
    2. `saliency_scorer` : Takes a trained model and return saliency scorers for inputs
    3. `rationale_extractors` : Models that take saliency scores and generate rationales
    4. `base_predictor.py` : Simple predictor to use with allennlp predict command as needed
3. `subcommands` : Subcommands to run saliency and rationale extractors since allennlp existingcommand semantics doesn't map quite as well to what we wanna do.
4. `training_config` : Contains jsonnet training configs to use with allennlp for each of the three types of models above.
5. `commands` : Actual bash scripts to run stuff.

Current setup is following  : 
1. First we run `commands/bert_train_script.sh` to train model A. This generates some output folder.
2. Next we run `commands/bert_saliency_script.sh` to get saliency scores on inputs. The output is stored in a subfolder of output_folder above.
3. Next we run `commands/bert_rationale_script.sh` to get rationales on saliency scores. The output is stored in a subsubfolder.
4. We train Model B on output generated in step 3 using `commands/bert_train_script.sh` . Here we need to pass and create a subsubsubfolder of Step 4.
