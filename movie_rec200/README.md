---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:19900
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 'Los Frikis Drama 2024 Gustavo idolizes his older brother Paco
    and his punk, "Frikis" bandmates. When word reaches the Frikis of a potential
    reprieve from the effects of the economic crisis, they do the now unthinkable:
    deliberately inject themselves with HIV to live at a government-run treatment
    home. It''s there that they create their own utopia to live and play music freely.'
  sentences:
  - Feast of the Fallen Angels Animation TV Movie Horror 1985 Datenshi-tachi no Kyōen
    is an OAV (1985)  where the younger sister of a family (who gets in way over their
    heads) ends up owing the yakuza money. Because she can't pay, she is taken by
    the gang members to be used as a sexual toy. Then she vows to take gruesome revenge
    upon her perpetrators.
  - Borderline Comedy Horror Thriller 2025 A helplessly romantic sociopath escapes
    from a mental institution and invades the home of a '90s pop superstar. He just
    wants to be loved; she just wants to survive.
  - Oppenheimer Drama History 2023 The story of J. Robert Oppenheimer's role in the
    development of the atomic bomb during World War II.
- source_sentence: Terrifier 3 Horror Thriller 2024 Five years after surviving Art
    the Clown's Halloween massacre, Sienna and Jonathan are still struggling to rebuild
    their shattered lives. As the holiday season approaches, they try to embrace the
    Christmas spirit and leave the horrors of the past behind. But just when they
    think they're safe, Art returns, determined to turn their holiday cheer into a
    new nightmare. The festive season quickly unravels as Art unleashes his twisted
    brand of terror, proving that no holiday is safe.
  sentences:
  - Sex Education Mistresses Comedy 1973 An erotic comedy depicting the sex adventures
    of  young wives living in an apartment complex, that are not satisfied with just
    having sex with their husbands.
  - Playboy Video Playmate Calendar 2000 Documentary 1999 Playboy Video Playmate Calendar
    2000
  - 'Sniper: The Last Stand Action Thriller 2025 To stop an arms dealer from unleashing
    a deadly superweapon, Ace sniper Brandon Beckett and Agent Zero are deployed to
    Costa Verde to lead a group of elite soldiers against an unrelenting militia.
    Taking an untested sniper under his wing, Beckett faces his newest challenge:
    giving orders instead of receiving them. With both time and ammo running low in
    a race to save humanity, the team must overcome all odds just to survive.'
- source_sentence: Interstellar Adventure Drama Science Fiction 2014 The adventures
    of a group of explorers who make use of a newly discovered wormhole to surpass
    the limitations on human space travel and conquer the vast distances involved
    in an interstellar voyage.
  sentences:
  - Your Fault Romance Drama 2024 The love between Noah and Nick seems unwavering
    despite their parents' attempts to separate them. But his job and her entry into
    college open up their lives to new relationships that will shake the foundations
    of both their relationship and the Leister family itself.
  - Azrael Action Horror Thriller 2024 In a world where no one speaks, a devout female
    hunts down a young woman who has escaped her imprisonment. Recaptured by its ruthless
    leaders, Azrael is due to be sacrificed to pacify an ancient evil deep within
    the surrounding wilderness.
  - The Parenting Horror Comedy 2025 Boyfriends Josh and Rohan plan a weekend getaway
    to introduce their parents, only to discover that their rental is home to a 400-year-old
    poltergeist.
- source_sentence: Alarum Action Crime Thriller 2025 Two married spies caught in the
    crosshairs of an international intelligence network will stop at nothing to obtain
    a critical asset. Joe and Lara are agents living off the grid whose quiet retreat
    at a winter resort is blown to shreds when members of the old guard suspect the
    two may have joined an elite team of rogue spies, known as Alarum.
  sentences:
  - Love Hurts Action Comedy Romance 2025 A realtor is pulled back into the life he
    left behind after his former partner-in-crime resurfaces with an ominous message.
    With his crime-lord brother also on his trail, he must confront his past and the
    history he never fully buried.
  - Dirty Angels Action Drama Thriller War 2024 During the United States' 2021 withdrawal
    from Afghanistan, a group of female soldiers posing as medical relief are sent
    back in to rescue a group of kidnapped teenagers caught between ISIS and the Taliban.
  - 'Bad Boys: Ride or Die Action Comedy Crime Thriller Adventure 2024 After their
    late former Captain is framed, Lowrey and Burnett try to clear his name, only
    to end up on the run themselves.'
- source_sentence: I, the Executioner Action Crime 2024 The veteran detective Seo
    Do-cheol and his team at Major Crimes, relentless in their pursuit of criminals,
    join forces with rookie cop Park Sun-woo to track down a serial killer who has
    plunged the nation into turmoil.
  sentences:
  - 'Spider-Man: No Way Home Action Adventure Science Fiction 2021 Peter Parker is
    unmasked and no longer able to separate his normal life from the high-stakes of
    being a super-hero. When he asks for help from Doctor Strange the stakes become
    even more dangerous, forcing him to discover what it truly means to be Spider-Man.'
  - The Island Action Crime Thriller 2023 When his brother is killed, LAPD officer
    Mark leaves the city to return to the island he grew up on. Seeking answers and
    ultimately vengeance, he soon finds himself in a bloody battle with the corrupt
    tycoon who's taken over the island paradise.
  - 'Spider-Man: Across the Spider-Verse Animation Action Adventure Science Fiction
    2023 After reuniting with Gwen Stacy, Brooklyn’s full-time, friendly neighborhood
    Spider-Man is catapulted across the Multiverse, where he encounters the Spider
    Society, a team of Spider-People charged with protecting the Multiverse’s very
    existence. But when the heroes clash on how to handle a new threat, Miles finds
    himself pitted against the other Spiders and must set out on his own to save those
    he loves most.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'I, the Executioner Action Crime 2024 The veteran detective Seo Do-cheol and his team at Major Crimes, relentless in their pursuit of criminals, join forces with rookie cop Park Sun-woo to track down a serial killer who has plunged the nation into turmoil.',
    'Spider-Man: No Way Home Action Adventure Science Fiction 2021 Peter Parker is unmasked and no longer able to separate his normal life from the high-stakes of being a super-hero. When he asks for help from Doctor Strange the stakes become even more dangerous, forcing him to discover what it truly means to be Spider-Man.',
    'Spider-Man: Across the Spider-Verse Animation Action Adventure Science Fiction 2023 After reuniting with Gwen Stacy, Brooklyn’s full-time, friendly neighborhood Spider-Man is catapulted across the Multiverse, where he encounters the Spider Society, a team of Spider-People charged with protecting the Multiverse’s very existence. But when the heroes clash on how to handle a new threat, Miles finds himself pitted against the other Spiders and must set out on his own to save those he loves most.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 19,900 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                            |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                            |
  | details | <ul><li>min: 7 tokens</li><li>mean: 65.7 tokens</li><li>max: 207 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 64.6 tokens</li><li>max: 207 tokens</li></ul> | <ul><li>min: 0.07</li><li>mean: 0.32</li><li>max: 0.55</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | label                            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
  | <code>Le Clitoris Animation Documentary 2016 Women are lucky, they get to have the only organ in the human body dedicated exclusively for pleasure: the clitoris! In this humorous and instructive animated documentary, find out its unrecognized anatomy and its unknown herstory.</code> | <code>Fast X Action Crime Thriller Adventure Mystery 2023 Over many missions and against impossible odds, Dom Toretto and his family have outsmarted, out-nerved and outdriven every foe in their path. Now, they confront the most lethal opponent they've ever faced: A terrifying threat emerging from the shadows of the past who's fueled by blood revenge, and who is determined to shatter this family and destroy everything—and everyone—that Dom loves, forever.</code> | <code>0.22680924677330516</code> |
  | <code>Baby Face Drama 1933 A young woman uses her body and her sexuality to help her climb the social ladder, but soon begins to wonder if her new status will ever bring her happiness.</code>                                                                                             | <code>Laila Comedy Romance 2025 Sonu Model, a renowned beautician from the old city, is forced to disguise himself as Laila, leading to a series of comedic, romantic, and action-packed events. Chaos ensues in this hilarious laugh riot</code>                                                                                                                                                                                                                                 | <code>0.21629780530929565</code> |
  | <code>The Vigilante Thriller Action 2023 Returning from Afghanistan, Jessica, a Spec OPS Marine, finds herself in a war she never imagined and discovers middle America suburbia has changed when her thirteen year old sister, Aimee, is abducted by sex traffickers.</code>               | <code>The Bayou Thriller Horror Action 2025 Vacation turns disaster when Houston grad Kyle and her friends survive a plane crash in the desolate Louisiana everglades, only to discover there's something way more dangerous lurking in the shallows.</code>                                                                                                                                                                                                                      | <code>0.5150914500588957</code>  |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4019 | 500  | 0.0023        |
| 0.8039 | 1000 | 0.0012        |


### Framework Versions
- Python: 3.11.12
- Sentence Transformers: 3.4.1
- Transformers: 4.50.3
- PyTorch: 2.6.0+cu124
- Accelerate: 1.5.2
- Datasets: 3.5.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->