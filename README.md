# Zero-shot classification using pretrained NLP models

This repo explores the performance of modern pretrained NLP models on a simplistic zero-shot 
classification task. The goal is to predict the gender of an individual given their name. Really simple 
and really dull :)

It is truly amazing the fact that we can leverage knowledge which is freely available over the web, 
'compress' it as a language model (along with other auxiliary tasks) 
and then transfer that knowledge to any classification problem of interest with almost zero cost. 

My simplistic analysis takes into account two models:

- Huggingface's transformers library and more specifically its zero-shot classification
  pipeline. By default is uses `bart-large-mnli` 
  (check [ref.](https://discuss.huggingface.co/t/new-pipeline-for-zero-shot-text-classification/681) )
   
- The Universal Sentence Encoder (USE). Tensorflow Hub does not provide ready-to-use pipelines (correct me if I am wrong). 
  That's why I build a simplistic one manually which serves as a baseline comparison.
  
  
Ultimately, the goal is to see how well gender information is encoded into the LM-generated embeddings but also
how efficiently we can extract this information.

<br />

  
### Details regarding the tasks
Our goal is to get first names as inputs and predict:
- the gender associated (female / male)
- and its `origin` (region where it is more commonly used)


<br />

### Evaluation
To assess the evaluation of the proposed solutions I have extracted the most popular first names per region from
 the corresponding [Wikipedia page](https://en.wikipedia.org/wiki/List_of_most_popular_given_names#Male_names)

A processed version of these tables can be found in the `/data` folder



Note 1: Some names are repeated if they are 'popular' in multiple areas

Note 2: In some countries the most common names are less than 10

Note 3: Israel's unisex names have been omitted


<br />

### Project setup
The structure of this repo is simple because the investigation is organised into Jupyter 
notebooks. You can replicate the experiments by creating a Python 3 environment
(virtualenv, Pipenv, ...) and installing the following dependencies:

- `tensorflow`
- `transformers`
- `tensorflow_hub`
- `sklearn`
- `numpy`
- `pandas`
- `jupyter`
 
 Here, I am describing a setup using the awesome [poetry](https://python-poetry.org/) tool.
You can install it by following the [instructions](https://python-poetry.org/docs/)
 
 Then run:

`poetry install`

To create the virtual env and install the module then 

`poetry shell`

to activate the poetry virtual env

and finally you can give a (space separated) list of names for gender prediction by specifying the
backend NLP model as

`predict <name> --use4` e.g. 

`predict George Donald Mary Georgia` (no flag by default Huggingface backend)
or 

`predict Thanos Dimitris Aggeliki --use4` (Universal Sentence Encoder backend)

Note: The first time you run it, it will take some time to download all the 
necessary pretrained NLP models
<br />

### References 
(if you want to spend 2 hours reading NLP related material)

- [GPT-3 paper](https://arxiv.org/abs/2005.14165) 
- [Zero-Shot Learning in Modern NLP](https://joeddav.github.io/blog/2020/05/29/ZSL.html) 
- the awesome [Huggingface's Transformers library](https://github.com/huggingface/transformers)    


### Next steps

1. Evaluate on a larger corpus 
2. Try few-shot instead of just zero-shot
3. Use region-specific gender attributes e.g. `femme`, `homme` in French