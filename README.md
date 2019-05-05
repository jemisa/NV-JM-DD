# NV-JM-DD

A collection of resources related to Project Name_TBD

## Introduction, Theory  

### Must Watch
CS224n: Natural Language Processing with Deep Learning  
http://onlinehub.stanford.edu/cs224

### NLP, General 
Repository to track the progress in Natural Language Processing  
https://github.com/sebastianruder/NLP-progress

Modern Deep Learning Techniques Applied to Natural Language Processing  
https://nlpoverview.com/

### Word/Sentence Contextual representation, Language Models 
Methods for computing sentence representations from pretrained word embeddings without any additional training  
https://code.fb.com/ml-applications/random-encoders/

Contextual Word Representations  
https://arxiv.org/abs/1902.06006

Sentence and Contextualized Word Representations  
https://www.youtube.com/watch?v=Ioqrw4sCcwQ

Introduction and discussion on various Generalized Language Models  
https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html

Cr5, model for crosslingual document embedding
Input: text in any of 28 languages
Output: language-independent vector representation, so you can compare text across langs
Pre-trained model and API: https://github.com/epfl-dlab/Cr5 (Numpy/cython/Intel stuff)
Paper: "Crosslingual Document Embeddingas Reduced-Rank Ridge Regression" https://dlab.epfl.ch/people/west/pub/Josifoski-Paskov-Paskov-Jaggi-West_WSDM-19.pdf


Contextual word representations derived from large-scale neural language models are successful across a diverse set of NLP tasks, suggesting that they encode useful and transferable features of language. To shed light on the linguistic knowledge they capture, we study the representations produced by several recent pretrained contextualizers (variants of ELMo, the OpenAI transformer language model, and BERT) with a suite of sixteen diverse probing tasks. 
Linguistic Knowledge and Transferability of Contextual Representations - https://arxiv.org/abs/1903.08855
-->
"our analysis of patterns in the transferability of contextualizer layers shows that 
1) the lowest layer of LSTMs encodes the most transferable features, while transformers’ middle layers are most transferable. 
2) We find that higher layers in LSTMs are more task-specific (and thus less general)
3) while transformer layers do not exhibit this same monotonic increase in task-specificity. 
Prior work  has  suggested  that  higher-level  contextualizer layers may be expressly encoding higher-level semantic information. Instead, it seems likely that certain  high-level  semantic  phenomena  are  incidentally  useful  for  the  contextualizer’s  pretraining task,  leading to their presence in higher layers.    
4) Lastly,  we  find  that  bidirectional  language model  pretraining  yields  representations  that  are more transferable in general than eleven other candidate pretraining tasks"
   
### Knowledge and Representations
A Unified Theory of Inference for Text Understanding  
https://www2.eecs.berkeley.edu/Pubs/TechRpts/1987/CSD-87-339.pdf

Knowledge Graph Intros    
https://towardsdatascience.com/the-data-fabric-for-machine-learning-part-2-building-a-knowledge-graph-2fdd1370bb0a
https://medium.com/@sderymail/challenges-of-knowledge-graph-part-1-d9ffe9e35214
Better Knowledge Graphs Through Probabilistic Graphical Models - https://www.youtube.com/watch?v=z_VzaNy36xE

A general framework for learning context-based action representation for Reinforcement Learning  
https://arxiv.org/abs/1902.01119

Deep Learning with Knowledge Graphs  
https://medium.com/octavian-ai/deep-learning-with-knowledge-graphs-3df0b469a61a

Geometry of Thought  
http://nautil.us/blog/new-evidence-for-the-geometry-of-thought

Geometry of Meaning  
https://www.youtube.com/watch?v=L0X9mEe9aY0&app=desktop
  
Interacting Conceptual Spaces    
https://arxiv.org/abs/1608.01402  

Decentralised Knowledge Graph  
https://www.youtube.com/watch?v=hm9ibPZOUcw
https://fosdem.org/2019/schedule/event/graph_weaviate_knowledge_graph/

Text Generation from Knowledge Graphs with Graph Transformers - https://arxiv.org/abs/1904.02342
and the author presenting the paper: https://www.youtube.com/watch?v=BiRyvB2NmCM


Understanding Graph Attention Network (GAT) - https://www.dgl.ai/blog/2019/02/17/gat.html

### Sign Language
Sign Language & Linguistics  
https://benjamins.com/catalog/sll

### Benchmarks

Introducing SuperGLUE: A New Hope Against Muppetkind - https://medium.com/@wang.alex.c/introducing-superglue-a-new-hope-against-muppetkind-2779fd9dcdd5
 a benchmark for evaluating general-purpose NLP models based on evaluation on a diverse set of language understanding tasks.

## models
### Transformer
The Annotated Transformer - http://nlp.seas.harvard.edu/2018/04/03/attention.html
The Illustrated Transformer - https://jalammar.github.io/illustrated-transformer/
How Transformers Work - https://towardsdatascience.com/transformers-141e32e69591

The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning) - http://jalammar.github.io/illustrated-bert/
Transformer-XL Explained:  - https://towardsdatascience.com/transformer-xl-explained-combining-transformers-and-rnns-into-a-state-of-the-art-language-model-c0cfe9e5a924




### Sparse Transformer
https://openai.com/blog/sparse-transformer/

that predicts what comes next in a sequence,  30x longer than possible previously.


it does text part (7.2 in the paper )

"parse Transformers and showed they attain equivalent or better performance on density modeling of long sequences than standard Transformers while requiring significantly fewer operations. "

## Tool, Code, Application  

### NLP
StanfordNLP, a Python natural language analysis package  
https://stanfordnlp.github.io/stanfordnlp/

OpenAI GPT2  
https://openai.com/blog/better-language-models/  

PyTorch Pretrained BERT: The Big & Extending Repository of pretrained Transformers  
https://github.com/huggingface/pytorch-pretrained-BERT  


Generalized Language Models, coverts in four parts CoVe, ELMo & Cross-View Training, Part 2: ULMFiT & OpenAI GPT, Part 3: BERT & OpenAI GPT-2
and Part 4: Common Tasks & Datasets
https://www.topbots.com/generalized-language-models-cove-elmo/

Open WebText – an open source effort to reproduce OpenAI’s WebText dataset, to train GPT and other Transformers  models (sparse, XL,...)
https://skylion007.github.io/OpenWebTextCorpus/

### Knowledge 
How NASA experiments with knowledge discovery  
https://linkurio.us/blog/how-nasa-experiments-with-knowledge-discovery/

### Inspection tools
Giant Language model Test Room  
http://gltr.io/dist/index.html  

### Generation 
Plotto, "algebra" for stories   
https://github.com/garykac/plotto/blob/gh-pages/how-to.md

Doctoral thesis: “Humour-in-the-loop: Improvised Theatre with Interactive Machine Learning Systems” - https://korymathewson.com/phd/
===> references dAIrector, a structured graph plot generation using Plotto and TVTropes
dAIrector: Automatic Story Beat Generation through Knowledge Synthesis - https://arxiv.org/abs/1811.03423
===> Shaping the Narrative Arc: An Information-Theoretic Approach to Collaborative Dialogue - https://arxiv.org/abs/1901.11528
===> https://improbotics.org/ - An Improvised Theatre Experiment

### Graph neural network in NLP, Geometric Deeplearning on graphs
Geometric deep learning   
http://geometricdeeplearning.com/  


### Colorization, Super resolution (images)
Decrappification, DeOldification, and Super Resolution - https://www.fast.ai/2019/05/03/decrappify/
(sample colorization: https://mobile.twitter.com/citnaj/status/1124565367743963136
original: https://www.youtube.com/watch?v=dQRAsKnCmX8 (colorized scene stats around 13:30) )

## Data

### Corpus
A straightfoward library that allows you to crawl, clean up, and deduplicate webpages to create massive monolingual datasets  
https://github.com/chiphuyen/lazynlp

### Language, Dictionary
Sign Language  
https://www.ethnologue.com/subgroups/sign-language

Oxford 3000  
https://www.smartcom.vn/the_oxford_3000.pdf

Simple English  
https://en.wikipedia.org/wiki/Simple_English_Wikipedia

Natural Semantic Metalanguage  
https://en.wikipedia.org/wiki/Natural_semantic_metalanguage


## Blogs, Newsletters  
Cognitionx, AI ADVICE PLATFORM  
https://cognitionx.com/directory/news/  

Towards Data Science, Sharing concepts, ideas, and codes  
https://towardsdatascience.com/

Jay Alammar's blog, Visualizing machine learning one concept at a time  
http://jalammar.github.io/

DEEP LEARNING EXPLAINED   
https://www.lyrn.ai/  

Import AI (weekly)
https://us13.campaign-archive.com/home/?u=67bd06787e84d73db24fb0aa5&id=6c9d98ff2c

NLP News (monthly)
http://newsletter.ruder.io/

## Others
The power of words to shape perception  
https://www.scientificamerican.com/article/our-language-affects-what-we-see/

## State of the art summary

State-of-the-art Multilingual Lemmatization
An analysis of state-of-the-art lemmatizers that work for tens of languages
https://towardsdatascience.com/state-of-the-art-multilingual-lemmatization-f303e8ff1a8

Repository to track the progress in Natural Language Processing (NLP), including the datasets and the current state-of-the-art for the most common NLP tasks. https://nlpprogress.com/
