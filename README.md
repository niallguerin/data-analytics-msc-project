## data-analytics-msc-project
MSc - Data Analytics Project from 2019. This project was a prototype and research exercise to assess if I could decouple my original [vendelligence](https://github.com/niallguerin/vendelligence-webapp-personal) information retrieval application from the Google CSE API it was reliant on.

The goal here was to build an information management prototype independent of paid-for APIs for both search and machine learning functionality. Vendelligence was missing best answer prediction support, so you had to still rely on filters to wade through results. This prototype added an ML layer to pinpoint single best answers from a search resultset.

The SO dataset was processed and staged into My SQL using SQL scripts from this [GitHub project](https://github.com/collab-uniba/emse_best-answer-prediction). Refer to Primary Reference literature review papers below for their research paper and citation.

I added additional SQL scripts for information retrieval query comparison tests to check question-answer IDs in the overall SO dataset. This was done to facilitate chunked information extraction and ensure input queries were actually in the CSV datasets before building the corpus in Gensim. It also helped with train:test ML model training as we end up with manually held-out datasets of conrollable size from the entire SO dataset.

Feature analysis was based on the primary paper in the literature review as they used the Boruta package in R to automate feature detection. I performed my own independent test cases of feature reduction after familiarising myself with the dataset which dropped a number of those features and still saw good performance using XGBoost algorithm.

Using file splitter utilities from the My SQL DB data exports to file system, Gensim was used to reconstruct the corpus in [Gensim](https://radimrehurek.com/gensim/) file format. This supported NLP functionality and query similarity searching against input query test cases, using tf-idf in case of this project. Gensim supports multiple other options beyond tf-idf e.g. [Latent Semantic Indexing](https://en.wikipedia.org/wiki/Latent_semantic_analysis).

Two ML algorithms were chosen for the machine learning answer prediction module based on the top-performing list of algorithms identified in the Primary Reference research paper below although I then choose my own preference for implementation: sklearn XGBoost and Keras neural network. The ML module prototype focuses on performance evaluation of both ML algorithms against the Gensim query search resultsets, which act as a filter layer before passing to the best answer prediction layer.

Finally, a simple (and buggy) command-line snippet tool is included that provides the user with the response in weblink format, so answer is opened in their web browser from the Python console.

## Change Log
- Reviewed and re-tested in PyCharm October 2022 (env = Python 3.7) 
- Added bug fixes for deprecated methods in TensorFlow and XGBoost imported library function call sections
- Trimmed README file

## Dataset Downloads
- [Stack Exchange Dataset Download from Internet Archive](https://archive.org/details/stackexchange)
- [Stack Exchange Dataset Blog from Internet Archive](https://stackoverflow.blog/2014/01/23/stack-exchange-cc-data-now-hosted-by-the-internet-archive/?_ga=2.36385504.11393660.1665309316-153577822.1660855372)

## Directory and Code Summary
- 1 is the data staging directory scripts based on Calefato research paper and updates I applied on my local MySQL DB afterwards
- 2 is the feature engineering scripts so that process is decoupled from ML model train:test activity. Features are predefined based on research paper in Primary Reference as they used [Boruta package in R](https://www.rdocumentation.org/packages/Boruta/versions/7.0.0/topics/Boruta) as part of their original SO dataset feature analysis phase to avoid redundant features
- 3 is the Information Retrieval engine using Gensim as a prototype corpus and for tf-idf query similarity inputs for test cases
- 4 is the ML model training scripts for XGBoost ML model and the evaluation reports from scikit-learn
- 5 is the ML model training scripts for Keras TensorFlow Neural Network model and the evaluation reports from scikit-learn
- README.pdf is the instructions for data staging, file splitter utilities for Mac and Windows for CSV file chunking, and workflow summary for Python code script pipelines as they can be run without the IR module or end-to-end with it as this was created back in 2019 with view to modularizing it hence why it is not just a single large code file in a Jupyter Notebook

## Workflow Summary
1. The data from MySQL is chunked using two splitter tools; one for Mac OSX and one for Windows. The files can be configured to various input dataframe sizes via Python/Pandas as required.
2. The feature engineering scripts perform routine NLP tasks like stopword removal and tokenizing sentences and other forms of answer body processing.
3. The workflow can go straight to the ML model train:test stage if desired at this point using the ML models in directories 4 and 5. Note a number of transformations are run at this stage - mandatory - for text to numeric transformations and - optional - the NLP tokenization for text. The neural network would possibly (or not) perform better once additional normalization is performed on the dataset input to the model train:test, but due to size of project at time, only minimal experiments were done there on the dataset input to the neural network model. For additional background on transforming data for ML training, refer to this Google developers [resource](https://developers.google.com/machine-learning/data-prep/transform/introduction).
4. For the full end-to-end workflow, the Information Retrieval engine needs to run which contains a corpus constructed using Gensim from the dataset files. It uses TF-IDF to perform query similarity search space checks against 5 predefined input query test cases. Lastly, a series of output result files from the merge script in that engine directory will produce resultsets for precision@k i.e. 5, 10, 15 record resultsets.
5. The resultsets for precision@k can then be fed into the ML model for best answer prediction from the filtered resultset.
6. A snippet is produced in the CLI with link to the web resource from Stack Overflow using a correlation function between Question ID and Answer ID.

## Notes: 
- The dataset files and pipeline preprocessing stages on the dataset are not uploaded to GitHub as the staging files are GB in size.
- The processed dataset files and even chunked files can range into the hundreds of MB too, so they are not uploaded here either as too large.
- The neural network was resource-intensive and so testing was pushed onto a dedicated Windows workstation with more memory and CPU resources versus my MacBook Pro which struggled to run it in a timely manner. Minimal time was invested in 2019 as further steps like optimizing the dataset input, so the model performs better, would be needed to improve performance. Batch and Epoch parameters had most impact after Batch Normalization was addded. It was mainly used for artificial neural network prototyping and experimentation and research into such models in an applied setting
- Dataset inputs were tested across a range of sample sizes from the original 67GB export - 1K, 20K, 100K, 1 million, 3 million records. Note the entire dataset in MySQL when I staged it was about 27 million records from a 67GB source input dataset
- My dataset is not the same as one [here](https://github.com/collab-uniba/emse_best-answer-prediction) as I took my snapshot over 2 years later from Stack Overflow public dataset downloads. You should read their paper and do your own walkthrough in R to understand the approach they used for evaluation

## Comment in 2022
Today there are other ML models (pre-trained from e.g. [HuggingFace](https://huggingface.co/tasks/question-answering)) dedicated solely to performing use case support on the SO dataset e.g. [HuggingFace SO dataset sample remarks](https://huggingface.co/datasets/so_stacksample). There are other Question Answering pre-trained models from likes of [OpenAI for Question Answering](https://beta.openai.com/docs/guides/answers), which can also be used for standalone question answering module, but keep in mind their [pricing](https://openai.com/api/pricing/) if you want to analyze larger data volumes.

Based on lessons learned from this project, in particular the hardware performance bottlenecks and ML model under-performance encountered with the Keras neural network after limited dataset preprocessing was done for that ML model, my focus has shifted to dataset statistical analysis tools and libraries, feature analysis, and open source projects and platforms that support that task.

### Primary References
F. Calefato, F. Lanubile, and N. Novielli (2018) [???An Empirical Assessment of Best-Answer Prediction Models in Technical Q&A Sites.](https://collab.di.uniba.it/fabio/wp-content/uploads/sites/5/2018/07/EMSE-D-17-00159_R3.compressed.pdf)??? Empirical Software Engineering Journal, DOI: 10.1007/s10664-018-9642-5

The Calefato (2018) paper does empirical evaluation of comprehensive list of ML algorithms and suggests using that knowledge to help build solutions for CQA platforms. They are not aiming to show an integrated workflow, rather a very useful methodology for how one could use their findings to build an integrated solution as they highlight in their own conclusions, so this is my explicit goal here which attempts to leverage the insights from their findings as they have already identified optimal ML algorithms (keeping in mind it was 2017)

Greco C., Haden T., & Damevski K., (2018), [StackInTheFlow](https://github.com/vcu-swim-lab/stack-intheflow): Behavior-Driven Recommendation System for Stack Overflow Posts. 2018 ACM/IEEE 40th International Conference on Software Engineering: Companion Proceedings

The research paper for the SITF tool created by Damevski and his team shows a context-sensitive personalized recommender system. While it uses only Stack Overflow and integrates with the Stack Overflow API, it shows a finished working-in-the-field tool, integrated into a context-driven environment (for code as they were focusing on IDE and developer assistant tasks).

My own GitHub project called [Vendelligence](https://github.com/niallguerin/vendelligence-webapp-personal)

The thesis project aimed to prototype a replacement for vendelligence and the information retrieval layer in that system relying on the paid-for Google CSE API which was a third-party cloud service. In addition, it tried to introduce a Question-Answer automation layer evaluating two ML model algorithms.

### Additional References
Aggarwal, C., (2016), Recommender Systems (eBook)

Bailey et al., Advances in Knowledge Discovery and Data Mining (2016), 20th Pacific-Asia Conference, PAKDD 2016 Auckland, New Zealand, April 19???22, 2016 Proceedings, Part I

Calefato, F., Lanubile, F., & Novielli, N., (2018), How to ask for technical help? Evidence-based guidelines for writing questions on Stack Overflow

Chen, T. & Guestrin, C., (2016), XGBoost: A Scalable Tree Boosting System

Cleverdon C. & Keen, M. (1966), Factors Determining the Performance of Indexing Sysems, ASLIB Cranfield Research Project

Convertino, G. et al., (2017), Toward a mixed-initiative QA system: from studying predictors in Stack Exchange to building a mixed-initiative tool

Friedman, J., (1999), Greedy Function Approximation: A Gradient Boosting Machine

Huang, C. et al. (2018), Software expert discovery via knowledge domain embeddings in a collaborative network

Lai, T. M., Trung, B., & Li, S., (2018), A Review on Deep Learning Techniques Applied to Answer Selection

Larsen et al., Flexible Query Answering Systems (2013), 10th International Conference, FQAS 2013 Granada, Spain, September 2013 Proceedings

Manning, C., Raghaven, P. & Sh??tze, H. (2009), An Introduction to Information Retrieval, Online Edition

Nakov, P., et al. (2017), SemEval-2017 Task 3: Community Question Answering

Nikzad-Khasmakhi, N., Balafar, M.A., & Reza Feizi-Derakhshi, M. (2019), The state-of-the-art in expert recommendation systems

Salton, G., Fox, E. A. & Wu, H., (1983), Extended Boolean Information Retrieval

San Pedro, J. & Karatzoglou, A., (2014), Question recommendation for collaborative question answering systems with RankSLDA

Ponzanelli, L., et al. (2017), Supporting Software Developers with a Holistic Recommender System

Ponzanelli, L., Holistic Recommender Systems for Software Engineering (2014), Companion Proceedings of the 36th International Conference on Software Engineering

Ponzanelli, L., Bacchelli, A., & Lanza, M., (2013), Seahawk: Stack Overflow in the IDE

Procaci, T. B. et al. (2019), Experts and likely to be closed discussions in question and answer communities: An analytical overview

Russell, S. & Norvig, P., (2010), Artificial Intelligence: A Modern Approach, 3rd Edition

Sakata, W. & Shibata, T., (2019) FAQ Retrieval using Query-Question Similarity and BERT-Based Query-Answer Relevance

Shah, C. & Pomerantz., J. (2010), Evaluating and Predicting Answer Quality in Community QA

Karan, M. & Snajder, J., (2018), Paraphrase-focused learning to rank for domain-specific frequently asked questions retrieval

Treude, C., Barzilay, O., & Storey, M. A., (2011), How Do Programmers Ask and Answer Questions on the Web?

Wang., S., Tse-Hsun, C. & Hassan, A E., (2018), Understanding the factors for fast answers in technical Q&A websites

Witten, I. H. & Frank, E., (2011), Data Mining: Practical Machine Learning Tools and Techniques, 3rd Edition

Zeng, W., et al. (2014), Uncovering the information core in recommender systems
