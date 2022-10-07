## data-analytics-msc-project
MSc - Data Analytics Project - 2019.

- Reviewed and re-tested in PyCharm October 2022
- Added bug fixes for deprecated methods TensorFlow and XGBoost code sections

## Goal
The aim of the research project and prototype was to build on a static Information Retrieval (IR) system I created for myself to assist me on customer implementation projects in the IT domain back in 2015 and use the tools and techniques from the course and a personal interest in technical Question and Answering systems to provide a feasible workflow for future open source development projects that I can develop for personal use or as open source projects to assist me or colleagues in work settings.

This project aims to harness Information Retrieval (IR) and filtering techniques built from scratch using Python scripts and the Gensim  open source library integrated with a machine learning pipeline evaluating both a Neural Network (NN) model and Extreme Gradient Boost (XGB) ensemble model; the end goal is for the system workflow to provide best answer snippets from my input queries against a technical dataset, in this case Stack Overflow. It will then evaluate whether the method worked and to what level of performance based on IR and ML standard measurement approaches for the workflow process.

The project focuses on integrating the approaches taken in two research papers by a team in Italy and USA into a single integrated system workflow consisting of a sequence of staging steps in a local database followed by a sequential Python script workflow that ultimately generates a prediction for the best answer snippet, which is assessed by me as a user, as to whether it solves my query goal or not.

The project consists of an integrated workflow between the IR system component step and the output of this step becomes the input of the Machine Learning pipeline step. The first part is modelled on the Java-based StackInTheFlow (SITF) application built by Damevski and his team (2018). It is referred to as a personalized recommender system component in the paper. However, throughout my thesis, I generally refer to an IR system component as having inspected the methods and code in the SITF open source code repository on GitHub and initially researching and evaluating recommender system hybrid approaches, I moved away from them. I am not using any of the methods classically associated with a recommender system component based on definitions (Aggarwal, C., 2016).

The second machine learning (ML) modelling and prediction workflow step is modelled extensively on the feature engineering and empirical research methods published by Calefato (2018). That paper provides an open source framework and methodology that lets you plug in Stack Overflow data and generate best answer predictions and evaluates 26 machine learning algorithms across a number of test exercises. While it provides sample inputs, R Caret code and scripts to run them, it did not provide the feature engineering scripts for feature engineering used in their case needed to make the R Caret framework work on new or customized Stack Overflow datasets as their paper is from 2 years ago. I wanted to re-use a tried and trusted empirical method and because it specifically targets the Stack Overflow CQA dataset (and SAP community network , which is important to me as I will be working with SAP® from September) and evaluates such a large list of ML algorithms and their performance in best answer prediction. They also touch on edge cases which are important, but which I do not focus on in this project.

The thesis project is aimed at being a next step extension to an earlier information retrieval and filtering tool project from 2015-2016, Vendelligence . Vendelligence, which simply stood for ‘Vendor Intelligence’, consisted of curated Google Custom Search Engine  XML repositories and a Javascript client integrated with a JSON-driven Google Search API  web service. It allowed free text queries and because it used a custom-curated dataset, it indexed both CQA website data (including Stack Overflow  website) and static data like vendor help documentation for software releases be it commercial or open source (SAP Help Library , Spring Framework Project Release documentation). Vendelligence allowed simple CRUD operations, allowing queries to be clipped, re-used, and responses and annotation notes per query stored in a MySQL database. This all ran inside a Spring Boot  container hosted on Amazon Web Services  cloud platform and allowed multi-user login.

After learning new technologies and techniques on the Msc in Data Analytics course, I decided to revisit my old Vendelligence project, but target the next step which was to generate automated best answer prediction from initial IR and filtering tasks. Vendelligence did a good job of IR and filtering only because it leveraged the Google Search API, so was using Google search functions available over the API, but I had to pay fees once my query rate went over a daily quota to the API (100 query limit per day). I was advised that I would need to create my own IR and Filtering component from scratch if I ever wanted to decouple the application from dependencies on Google or other vendor APIs like Elasticsearch , which I had also prototyped a question answering system on during work on Vendelligence.

I wanted to implement an ML solution or use an empirical research-based framework I could plug into to solve the second problem for users of having to wade through huge result sets even with good Google API and custom filters I added to Google API queries at the time based on topic categories I created to augment the query expansion step per vendor (commercial or open source). There was still too much data coming back to manually filter and check does the system accurately answer the question or not.

This manual filtering and analysing of search query results (web, native in-house knowledge bases, Slack and other messaging channels) was a major issue I had experienced in any knowledge management and research use case using vendor proprietary and open web information systems when working as a consultant. Customers likewise complained about how long it took to dig out the correct and most reliable answer from a large list of answers on a given domain or sub-domain, hence their preference for human consulting support on projects.

The research paper for the SITF tool created by Damevski and his team shows a context-sensitive personalized recommender system. While it uses only Stack Overflow and integrates with the Stack Overflow API, it shows a finished working in-the-field tool, integrated into a context-driven environment (for code as they were focusing on IDE and developer assistant tasks). I wanted a general information retrieval client for technical consultants, an information development environment, so if the thesis shows reasonable question answering prediction accuracy and addresses the research questions, then I can build an application that can expose the information retrieval and filtering layer and ML prediction layer as service endpoints for a remote client using a microservice architecture.

The empirical research by Calefato and his team showed it was possible based on building on their findings to target this problem of best answer prediction as a binary classification task in machine learning on the Stack Overflow (and other CQA) datasets: the question is either solved (true) or not solved (false) from the available answers to that question. There are use cases per the edge cases documented by Calefato. where there is no answer by the community or where there is only one answer, so the information quality of the answer may be in doubt. This is something Calefato addresses in his paper, though I have not focused on such single-answer or no-answer cases in my own hybrid implementation workflow process.

The goal of the thesis is to implement a working pipeline the merges the techniques from both research papers, in particular leveraging the feature engineering and optimal algorithm findings from Calefato. rather than randomly picking an ML algorithm, to build an end-to-end pipeline from query input to retrieval of relevant result set to using the ML model to identify the best answer snippet as the final step.

I am trying to integrate two research paper methodologies, with a simplified information retrieval system in this implementation due to time constraints.

I wanted to use more traditional approaches first on this iteration before experimenting with what are more complex approaches methods like BERT from modern question answering system research field. I can learn from the findings and limitations in this project but as one of the RQs is to look at performance and feasibility of running this as a personal project on a future basis on a laptop versus costly cloud servers for ML training or complex architectures, this is an important for me to keep in mind in terms of choosing implementation strategies for the integrated workflow approach in my own system projects.

As of October 2022, I am now working on a modularized system designed for my own personal use as I am best able to assess whether it works for me as the end user and it provides a test area to apply the lessons learned in various online ML courses and labs.

### Primary References
F. Calefato, F. Lanubile, and N. Novielli (2018) “An Empirical Assessment of Best-Answer Prediction Models in Technical Q&A Sites.” Empirical Software Engineering Journal, DOI: 10.1007/s10664-018-9642-5

The Calefato (2018) paper does empirical evaluation of comprehensive list of ML algorithms and suggests using that knowledge to help build solutions for CQA platforms.
They are not aiming to show an integrated workflow, rather a very useful methodology for how one could use their findings to build an integrated solution as they highlight
in their own conclusions, so this is my explicit goal here which attempts to leverage the insights from their findings

Greco C., Haden T., & Damevski K., (2018), StackInTheFlow: Behavior-Driven Recommendation System for Stack Overflow Posts. 2018 ACM/IEEE 40th International Conference on Software Engineering: Companion Proceedings

The research paper for the SITF tool created by Damevski and his team shows a context-sensitive personalized recommender system. While it uses only Stack Overflow and integrates with the Stack Overflow API,
it shows a finished working-in-the-field tool, integrated into a context-driven environment (for code as they were focusing on IDE and developer assistant tasks).

### Additional References
Aggarwal, C., (2016), Recommender Systems (eBook)

Bailey et al., Advances in Knowledge Discovery and Data Mining (2016), 20th Pacific-Asia Conference, PAKDD 2016 Auckland, New Zealand, April 19–22, 2016 Proceedings, Part I

Calefato, F., Lanubile, F., & Novielli, N., (2018), How to ask for technical help? Evidence-based guidelines for writing questions on Stack Overflow

Chen, T. & Guestrin, C., (2016), XGBoost: A Scalable Tree Boosting System

Cleverdon C. & Keen, M. (1966), Factors Determining the Performance of Indexing Sysems, ASLIB Cranfield Research Project

Convertino, G. et al., (2017), Toward a mixed-initiative QA system: from studying predictors in Stack Exchange to building a mixed-initiative tool

Friedman, J., (1999), Greedy Function Approximation: A Gradient Boosting Machine

Huang, C. et al. (2018), Software expert discovery via knowledge domain embeddings in a collaborative network

Lai, T. M., Trung, B., & Li, S., (2018), A Review on Deep Learning Techniques Applied to Answer Selection

Larsen et al., Flexible Query Answering Systems (2013), 10th International Conference, FQAS 2013 Granada, Spain, September 2013 Proceedings

Manning, C., Raghaven, P. & Shütze, H. (2009), An Introduction to Information Retrieval, Online Edition

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
