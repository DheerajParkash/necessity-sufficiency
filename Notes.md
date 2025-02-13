## The prominent use cases for feature attribution are:
* Debug Models: verify that models make predictions for the right reasons.
* Audit Models: verify that models are not looking at attributes that encode bias (gender, race, among others) when making decisions.
* Optimize models: simplify correlated features and remove features that do not contribute to predictions.
* Feature attributions do not have any understanding of the model they are explaining. They simply explain what the model predicts, not caring if the prediction is right or wrong.


## Abstract
* Introduced the Novel Feature attribution methods to explain hate speech detection 
* By providing two complementary socres: necessity and sufficiency.
	- Necessity: How crucial a feature is for the classifier’s decision (i.e., if changed, does the prediction change?).
	- Sufficiency: How sufficient a feature is for a particular decision (i.e., if kept, does the prediction remain unchanged?).
	- The Authors shows both scores expose biases, particularly concerning identity terms.

## Inroduction
* Explainability in AI (XAI) is essential for debugging, ensuring fairness, and making AI decisions understandable.
* Standard feature attribution methods assign an importance score but lack operational clarity.
* Two interpretations of importance:
	- If changing a feature changes the prediction → Necessity
	- If keeping a feature guarantees the same prediction → Sufficiency
* Example from hate speech detection: "I hate women"
	- The word "women" should have low sufficiency (mentioning an identity shouldn't trigger a hate label).
	- The word "women" should have high necessity (hate speech requires a target).
* Model-agnostic feature attribution methods often perturb the input to be explained.
	- it is common to either drop the chosen tokens, or replace them with the mask token
	- Deleting tokens: Removing tokens can disrupt fluency, causing generated text to move outside the distribution.
	- Masking tokens: Replacing tokens with a mask token helps maintain fluency but introduces two challenges:
		* Loss of model-agnosticism
		* Distribution shift: consistent with pre-trained and incosistent with distribution learned during fine-tuning.
	- the paper proposes using a generative model for perturbations to maintain fluency. so the perturbed instances are close to the true data manifold

* Supervised discriminative models depend on dataset correlations, but different models may emphasize different correlations due to architecture, biases, and training methods.
* To distinguish between correlations and direct causes of predictions, **the concept of interventions from causal inference is used**.
* Previous work on causal definitions for XAI has focused on tabular data, but in NLP, the complexity arises from features being words in context, where perturbations are generated to ensure fluency while minimizing their direct influence on the prediction.
* Contemporary hate speech classifiers often learn spurious correlations, such as between identity terms and the hate class, leading to increased discrimination of marginalized groups.
* Necessity and sufficiency metrics for identity terms in hateful sentences reveal classifiers' tendency to over-rely on identity terms or neglect the target's characteristics, such as protected groups vs. non-human entities.
* The contributions of this work are as follows
	- Introduced a methodology for calculating necessity and sufficiency metrics for text data, offering deeper insights than traditional single-metric attribution.
	- Used a generative model to produce input perturbations, addressing out-of-distribution issues seen with token deletion and masking.
	- Applied this methodology to hate speech classification, demonstrating its ability to detect and explain biases in classifiers.

## Background and Related Work
* Types of explanations in AI:
	- Local (specific to one instance) vs. Global (explaining overall model behavior).
	- Post-hoc (after model prediction) vs. Self-explaining (built into the model).
	- The proposed method is local & post-hoc.
* Existing XAI techniques for NLP:
	- Feature Attribution (SHAP, LIME) assigns importance scores but doesn’t separate necessity vs. sufficiency.
	- Counterfactual Explanations (generate similar inputs with different labels).
	- **Combination of feature attribution and counterfactual generation used in this work enables computing local feature importance and providing counterfactual justifications.**
	- Causal Explanation Methods (probabilistic causality to measure necessity and sufficiency).
* Challenges in NLP explanations:
	- In tabular data, feature changes are well-defined; in NLP, token changes must maintain fluency.
	- Standard **masking-based methods (e.g., BERT)** are **not truly model-agnostic** and can bias explanations.*

## Proposed Method
* ***Key Idea**: Apply **causal interventions** to estimate **necessity and sufficiency** for a token in text classification.
	- **Casual interpretations:** The intuition is that if a random variable causes another, intervening on the first will affect the second, while a correlation without causation won't be influenced by intervention.
* Necessity:
	- A feature 𝑥𝑖=𝑎 is necessary for the model's prediction 𝑦 if, in similar inputs (from the neighborhood of 𝑥), changing the feature 𝑥𝑖 from 𝑎 to another value  a0 causes the prediction to change from  𝑦 to a different outcome 𝑦0
	- In simple terms: If changing the feature causes a change in the prediction, the feature is necessary for that prediction.
* Sufficiency:
	- A feature 𝑥𝑖=𝑎 is sufficient for the prediction 𝑦 if, in similar inputs, changing the feature 𝑥𝑖 to 𝑎 flips the model's prediction to 𝑦, regardless of the other features.
    - In simple terms: If changing the feature to 𝑎 directly leads to the prediction 𝑦, the feature is sufficient to cause that prediction.
* NLP challenges in interventions: Unlike tabular data, changing a token affects the full text, making interventions complex.
* Balanced replacements: Words are replaced based on context and task-specific constraints to ensure plausible but meaningful changes.
* Distinguishing causal effects: By carefully replacing correlated words, we can identify which feature the model truly relies on for predictions.	
* Efficient estimation: Instead of modifying one token at a time, we perturb groups of tokens to reduce computation costs.
* Necessity estimation: Measures prediction changes when the target token is present in perturbed subsets, with higher weight given to minimal modifications.
* Sufficiency estimation: Measures prediction changes when the target token is excluded, ensuring meaningful causal attribution.

## Explaining Hate Speech Models
* Focus on positive class explanations: We only explain hate speech predictions, not neutral ones, as explanations for "non-hate" are not meaningful.
* Use of infilling language model (ILM): Perturbations are generated from non-hateful data to avoid misleading correlations.
	- Training the perturbation model only  on the neutral examples allows us to distinguish direct causes of the model prediction from correlations in data.
	- Using the non-hateful data distribution to train the infilling model helps avoid such cases, and enables the method to attribute importance to a token only when the classifier relies on it directly.
	- Datasets: Twitter (Founta et al., 2018), Reddit(Vidgen et al., 2021), Wikipedia comments(Wulczyn et al., 2017) and news article comments(Borkan et al., 2019).
* Dataset-agnostic approach: While trained on specific hate speech datasets, the method generalizes across classifiers.

## Experiments
* Focus: Investigating the necessity and sufficiency of identity mentions for hate speech detection.
* Bias: Hate speech classifiers may falsely identify hate speech based solely on mentions of identity terms.
* Tool Used: HateCheck suite to test hate speech classifiers, specifically targeting women and Muslims.
* Datasets: BERT classifiers trained on three datasets related to hate speech and abusive language.
	- **Twitter (Founta et al., 2018)**
	- **Reddit(Vidgen et al., 2021)**
	- **(Davidson et al.,2017),**
* Hypothesis: Examining the effect of distinguishing 'hate speech' vs. 'abusive language' on necessity and sufficiency of identity terms.
	- if **sufficiency is high** then the model will have a **high error rate** on the test cases that capture **non-hate group identity mentions**.
	- if **necessity is low** then the model will have a **high error rate** on the test cases that cover **abuse against non-protected targets**.
* Hypothesis 1:
	- Expect lower necessity for identity terms in models trained on "abusive" labels compared to those trained on "hate" labels.
* Hypothesis 2:
	- High sufficiency of identity terms in explicit hate speech could indicate over-sensitivity to identity terms.
	- Models with this sensitivity are expected to perform poorly on neutral or positive sentences mentioning identity terms.
* Hypothesis 3:
	- Low necessity for identity terms in hate speech suggests under-reliance on identity and over-reliance on context.
	- This could lead to poor model performance on tasks identifying abusive language not targeting specific identities.
### Implementation
* Necessity and Sufficiency Calculation:
	- Identity tokens (e.g., "women", "Muslims") are masked in test cases to evaluate necessity and sufficiency.
	- The language model generates infillings (replacement for masked tokens) to calculate the importance of identity terms.
	- **100 perturbations** per token are used to ensure a consistent number of perturbations for each test case.
	- **Necessity and sufficiency scores** are calculated only for correctly classified test cases (i.e., positive predictions).

* Test Cases:
	- Results focus on explicitly hateful test cases targeting women and Muslims.
	- Additional test cases include neutral/positive mentions of identity terms (F18 and F19) and abusive language not targeting protected identities (F22, F23, F24).
* Baselines for Comparison:
	- SHAP and LIME are used to calculate token importance for comparison.
	- Default parameters are used for these methods, focusing on correctly predicted test cases.
* Table References:
	- Table 2: Shows the necessity and sufficiency scores for the models.
	- Table 3: Provides the classification results for test cases involving identity terms and abusive language.
	- Appendix C: Details results for necessity and sufficiency calculated using masking (instead of perturbing tokens).


## Results and Discussion 
* Necessity and Sufficiency Attribution:
	- Token “Muslims” is more sufficient than “women” for hate speech classification.
	- Token “disgust” is more necessary for “women” than “Muslims”.
* Hypothesis 1 - Necessity:
	- Models trained on "abuse" labels show lower necessity for identity terms compared to those trained on "hate" labels.
	- This supports that identity terms are necessary for identifying hate speech but not necessarily for abusive language.
* Hypothesis 2 - Sufficiency:
	- High sufficiency for identity terms (e.g., "Muslims") indicates over-sensitivity to identity terms.
	- Models with higher sufficiency show higher error rates for neutral or positive mentions of identity (F18, F19).
* Hypothesis 3 - Necessity vs. Abusive Context:
	- Low necessity for identity terms correlates with high false positives for abuse cases not targeting protected identities.
	- The Founta2018-abuse model, showing the lowest necessity for identity terms, has the highest false positives for non-targeted abuse.
* Model Performance:
	- Vidgen2021-hate shows the highest necessity for identity terms and the lowest error rates on abuse test cases (F22, F23, F24).
	- Davidson2017-abuse exhibits lower sufficiency for identity terms and lower error rates for "Muslims" in neutral or positive contexts.

### Comparison of Average SHAP and LIME Values with Necessity and Sufficiency
* Comparison of Attribution Methods:
	- SHAP and LIME provide average importance values for the identity terms ("Muslims" and "women") in hate speech detection.
	- Founta2018-abuse shows high sufficiency for "Muslims," while Davidson2017-abuse shows high necessity for the same target. This difference reflects the models' varying biases, which SHAP cannot capture effectively.

* Biases in False Positive Rates:
	- Founta2018-abuse and Davidson2017-abuse have different false-positive rates for non-abusive mentions of "Muslims," which the necessity and sufficiency scores capture, but SHAP misses.

* LIME's Limitations:
	- LIME is less consistent than SHAP in capturing model biases. For example, Davidson2017-hate has a high false-positive rate for neutral mentions of Muslims, but LIME gives a low negative score, failing to detect this bias.
	- For "women," LIME assigns similar average importance to target tokens for Founta2018-abuse and Vidgen2021-hate, whereas other metrics (including necessity and sufficiency) highlight significant differences in model behavior.

* Effectiveness of Necessity and Sufficiency:
	- The necessity and sufficiency scores are more effective in distinguishing biases like over-sensitivity (high sufficiency) and under-sensitivity (low necessity) to identity terms than SHAP and LIME.
	- Necessity and sufficiency provide more meaningful explanations for false positives and model biases compared to SHAP and LIME.

### Qualitative analysis
* Perturbation Generation Success:
	- Majority of generated perturbations successfully replace the original offensive text with non-offensive alternatives.
	- There is minimal repetition of the same infill, indicating good diversity.
* Quality of Perturbations:
	- The quality of perturbations is mixed and tends to decrease as the number of infills increases.
	- The method prioritizes diversity over fluency, with improvements on fluency planned for future work.
* Combination of HateCheck Functionalities:
	- Generated perturbations often combine multiple HateCheck functionalities (e.g., shuffling word order or changing phrasing).
	- Example: “I hate [IDENTITY]” might generate “This is a subreddit I hate,” combining word order changes and target modification.
* Broader and Non-conventional Replacements:
	- Many infills do not clearly fit into identity terms, individuals, or non-protected groups.
	- Example: Replacements like “I hate crying” or “I hate surprises” illustrate diverse counterfactual generation that covers a broader range of possible contexts.
* Complementary to Manual Checks:
	- The non-constrained, diverse generation of counterfactuals complements manual checks, providing better coverage of potential replacements to test the model's ability to detect hate speech in various contexts.

## Conclusion
* Contribution of Necessity and Sufficiency:
	- Introduced necessity and sufficiency as new, more informative metrics for explaining text classifiers.
	- These metrics provide clearer explanations of token importance than traditional methods.
	- The process of calculating necessity and sufficiency is intuitive and can be easily explained to non-experts.
* Improved Transparency and Understandability:
	- Perturbation-based explanations: The impact of input changes on the classifier output is clearly communicated.
	- These perturbations can be shown to users, enhancing the transparency of the explainability framework.
* Application to Hate Speech Detection:
	- The metrics were applied to explain differences between classifiers for identity-based hate speech and general abuse detection.
	- Helped explain observed over-sensitivity and under-sensitivity to mentions of protected identity groups, addressing fairness concerns in hate speech detection.
* Future Work Directions:
	- Plans to explore the effectiveness of necessity and sufficiency in other applications and languages.
	- Investigate how these metrics can be used for debugging models and for more effective communication of the model’s decision-making process to end-users.

## Jargons
* Model-agnostic refers to techniques, methods, or tools that can be applied across various types of machine learning models, regardless of the specific architecture or algorithm used.

## Suggestion: 
* Using mulitple representation of text 
* Explore fairness towards non identity-based groups
* other target_ident and functionality in hatecheck and 

## New Questions to be Explored:
1. How do necessity and sufficiency metrics perform in other NLP tasks, such as sentiment analysis, fake news detection, or toxicity detection?
	* Apply the same methodology (generating perturbations and calculating necessity/sufficiency) to a new task.
	* For example, in sentiment analysis, you could test how necessary/sufficient certain words (e.g., "love", "hate") are for positive/negative predictions.

1.1 How necessary/sufficient are certain words or phrases (e.g., "government", "hoax") for fake news detection?
	* Use a fake news dataset (e.g., LIAR or FakeNewsNet) and train a classifier.
	* Apply the perturbation method to calculate necessity/sufficiency scores for key terms.
	* Analyze whether the metrics reveal biases (e.g., over-reliance on certain phrases).

1.2 How necessary/sufficient are identity terms (e.g., racial slurs) for toxicity detection?
	* Use a toxicity detection dataset (e.g., Jigsaw’s Toxic Comment Classification Challenge) and train a classifier.
	* Apply the perturbation method to calculate necessity/sufficiency scores for identity terms.
	* Compare the results with hate speech detection to see if the metrics behave similarly.

1.3 How necessary/sufficient are sentiment-bearing words (e.g., "happy", "sad") for sentiment analysis predictions?
	* Use a sentiment analysis dataset (e.g., IMDB or SST) and train a classifier.
	* Apply the perturbation method to calculate necessity/sufficiency scores for sentiment words.
	* Analyze whether the metrics reveal over-reliance on specific words.
	
2. Apply the necessity/sufficiency framework to a new hate speech dataset (e.g., from a different platform like Facebook or Instagram).
	* Dataset Selection: Choose a new dataset (e.g., Hatebase, Gab Hate Corpus, or Reddit Hate Speech Dataset).
	* Train a Classifier: Fine-tune a BERT model on the new dataset.
	* Generate Perturbations: Use the fine-tuned GPT-2 model (or another generative model) to generate perturbations for the new dataset.
	* Calculate Necessity/Sufficiency: Apply the method to calculate necessity and sufficiency scores for identity terms.
	* Compare Results: Compare the necessity/sufficiency scores with those from the original paper to see if the findings generalize.

3. Can necessity and sufficiency metrics generalize across languages? For example, do identity terms in non-English languages (e.g., Spanish, Hindi) show similar necessity/sufficiency patterns as in English?
	* Use a multilingual dataset (e.g., Multilingual Hate Speech Dataset) and apply the perturbation method to generate explanations for classifiers trained on non-English data.
	* Compare the necessity and sufficiency scores across languages to see if the metrics are consistent.
4. Can more advanced generative models (e.g., GPT-3, T5, or BART) improve the quality of perturbations and the accuracy of necessity/sufficiency scores?
	* Replace GPT-2 with a more advanced model (e.g., GPT-3 or T5) for generating perturbations.
	* Compare the fluency and diversity of the generated perturbations with those from GPT-2.
	* Evaluate whether the improved perturbations lead to more accurate necessity/sufficiency scores.

---> work to do	
	* checking hatecheck for task specific test cases for questions 1.1,1.2,1.3 and test cases avaiable for hate and abuse in mulitple lanugaes for Quesiton 3
	* exploring the data sets , targets/anotations , thems and dimensions. For Questions 1-task specific and  2 and 3 in hate and abuse them

## Resources: 
1. https://cgarbin.github.io/machine-learning-interpretability-feature-attribution/
2. https://medium.com/geekculture/feature-attribution-in-explainable-ai-626f0a1d95e2
3. HateCheck Code: https://github.com/paul-rottger/hatecheck-experiments also founta data available there
4. HateCheck Data: https://github.com/paul-rottger/hatecheck-data


## check on 
* Our method can be used with any generator that can model the data distribution conditioned on the label.