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
* Dataset-agnostic approach: While trained on specific hate speech datasets, the method generalizes across classifiers.

## Jargons
* Model-agnostic refers to techniques, methods, or tools that can be applied across various types of machine learning models, regardless of the specific architecture or algorithm used.

## Suggestion: 
* Using mulitple representation of text 
* Explore fairness towards non identity-based groups

## Resources: 
1. https://cgarbin.github.io/machine-learning-interpretability-feature-attribution/
2. https://medium.com/geekculture/feature-attribution-in-explainable-ai-626f0a1d95e2
