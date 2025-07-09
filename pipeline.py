import re

import matplotlib.pyplot as plt

from pred_gen import SalesOpportunityAnalyzer


def print_list( mylist):
    [print(f"{val['strategy']}\n\t{val['rationale']}") for val in mylist]

def evaluate_strategy_quality(strategy_text, rationale_text):
    score = 0
    # Specificity: contains numbers or quantitative targets
    if re.search(r"\d", strategy_text + rationale_text):
        score += 1
    # Grounding: mentions benchmarks or words like percentile, average, or SHAP
    if re.search(r"benchmark|percentile|SHAP|average|mean|median", rationale_text, re.IGNORECASE):
        score += 1
    # Actionability: has imperative verbs like "increase", "reduce", "engage"
    if re.search(r"\b(increase|reduce|engage|remove|improve|accelerate|shorten)\b", strategy_text, re.IGNORECASE):
        score += 1
    # Relevance: references opportunity-specific factors
    if re.search(r"emails|response time|demo|discount|competitor", strategy_text + rationale_text, re.IGNORECASE):
        score += 1
    # Clarity: keeps rationale short enough
    if len(rationale_text) < 300:
        score += 1
    return score  # Max possible score: 5

def print_risk(risk_factors):
    print ("Risk Factors")

    headers = ["Feature", "Risk Value", "Record Value", "75th Percentile"]

    # Calculate max width for each column
    col_widths = [max(len(str(row[i])) for row in risk_factors + [headers]) for i in range(4)]

    # Print headers
    header_row = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for row in risk_factors:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(4)))


# ----- Example strategies to evaluate -----
# model, X_train, X_test, y_train, y_test, perc_75_dict, explainer, importances = build_model()
# shap_values = explainer(X_test)
grounded_outputs = []
opportunity_only_outputs = []
analyzer = SalesOpportunityAnalyzer()
print("Training model...")
analyzer.train_model()
for i in range(50):

    result = analyzer.analyze_opportunity()

    for strategy in result['opportunity_only_strategies']:
        opportunity_only_outputs.append(strategy)

    for strategy in result['prediction_strategies']:
        grounded_outputs.append(strategy)
    if i % 10 == 0:
        print("\nAnalyzing random opportunity...")
        print(f"\nOpportunity ID: {result['opportunity_id']}")
        print(f"Win Probability: {result['win_probability']:.2f}")
        print_risk(result['risk_factors'])
        print("\nPrediction-based Strategies:")
        print_list(result['prediction_strategies'])
        print("\nOpportunity_only Strategies:")
        print_list(result['opportunity_only_strategies'])
# ----- Evaluate each -----

grounded_scores = []
for out in grounded_outputs:
    grounded_scores.append(evaluate_strategy_quality(out["strategy"], out["rationale"]))

opportunity_only_scores = []
for out in opportunity_only_outputs:
    opportunity_only_scores.append(evaluate_strategy_quality(out["strategy"], out["rationale"]))

# ----- Calculate averages -----

avg_grounded = sum(grounded_scores) / len(grounded_scores)
avg_opportunity_only = sum(opportunity_only_scores) / len(opportunity_only_scores)

print(f"Average Predictively Enhanced strategy score: {avg_grounded:.2f}/5")
print(f"Average Opportunity Only strategy score: {avg_opportunity_only:.2f}/5")

# ----- Plot -----

labels = ["Predictively Enhanced Prompt", "Opportunity Only Prompt"]
scores = [avg_grounded, avg_opportunity_only]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, scores, color=["green", "red"])
plt.ylim(0, 5)
plt.ylabel("Average Quality Score (0-5)")
plt.title("Predictively Enhanced vs. Opportunity Only Prompt Strategy Quality")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, f"{height:.2f}", ha='center')
plt.tight_layout()
plt.show()
