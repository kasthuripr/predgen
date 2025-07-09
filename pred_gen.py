from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple, Any
import json
import random
import warnings
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from Phi3 import call_llm_with_tools

# Suppress warnings
warnings.filterwarnings("ignore")

class PrintOptions(Enum):
    """Options for formatting the output of strategy recommendations."""
    STRATEGY = auto()
    RATIONALE = auto()
    BOTH = auto()

@dataclass
class OpportunityData:
    """Data class to hold opportunity features and metadata."""
    deal_size_usd: float
    customer_tenure_days: int
    days_in_pipeline: int
    num_decision_makers: int
    num_demo_attended: int
    num_competitors: int
    content_downloads: int
    website_visits: int
    response_time_hrs: float
    discount_applied: bool
    competitor_mentioned: bool
    days_since_last_touch: int
    num_stakeholders: int
    num_emails: int
    num_meetings: int

class SalesOpportunityAnalyzer:
    """Analyzes sales opportunities and provides recovery strategies."""
    
    COLUMN_NAME_MAPPING = {
        "num_emails": "Number of emails",
        "num_meetings": "Number of meetings",
        "discount_applied": "Discount applied",
        "competitor_mentioned": "Competitor mentioned",
        "days_since_last_touch": "Days since last touch",
        "num_stakeholders": "Number of stakeholders",
        "deal_size_usd": "Deal size (USD)",
        "customer_tenure_days": "Customer tenure (days)",
        "num_decision_makers": "Number of decision makers",
        "num_demo_attended": "Number of demos attended",
        "days_in_pipeline": "Days in pipeline",
        "num_competitors": "Number of competitors",
        "content_downloads": "Content downloads",
        "website_visits": "Website visits",
        "response_time_hrs": "Average response time (hrs)",
        "closed_won": "Closed won"
    }
    
    def __init__(self, random_seed: int = 42):
        """Initialize the analyzer with a random seed for reproducibility."""
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.model = None
        self.explainer = None
        self.importances = None
        self.perc_75_dict = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def generate_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic sales opportunity data."""
        data = {
            "num_emails": np.random.poisson(10, n_samples),
            "num_meetings": np.random.poisson(5, n_samples),
            "discount_applied": np.random.binomial(1, 0.3, n_samples),
            "competitor_mentioned": np.random.binomial(1, 0.4, n_samples),
            "days_since_last_touch": np.random.randint(0, 60, n_samples),
            "num_stakeholders": np.random.randint(1, 5, n_samples),
            "deal_size_usd": np.random.lognormal(8, 1.5, n_samples).round(2),
            "customer_tenure_days": np.random.gamma(100, 2, n_samples).astype(int),
            "num_decision_makers": np.random.randint(1, 6, n_samples),
            "num_demo_attended": np.random.poisson(1.5, n_samples),
            "days_in_pipeline": np.random.gamma(30, 1, n_samples).astype(int),
            "num_competitors": np.random.poisson(2, n_samples),
            "content_downloads": np.random.poisson(3, n_samples),
            "website_visits": np.random.poisson(8, n_samples),
            "response_time_hrs": np.random.exponential(12, n_samples).round(1),
        }
        
        df = pd.DataFrame(data)
        
        # Simulate win/loss
        win_factors = (
            (df["num_emails"] > 4).astype(int) * 2 +
            (df["num_meetings"] > 2).astype(int) * 2 +
            (df["discount_applied"] == 0).astype(int) +
            (df["competitor_mentioned"] == 0).astype(int) * 2 +
            (df["days_since_last_touch"] < 20).astype(int) +
            (df["num_stakeholders"] > 2).astype(int) +
            (df["deal_size_usd"] < df["deal_size_usd"].median()).astype(int) +
            (df["customer_tenure_days"] > 30).astype(int) +
            (df["num_demo_attended"] > 0).astype(int) * 2 +
            (df["days_in_pipeline"] < 90).astype(int) +
            (df["num_competitors"] < 3).astype(int) +
            (df["content_downloads"] > 2).astype(int) +
            (df["website_visits"] > 5).astype(int) +
            (df["response_time_hrs"] < 8).astype(int) * 2
        )
        
        df["closed_won"] = (win_factors + np.random.normal(0, 1, n_samples)) >= 14
        return df.astype({"closed_won": int})

    def train_model(self, test_size: float = 0.2) -> None:
        """Train the predictive model."""
        df = self.generate_sample_data()
        
        # Store 75th percentile values
        self.perc_75_dict = df.describe().loc["75%"].to_dict()
        print ("75th percentile values")

        max_key_len = max(len(str(k)) for k in self.perc_75_dict.items())
        print(f"{'Key'.ljust(max_key_len)} | Value")
        print("-" * (max_key_len + 8))
        for k, v in self.perc_75_dict.items():
            print(f"{str(k).ljust(max_key_len)} | {v:.2f}")
        # Split data
        X = df.drop(columns="closed_won")
        y = df["closed_won"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed
        )
        
        # Train model
        self.model = GradientBoostingClassifier(random_state=self.random_seed)
        self.model.fit(self.X_train, self.y_train)
        
        # Initialize SHAP explainer
        self.explainer = shap.Explainer(self.model, self.X_train)
        self.importances = self._calculate_feature_importance()

    def _calculate_feature_importance(self) -> pd.DataFrame:
        """Calculate and return feature importances."""
        return pd.DataFrame({
            "feature": self.X_train.columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

    def analyze_opportunity(self, opportunity_id: int = None) -> Dict[str, Any]:
        """Analyze a specific opportunity and generate recovery strategies."""
        if opportunity_id is None:
            opportunity_id = random.randint(0, len(self.X_test) - 1)
        
        # Get SHAP values for the opportunity
        shap_values = self.explainer(self.X_test.iloc[[opportunity_id]])

        shap_list = shap_values[0].values.tolist()
        opportunity = self.X_test.iloc[opportunity_id]

        risk_factors = [
            (col, shap_list[i], opportunity[col], self.perc_75_dict[col])
            for i, col in enumerate(opportunity.index)
            if shap_list[i] < 0
        ]
        risk_factors.sort(key=lambda x: x[1])

        # Generate both opportunity_only and prediction-based strategies
        opportunity_only_strategies = self._generate_opportunity_only_strategies(opportunity_id)
        pred_strategies = self._generate_prediction_strategies(opportunity_id, risk_factors)
        
        return {
            "opportunity_id": opportunity_id,
            "win_probability": float(self.model.predict_proba(
                self.X_test.iloc[[opportunity_id]])[0][1]
            ),
            "risk_factors": risk_factors,
            "opportunity_only_strategies": opportunity_only_strategies,
            "prediction_strategies": pred_strategies
        }

    def _generate_opportunity_only_strategies(self, opportunity_id: int) -> List[Dict[str, str]]:
        """Generate strategies based on opportunity data only."""
        opportunity = self.X_test.iloc[opportunity_id]
        prompt = self._build_plain_prompt(opportunity)
        return self._call_llm(prompt)

    def _generate_prediction_strategies(self, opportunity_id: int, risk_factors) -> List[Dict[str, str]]:
        """Generate strategies based on prediction and SHAP values."""
        opportunity = self.X_test.iloc[opportunity_id]
        prompt = self._build_prediction_prompt(opportunity, risk_factors)
        return self._call_llm(prompt)

    def _build_plain_prompt(self, opportunity: pd.Series) -> str:
        """Build a prompt for opportunity_only strategy generation."""
        return f"""Role: You are a deal-closing expert.

Opportunity details:
{chr(10).join(f'- {self.COLUMN_NAME_MAPPING.get(col, col)}: {val}' for col, val in opportunity.items())}

Instructions:
Generate up to 3 recovery strategies that could improve the chance of winning this opportunity.
Base your recommendations only on the details above.
Provide a concise rationale for each strategy.

Always generate in the following format:
[{{"strategy": "<strategy>", "rationale": "<rationale>"}}]"""

    def _build_prediction_prompt(self, opportunity: pd.Series, risk_factors) -> str:
        """Build a prompt for prediction-based strategy generation."""
        proba = self.model.predict_proba([opportunity])[0][1]
        

        # Format risk factors
        risk_factors_str = "\n".join(
            f"- {self.COLUMN_NAME_MAPPING.get(col, col)}: {shap:.4f} (Current: {val}, 75th %ile: {p75})"
            for col, shap, val, p75 in risk_factors
        )
        
        # Format opportunity details
        details = "\n".join(
            f"- {self.COLUMN_NAME_MAPPING.get(col, col)}: {val}"
            for col, val in opportunity.items()
        )
        
        return f"""Role: You are an expert sales strategist with deep knowledge of organizational sales patterns.

Situation: This opportunity currently has a {proba:.2f} chance of being won.

Global organizational trends show these top risk factors correlated with losses:
{chr(10).join(f'- {self.COLUMN_NAME_MAPPING[col]}' for col in self.importances["feature"].head(5) if col in self.COLUMN_NAME_MAPPING)}

75th percentile organizational benchmarks:
{chr(10).join(f'- {self.COLUMN_NAME_MAPPING.get(col, col)}: {val}' for col, val in self.perc_75_dict.items())}

Opportunity details:
{details}

Factors currently negatively impacting this opportunity (Risk Factors):
{risk_factors_str}

Instructions:
1. Identify up to 3 clear recovery strategies targeting the above risk factors.
2. Base recommendations on both opportunity details and organizational benchmarks.
3. Exclude metrics already meeting/exceeding benchmarks.
4. Provide a concise, quantified rationale for each strategy.

Always generate in the following format:
[{{"strategy": "<strategy>", "rationale": "<rationale>"}}]"""

    def _call_llm(self, prompt: str) -> List[Dict[str, str]]:
        """Call the language model with the given prompt."""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "What is the best recovery strategy for this opportunity?"},
        ]
        
        try:
            response = call_llm_with_tools(messages)
            content = response["choices"][0]["message"]["content"]
            # Clean up the response
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error processing LLM response: {e}")
            return [{"strategy": "Error generating strategies", "rationale": str(e)}]

def main():
    # Example usage
    analyzer = SalesOpportunityAnalyzer()
    print("Training model...")
    analyzer.train_model()
    
    print("\nAnalyzing random opportunity...")
    result = analyzer.analyze_opportunity()
    
    print(f"\nOpportunity ID: {result['opportunity_id']}")
    print(f"Win Probability: {result['win_probability']:.2f}")
    
    print("\nOpportunity-Only Strategies:")
    for i, strategy in enumerate(result['opportunity_only_strategies'], 1):
        print(f"{i}. {strategy['strategy']}")
        print(f"   Reason: {strategy['rationale']}")
    
    print("\nPrediction-based Strategies:")
    for i, strategy in enumerate(result['prediction_strategies'], 1):
        print(f"{i}. {strategy['strategy']}")
        print(f"   Reason: {strategy['rationale']}")

if __name__ == "__main__":
    main()
