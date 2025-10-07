#!/usr/bin/env python3
"""
Example experiment script for LLM Context Effects.
This script demonstrates how to run a simple similarity judgment experiment.
"""

import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel, Field
import json


class SimilarityScore(BaseModel):
    """Pydantic model for structured output."""
    score: int = Field(description="The similarity score between 0 and 20")


def initialize_model(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """Initialize a language model for the experiment."""
    if model_name.startswith("gpt"):
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=20
        ).with_structured_output(SimilarityScore)
    else:
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            format="json"
        )


def create_similarity_prompt(country1: str, country2: str) -> str:
    """Create a similarity judgment prompt for a country pair."""
    return f"""On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, 
how similar are {country1} and {country2}?

Please rate overall similarity and base your judgement on the following factors:
1. Economy
2. Politics and Governance  
3. Culture, Religion and Ethnicity
4. History
5. Geography and Borders
6. International Relations and Influence
7. Defense and Military Conflict

Return your response as JSON with a 'score' field."""


def run_similarity_experiment(country_pairs: list, model_name: str = "gpt-4o-mini", 
                            temperature: float = 0.0) -> pd.DataFrame:
    """Run a similarity judgment experiment with order effects."""
    
    model = initialize_model(model_name, temperature)
    results = []
    
    print(f"Running experiment with {model_name} (temperature={temperature})")
    print(f"Testing {len(country_pairs)} country pairs...")
    
    for country1, country2 in country_pairs:
        print(f"\nTesting: {country1} vs {country2}")
        
        # Test both orders
        order1_prompt = create_similarity_prompt(country1, country2)
        order2_prompt = create_similarity_prompt(country2, country1)
        
        try:
            # Get similarity scores for both orders
            response1 = model.invoke(order1_prompt)
            response2 = model.invoke(order2_prompt)
            
            # Extract scores
            if hasattr(response1, 'score'):
                score1 = response1.score
            else:
                score1 = response1.get('score', 0)
                
            if hasattr(response2, 'score'):
                score2 = response2.score
            else:
                score2 = response2.get('score', 0)
            
            # Calculate order difference
            order_diff = score1 - score2
            
            result = {
                'country_pair': f"{country1}-{country2}",
                'model_name': model_name,
                'temperature': temperature,
                'score_order1': score1,
                'score_order2': score2,
                'order_difference': order_diff,
                'has_order_effect': abs(order_diff) > 0
            }
            
            results.append(result)
            
            print(f"  Order 1 ({country1}-{country2}): {score1}")
            print(f"  Order 2 ({country2}-{country1}): {score2}")
            print(f"  Difference: {order_diff} {'(Order Effect!)' if abs(order_diff) > 0 else '(No Order Effect)'}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame) -> None:
    """Analyze and display experiment results."""
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    # Basic statistics
    total_pairs = len(df)
    pairs_with_order_effects = df['has_order_effect'].sum()
    order_effect_rate = pairs_with_order_effects / total_pairs * 100
    
    print(f"Total country pairs tested: {total_pairs}")
    print(f"Pairs with order effects: {pairs_with_order_effects}")
    print(f"Order effect rate: {order_effect_rate:.1f}%")
    
    # Order difference statistics
    order_diffs = df['order_difference'].dropna()
    if len(order_diffs) > 0:
        print(f"\nOrder difference statistics:")
        print(f"  Mean: {order_diffs.mean():.2f}")
        print(f"  Std: {order_diffs.std():.2f}")
        print(f"  Min: {order_diffs.min()}")
        print(f"  Max: {order_diffs.max()}")
    
    # Show pairs with largest order effects
    print(f"\nTop 5 pairs with largest order effects:")
    top_effects = df.nlargest(5, 'order_difference')[['country_pair', 'order_difference']]
    for _, row in top_effects.iterrows():
        print(f"  {row['country_pair']}: {row['order_difference']:+d}")


def main():
    """Main function to run the example experiment."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or create a .env file with your API key")
    
    # Example country pairs (subset from the full study)
    example_pairs = [
        ("U.S.A.", "Mexico"),
        ("U.S.A.", "Canada"), 
        ("China", "North Korea"),
        ("Germany", "Austria"),
        ("England", "Ireland"),
        ("U.S.S.R.", "Poland"),
        ("France", "Algeria"),
        ("India", "Sri Lanka")
    ]
    
    print("🧪 LLM Context Effects - Example Experiment")
    print("="*60)
    print("This experiment tests for order effects in similarity judgments.")
    print("We'll test the same country pairs in both orders to see if")
    print("the LLM gives different similarity scores.\n")
    
    # Run experiment
    results_df = run_similarity_experiment(
        country_pairs=example_pairs,
        model_name="gpt-4o-mini",  # Change to your preferred model
        temperature=0.0
    )
    
    # Analyze results
    analyze_results(results_df)
    
    # Save results
    output_file = "example_experiment_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n📚 For more detailed experiments, see:")
    print("   - similarity_effect_single_prompt.py")
    print("   - similarity_effect_cot_prompt.ipynb")
    print("   - analyse_data.ipynb")


if __name__ == "__main__":
    main()
