import os
import json
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from pydantic import BaseModel, ValidationError
from cachetools import TTLCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini API
# Assuming API key is in environment variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "DUMMY_KEY"))

# Configure the model
generation_config = {
  "temperature": 0.1,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config
)

# Caching layer: Cache up to 1000 items for 1 hour (3600 seconds)
# Key will be a hash of the input parameters
explanation_cache = TTLCache(maxsize=1000, ttl=3600)

class AlertResponse(BaseModel):
    human_impact: str
    actionable_advice: str

FEW_SHOT_PROMPT = """
You are an expert logistics and supply chain AI assistant. Your task is to analyze predictive anomaly scores, weather conditions, and supplier data to generate a concise, one-sentence actionable logistics advice and a one-sentence human impact explanation.

Return ONLY a valid JSON object matching this schema:
{"human_impact": "string", "actionable_advice": "string"}

Example 1:
Input:
- ML Score: 0.89 (High Risk), Features: ["rainfall_spike"]
- Weather: 150mm rainfall in last 2 hours
- Supplier: 50 workers, ₹20L perishable cargo

Output:
{"human_impact": "Severe flooding puts 50 workers and ₹20L in perishable cargo at immediate risk.", "actionable_advice": "Reroute shipment via NH44 and hold perishable goods at the inland temperature-controlled warehouse."}

Example 2:
Input:
- ML Score: 0.95 (Critical Risk), Features: ["velocity_plunge", "historical_delay_variance"]
- Weather: Clear, but road collapsed
- Supplier: 200+ workers, ₹50L electronic components

Output:
{"human_impact": "A major road collapse endangers a convoy carrying ₹50L in electronics and disrupts transit for over 200 workers.", "actionable_advice": "Halt convoy at the nearest safe zone and dispatch alternative transport via the eastern corridor."}

Current Input:
- ML Score: {ml_score} ({risk_level}), Features: {features}
- Weather: {weather}
- Supplier: {supplier_data}

Output:
"""

def generate_explanation(
    ml_score: float,
    features: List[str],
    weather_data: str,
    supplier_data: str
) -> Dict[str, str]:
    """
    Generates human-readable impact and actionable advice using Gemini.
    Incorporates caching to prevent redundant API calls.
    """
    risk_level = "Critical Risk" if ml_score >= 0.9 else "High Risk" if ml_score >= 0.7 else "Moderate Risk"
    
    # Create cache key
    cache_key = f"{ml_score}_{','.join(features)}_{weather_data}_{supplier_data}"
    
    if cache_key in explanation_cache:
        logger.info("Cache hit for explanation generation.")
        return explanation_cache[cache_key]

    prompt = FEW_SHOT_PROMPT.format(
        ml_score=ml_score,
        risk_level=risk_level,
        features=json.dumps(features),
        weather=weather_data,
        supplier_data=supplier_data
    )

    try:
        logger.info("Calling Gemini API for explanation generation.")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Prevent JSONDecodeError: Strip markdown backticks if Gemini wrapped the JSON
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse and validate JSON
        parsed_json = json.loads(response_text)
        validated_response = AlertResponse(**parsed_json)
        
        result = validated_response.model_dump()
        
        # Update cache
        explanation_cache[cache_key] = result
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        return {
            "human_impact": "Error generating human impact assessment due to API failure.",
            "actionable_advice": "Please consult the dashboard manually for routing decisions."
        }
    except ValidationError as e:
        logger.error(f"Gemini response did not match expected schema: {e}")
        return {
            "human_impact": "Error validating impact assessment format.",
            "actionable_advice": "Manual review required for this segment."
        }
    except Exception as e:
        logger.error(f"Unexpected error during Gemini API call: {e}")
        return {
            "human_impact": "Service unavailable to assess impact.",
            "actionable_advice": "System fallback: use historical safe routes."
        }

if __name__ == "__main__":
    # Test the function
    res = generate_explanation(
        ml_score=0.92,
        features=["rainfall_spike", "velocity_plunge"],
        weather_data="200mm rainfall in Chennai",
        supplier_data="150 workers, ₹80L medical supplies"
    )
    print("Test Response:", res)
