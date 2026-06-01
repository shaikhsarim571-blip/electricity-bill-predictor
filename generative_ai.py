import os
from typing import Optional
import json

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import openai
except ImportError:
    openai = None


class AIOptimizer:
    def __init__(self, config, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.provider = config.LLM_PROVIDER
        self.client = None
        try:
            self.client = self._initialize_client()
        except Exception:
            self.client = None
    
    def _initialize_client(self):
        """Initialize LLM client"""
        if not self.api_key:
            return None
        if self.provider == "groq" and Groq:
            return Groq(api_key=self.api_key)
        elif self.provider == "openai" and openai:
            openai.api_key = self.api_key
            return openai
        return None
    
    def generate_optimization_advice(self, user_profile: dict, prediction: dict, bill_info: dict):
        """Generate personalized energy saving recommendations"""
        
        prompt = self._build_prompt(user_profile, prediction, bill_info)
        
        if self.provider == "groq" and self.client:
            return self._call_groq(prompt)
        elif self.provider == "openai" and self.client:
            return self._call_openai(prompt)
        else:
            return self._fallback_recommendation(user_profile, prediction, bill_info)
    
    def _build_prompt(self, user_profile: dict, prediction: dict, bill_info: dict) -> str:
        """Build the prompt for LLM"""
        
        current_consumption = user_profile.get('current_consumption', 0)
        segment = user_profile.get('segment', 'Unknown')
        predicted_consumption = prediction.get('predicted_units', 0)
        current_bill = bill_info.get('current_bill', 0)
        
        prompt = f"""You are an energy efficiency advisor for Pakistan's electricity system.

User Profile:
- Current Monthly Consumption: {current_consumption:.0f} kWh
- Consumer Segment: {segment}
- Predicted Next Month: {predicted_consumption:.0f} kWh
- Current Bill: PKR {current_bill:,.0f}

Tariff Structure (NEPRA):
- 0-100 kWh: Protected Status (Rate: PKR 18.50/unit)
- 101-200 kWh: Normal (Rate: PKR 20.50/unit)
- 201-300 kWh: High (Rate: PKR 24.75/unit)
- 301-500 kWh: Very High (Rate: PKR 28.90/unit)
- 500+ kWh: Industrial (Rate: PKR 32.50/unit)

Provide EXACTLY 5 actionable, specific recommendations in JSON format:
{{
    "recommendations": [
        {{"priority": 1, "action": "...", "potential_savings_pkr": X, "effort": "Easy|Medium|Hard"}},
        {{"priority": 2, "action": "...", "potential_savings_pkr": X, "effort": "Easy|Medium|Hard"}},
        ...
    ],
    "summary": "Brief overall assessment in 2-3 sentences",
    "target_consumption": X,
    "estimated_monthly_savings": Y
}}

Focus on practical, implementable steps for the {segment} segment."""
        
        return prompt
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API"""
        try:
            message = self.client.messages.create(
                model=self.config.GROQ_MODEL,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error calling Groq: {str(e)}"
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"
    
    def _fallback_recommendation(self, user_profile: dict, prediction: dict, bill_info: dict) -> str:
        """Fallback recommendation without LLM"""
        
        segment = user_profile.get('segment', 'Unknown')
        current = user_profile.get('current_consumption', 0)
        predicted = prediction.get('predicted_units', 0)
        
        recommendations = []
        
        if segment == "Low-Income/Protected":
            recommendations.append("Maintain current consumption levels to preserve protected status")
            recommendations.append("Focus on LED lighting to maintain comfort within 100 kWh limit")
        
        elif segment == "Middle-Class/Slab-Breacher":
            target = 150
            if current > target:
                savings = current - target
                recommendations.append(f"Reduce consumption by {savings:.0f} kWh to lower tariff slab")
                recommendations.append("Optimize AC usage during peak hours (5-9 PM)")
        
        elif segment == "Heavy Commercial/AC-Heavy":
            recommendations.append("Install smart thermostats for temperature management")
            recommendations.append("Use off-peak hours (midnight-6 AM) for heavy loads")
            recommendations.append("Consider solar backup for daytime consumption")
        
        recommendations.append("Schedule energy audits quarterly")
        recommendations.append("Monitor consumption trends weekly through smart meters")
        
        return "\n".join([f"• {r}" for r in recommendations])
    
    def extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from LLM response"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        return None
