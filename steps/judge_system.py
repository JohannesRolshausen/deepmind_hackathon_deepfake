from typing import List, Dict
from steps.base import BaseStep
from core.schemas import TaskInput, StepResult
from core.llm import query_llm
import json
import PIL.Image

class JudgeSystem(BaseStep):
    def run(self, input_data: TaskInput) -> StepResult:
        print(f"⚖️ Starting Judge System Debate for: {input_data.image_path}")
        
        try:
            img = PIL.Image.open(input_data.image_path)
        except Exception as e:
            print(f"⚠️ Could not load image at {input_data.image_path}: {e}")
            img = None

        max_rounds = 3
        debate_history: List[Dict[str, str]] = []
        
        final_judgment = None
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n--- Debate Round {round_num} ---")
            
            # 1. Debate Agents
            # Pro-Fake Agent
            pro_fake_prompt = self._create_agent_prompt(
                role="Pro-Fake",
                input_data=input_data,
                history=debate_history,
                round_num=round_num
            )
            fake_argument = query_llm(pro_fake_prompt, images=[img] if img else None)
            print(f"😈 Pro-Fake: {fake_argument[:100]}...")
            
            # Pro-Real Agent
            pro_real_prompt = self._create_agent_prompt(
                role="Pro-Real",
                input_data=input_data,
                history=debate_history,
                round_num=round_num
            )
            real_argument = query_llm(pro_real_prompt, images=[img] if img else None)
            print(f"😇 Pro-Real: {real_argument[:100]}...")
            
            # Update History
            debate_history.append({
                "round": round_num,
                "pro_fake": fake_argument,
                "pro_real": real_argument
            })
            
            # 2. Judge Agent
            judge_prompt = self._create_judge_prompt(
                input_data=input_data,
                history=debate_history,
                round_num=round_num,
                max_rounds=max_rounds
            )
            
            judge_response = query_llm(judge_prompt, images=[img] if img else None)
            
            try:
                # Expecting JSON from Judge
                # Clean up potential markdown code blocks
                cleaned_response = judge_response.replace("```json", "").replace("```", "").strip()
                judge_decision = json.loads(cleaned_response)
                
                decision = judge_decision.get("decision")
                reasoning = judge_decision.get("reasoning")
                print(f"👨‍⚖️ Judge: {decision} - {reasoning[:100]}...")
                
                if decision == "TERMINATE" or round_num == max_rounds:
                    final_judgment = judge_decision
                    break
                    
            except json.JSONDecodeError:
                print(f"⚠️ Judge returned invalid JSON: {judge_response}")
                # Fallback or continue
                if round_num == max_rounds:
                     final_judgment = {"decision": "TERMINATE", "final_verdict": "Inconclusive", "explanation": "Judge failed to return valid JSON."}

        # Format Final Output
        return StepResult(
            source="JudgeSystem",
            content=final_judgment
        )

    def _create_agent_prompt(self, role: str, input_data: TaskInput, history: List[Dict], round_num: int) -> str:
        history_text = ""
        for h in history:
            history_text += f"Round {h['round']}:\nPro-Fake: {h['pro_fake']}\nPro-Real: {h['pro_real']}\n\n"
            
        stance = "You are arguing that the image is AI-GENERATED (Deepfake)." if role == "Pro-Fake" else "You are arguing that the image is REAL (not a deepfake)."
        
        return f"""
        You are a Debate Agent in a forensic analysis system.
        {stance}
        
        Input Image Path: {input_data.image_path}
        User Text: {input_data.text}
        
        Debate History:
        {history_text}
        
        Current Round: {round_num}
        
        Your task:
        1. Analyze the input and the history.
        2. Provide a strong, concise argument supporting your stance.
        3. Refute the opponent's points from previous rounds if applicable.
        
        Return ONLY your argument as plain text.
        """

    def _create_judge_prompt(self, input_data: TaskInput, history: List[Dict], round_num: int, max_rounds: int) -> str:
        history_text = ""
        for h in history:
            history_text += f"Round {h['round']}:\nPro-Fake: {h['pro_fake']}\nPro-Real: {h['pro_real']}\n\n"
            
        return f"""
        You are the Judge Agent supervising a debate about whether an image is a deepfake.
        
        Input Image Path: {input_data.image_path}
        User Text: {input_data.text}
        
        Debate History:
        {history_text}
        
        Current Round: {round_num} / {max_rounds}
        
        Your task:
        1. Evaluate the arguments from both sides.
        2. Decide if the debate has reached sufficient clarity to make a final decision.
        3. If YES or if this is the final round ({max_rounds}), output "TERMINATE" and your final verdict.
        4. If NO and rounds remain, output "CONTINUE".
        
        Return a JSON object with this structure:
        {{
            "decision": "TERMINATE" or "CONTINUE",
            "reasoning": "Brief explanation of why you are terminating or continuing",
            "final_verdict": "Real" or "Fake" or "Inconclusive" (Only required if decision is TERMINATE),
            "explanation": "Final detailed explanation for the user" (Only required if decision is TERMINATE),
            "probability_score": <int 0-100> (Probability of being Fake, Only required if decision is TERMINATE)
        }}
        """
