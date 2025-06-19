import openai
import base64
import requests
import json
import random
import os
import re # For robust JSON parsing
from typing import List, Dict, Any, Tuple, Optional, Union

# --- Configuration ---
# Set OPENAI_API_KEY as an environment variable or pass it to the constructor.

class MapInterpretation:
    def __init__(self, 
                 categorized_qa_pairs: Dict[str, List[Dict[str, str]]], 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-4o"):
        """
        Initializes the MapInterpretation class.

        Args:
            categorized_qa_pairs: A dictionary where keys are category names (str)
                                  and values are lists of QA dicts for that category.
                                  Example: 
                                  {
                                      "Roads": [{"q": "...", "a": "..."}],
                                      "Buildings": [{"q": "...", "a": "..."}]
                                  }
            api_key: Your OpenAI API key.
            model: The OpenAI model to use.
        """
        if not categorized_qa_pairs or not all(isinstance(v, list) and v for v in categorized_qa_pairs.values()):
            raise ValueError("categorized_qa_pairs cannot be empty and each category must have QA pairs.")
        self.categorized_qa_pairs = categorized_qa_pairs
        self.model = model
        
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it as an argument.")
        self.client = openai.OpenAI(api_key=resolved_api_key)

        self.current_conversation_history: List[Dict[str, Any]] = []
        self.current_map_image_base64: Optional[str] = None
        self.map_verified_successfully: bool = False
        
        print(f"MapInterpretation initialized with {len(categorized_qa_pairs)} QA categories, using model {self.model}.")

    def _encode_image_to_base64(self, image_path_or_url: str) -> Optional[str]:
        # (Same as before)
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(image_path_or_url, stream=True)
                response.raise_for_status()
                image_bytes = response.content
            else:
                with open(image_path_or_url, "rb") as image_file:
                    image_bytes = image_file.read()
            return base64.b64encode(image_bytes).decode('utf-8')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image from URL {image_path_or_url}: {e}")
            return None
        except FileNotFoundError:
            print(f"Image file not found at path: {image_path_or_url}")
            return None
        except Exception as e:
            print(f"Error processing image {image_path_or_url}: {e}")
            return None

    def _normalize_answer(self, answer: str) -> str:
        # (Same as before)
        return answer.lower().strip()

    def _call_llm_with_history(self, messages_to_send: List[Dict[str, Union[str, List[Dict[str,str]]]]], temperature: float = 0.2, max_tokens: int = 500) -> str:
        # (Same as before, ensuring messages_to_send type hint is flexible for multimodal content)
        print(f"\n--- Sending to LLM ({self.model}) ---")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send, # type: ignore 
                max_tokens=max_tokens,
                temperature=temperature
            )
            llm_response_content = response.choices[0].message.content
            if llm_response_content is None:
                print("LLM returned None content.")
                return "ERROR_LLM_NO_CONTENT"
            # print(f"LLM Raw Response: {llm_response_content[:500]}{'...' if llm_response_content and len(llm_response_content) > 500 else ''}")
            return llm_response_content
        except Exception as e:
            print(f"Error during LLM API call: {e}")
            return "ERROR_API_CALL"

    def _parse_llm_json_list_response(self, raw_llm_response: str, num_expected_answers: int) -> List[str]:
        """Parses LLM response expected to be a JSON list of strings."""
        llm_answers_list: List[str] = []
        if "ERROR_API_CALL" in raw_llm_response or "ERROR_LLM_NO_CONTENT" in raw_llm_response :
                return ["ERROR_API_RESPONSE"] * num_expected_answers
        try:
            parsed_content = raw_llm_response
            match = re.search(r"```json\s*([\s\S]*?)\s*```", parsed_content)
            if match:
                parsed_content = match.group(1)
            else:
                parsed_content = parsed_content.strip()
            
            parsed_answers = json.loads(parsed_content)
            if isinstance(parsed_answers, list) and all(isinstance(ans, str) for ans in parsed_answers):
                llm_answers_list = parsed_answers
            else:
                print(f"Warning: LLM response was valid JSON but not a list of strings: {parsed_answers}")
                return ["ERROR_INVALID_FORMAT"] * num_expected_answers
        except json.JSONDecodeError:
            print(f"Error: LLM response was not valid JSON after attempting to clean: {raw_llm_response}")
            return ["ERROR_JSON_DECODE"] * num_expected_answers
        return llm_answers_list

    def verify_map_understanding(self, 
                                 image_path_or_url: str, 
                                 questions_per_category: int = 2, 
                                 max_retries_per_category: int = 2) -> bool:
        """
        Verifies LLM's understanding by asking questions from different categories.
        Retries on a per-category basis if answers are incorrect, picking new random questions for that category.
        """
        self.current_conversation_history = []
        self.map_verified_successfully = False
        self.current_map_image_base64 = self._encode_image_to_base64(image_path_or_url)

        if not self.current_map_image_base64:
            print("Failed to load and encode map image.")
            return False

        system_prompt_content = (
            "You are an expert map interpreter. Analyze the provided map image carefully. "
            "You will be asked questions about specific categories related to the map. "
            "For each set of questions, provide your answers *only* as a JSON list of strings, corresponding to the order of the questions."
            "Do not add any other explanatory text before or after the JSON list itself."
        )
        self.current_conversation_history.append({"role": "system", "content": system_prompt_content})
        
        # Initial user message with the image, added once.
        # Subsequent turns will refer to this context.
        initial_user_prompt_text = "Here is an image of a map. I will now ask you questions about it in categories."
        initial_user_message_content_parts = [
            {"type": "text", "text": initial_user_prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.current_map_image_base64}"}}
        ]
        self.current_conversation_history.append({"role": "user", "content": initial_user_message_content_parts})
        # Add a placeholder assistant response to keep alternating roles for subsequent actual Q&A.
        self.current_conversation_history.append({"role": "assistant", "content": "Understood. I have analyzed the map. I am ready for your questions."})


        all_categories_passed = True
        for category_name, qa_list_for_category in self.categorized_qa_pairs.items():
            print(f"\n--- Verifying Category: {category_name} ---")
            if not qa_list_for_category:
                print(f"Warning: No QA pairs for category '{category_name}'. Skipping.")
                continue

            passed_this_category = False
            # Keep track of questions already asked in this category for this verification run
            # to avoid re-picking the same set on retry if possible.
            # This is a simple per-category, per-verification run cache.
            asked_question_indices_in_category: Set[int] = set()


            for attempt in range(max_retries_per_category + 1):
                print(f"  Attempt {attempt + 1}/{max_retries_per_category + 1} for category '{category_name}'")

                # Select k random questions for this category attempt
                # Ensure we don't ask more than available or re-ask if possible
                available_indices = [i for i, _ in enumerate(qa_list_for_category) if i not in asked_question_indices_in_category]
                
                num_to_select = min(questions_per_category, len(available_indices))
                if num_to_select == 0 and len(qa_list_for_category) > 0 : # Exhausted unique questions, allow re-picking
                    print(f"    Note: All unique questions in '{category_name}' asked for this verification run. Re-sampling allowed.")
                    asked_question_indices_in_category.clear() # Reset for this specific case
                    available_indices = list(range(len(qa_list_for_category)))
                    num_to_select = min(questions_per_category, len(available_indices))

                if num_to_select == 0: # Still no questions (e.g., category empty or questions_per_category is 0)
                    print(f"    No questions to ask for '{category_name}' in this attempt. Marking as passed (vacuously true).")
                    passed_this_category = True # Or handle as error if questions_per_category > 0
                    break


                selected_indices = random.sample(available_indices, num_to_select)
                current_questions_being_asked = [qa_list_for_category[i] for i in selected_indices]
                
                # Update asked_question_indices_in_category for the *next* retry attempt for this category
                for idx in selected_indices:
                    asked_question_indices_in_category.add(idx)


                question_texts = [qa['question'] for qa in current_questions_being_asked]
                
                prompt_lines = [
                    f"Now, focusing on the '{category_name}' category for the map provided earlier.",
                    "Please answer the following questions based *only* on the map image.",
                    "Provide your answers as a JSON list of strings, in the same order as the questions.\n",
                    "Questions:"
                ]
                for i, q_text in enumerate(question_texts):
                    prompt_lines.append(f"{i+1}. {q_text}")
                prompt_lines.append("\nYour JSON response:")
                category_user_prompt_text = "\n".join(prompt_lines)

                # Add user message to history for this category's turn
                # The image is already in history from the initial user message.
                # OpenAI models can refer to images in previous turns of a conversation.
                messages_for_this_call = self.current_conversation_history + [
                    {"role": "user", "content": category_user_prompt_text}
                ]
                
                estimated_max_tokens = 50 + (len(current_questions_being_asked) * 30)
                raw_llm_response = self._call_llm_with_history(messages_for_this_call, temperature=0.1, max_tokens=estimated_max_tokens)

                # Update main conversation history
                self.current_conversation_history.append({"role": "user", "content": category_user_prompt_text})
                self.current_conversation_history.append({"role": "assistant", "content": raw_llm_response})

                llm_answers_list = self._parse_llm_json_list_response(raw_llm_response, len(current_questions_being_asked))
                
                # --- Compare answers for this category attempt ---
                incorrect_in_this_attempt = False
                if len(llm_answers_list) != len(current_questions_being_asked):
                    print(f"    Warning: LLM returned {len(llm_answers_list)} answers for category '{category_name}', but {len(current_questions_being_asked)} questions were asked.")
                    incorrect_in_this_attempt = True
                else:
                    for i, qa_pair_expected in enumerate(current_questions_being_asked):
                        llm_ans_norm = self._normalize_answer(llm_answers_list[i])
                        correct_ans_norm = self._normalize_answer(qa_pair_expected['answer'])
                        
                        print(f"    Q: {qa_pair_expected['question']}")
                        print(f"      Expected: '{qa_pair_expected['answer']}' (Normalized: '{correct_ans_norm}')")
                        print(f"      LLM Said: '{llm_answers_list[i]}' (Normalized: '{llm_ans_norm}')")

                        if llm_ans_norm != correct_ans_norm:
                            print("      Result: INCORRECT")
                            incorrect_in_this_attempt = True
                        else:
                            print("      Result: CORRECT")
                
                if not incorrect_in_this_attempt:
                    print(f"  +++ Category '{category_name}' passed this attempt! +++")
                    passed_this_category = True
                    break # Move to the next category
                else:
                    print(f"  --- Incorrect answer(s) in category '{category_name}' for this attempt. ---")
                    if attempt >= max_retries_per_category:
                        print(f"  --- Max retries for category '{category_name}' reached. This category FAILED. ---")
                        all_categories_passed = False # Mark overall failure
                        break # Stop trying this category
                    else:
                        print(f"    Will retry category '{category_name}' with new questions if available.")
            
            if not passed_this_category: # If after all retries, category didn't pass
                all_categories_passed = False # Ensure this is set
                break # Stop verifying other categories if one has definitively failed

        self.map_verified_successfully = all_categories_passed
        if self.map_verified_successfully:
            print("\nSUCCESS: All categories verified successfully!")
        else:
            print("\nFAILURE: Not all categories could be verified successfully.")
        return self.map_verified_successfully


    def analyze_vehicle_interactions(self, vehicle_actions: List[str]) -> Optional[str]:
        """
        Analyzes vehicle interactions on the map, using the established conversation history.
        """
        if not self.map_verified_successfully or not self.current_map_image_base64:
            print("Error: Map understanding has not been successfully verified for the current map, or no map is loaded.")
            print("Please call `verify_map_understanding()` successfully first.")
            return None

        actions_text = "\n".join([f"- {action}" for action in vehicle_actions])
        analysis_prompt_text = (
            "You have previously demonstrated an understanding of the provided map by correctly answering questions about its features across several categories. "
            "Now, considering that same map (which you have already analyzed) and the following vehicle actions, "
            "please analyze potential interactions between vehicles. \n\n"
            "Focus on identifying and describing:\n"
            "1. Conflicts: Situations where vehicles might compete for the same space or right-of-way.\n"
            "2. Potential Collisions: Scenarios where a collision is likely if actions continue unchanged.\n"
            "3. Necessary Yielding Maneuvers: Who should yield to whom based on standard traffic rules or map context.\n"
            "4. Any other safety-critical observations.\n\n"
            "If no significant interactions or safety concerns are found, please state that clearly.\n\n"
            "Vehicle Actions:\n"
            f"{actions_text}\n\n"
            "Your detailed analysis of interactions:"
        )

        # For the analysis, we still send the image again with the prompt for max robustness,
        # even though it's in the history.
        user_message_for_analysis_content_parts = [
            {"type": "text", "text": analysis_prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.current_map_image_base64}"}}
        ]
        
        messages_for_analysis_call = self.current_conversation_history + [
            {"role": "user", "content": user_message_for_analysis_content_parts}
        ]

        print("\n--- Requesting Vehicle Interaction Analysis ---")
        llm_analysis_response = self._call_llm_with_history(messages_for_analysis_call, temperature=0.4, max_tokens=1000)

        if "ERROR_API_CALL" not in llm_analysis_response and "ERROR_LLM_NO_CONTENT" not in llm_analysis_response:
            self.current_conversation_history.append({"role": "user", "content": user_message_for_analysis_content_parts})
            self.current_conversation_history.append({"role": "assistant", "content": llm_analysis_response})
            # print(f"\nLLM Interaction Analysis Output:\n{llm_analysis_response}") # Already printed by _call_llm
            return llm_analysis_response
        else:
            print("Failed to get interaction analysis from LLM.")
            return None

# --- Example Usage ---
if __name__ == "__main__":
    # Categorized QA pairs
    categorized_sample_qa_pairs = {
        "Structures & Roads": [
            {"question": "How many branching roads does the intersection have?", "answer": "4"},
            {"question": "How many lanes does Road 0 have going towards the intersection?", "answer": "2"},
            {"question": "How many lanes does Road 1 have going towards the intersection?", "answer": "1"},
            {"question": "How many lanes does Road 3 have going towards the intersection?", "answer": "2"},
            {"question": "How many lanes does Road 4 have going towards the intersection?", "answer": "1"},
            {"question": "How many lanes does Road 0 have going away from the intersection?", "answer": "1"},
            {"question": "How many lanes does Road 1 have going away from the intersection?", "answer": "1"},
            {"question": "How many lanes does Road 3 have going away from the intersection?", "answer": "1"},
            {"question": "How many lanes does Road 4 have going away from the intersection?", "answer": "1"}
        ],
        "Crossing routes": [
            {"question": "If one vehicle goes from Road 0 to Road 3, another goes from Road 1 to Road 4, will their paths at the intersection cross?", "answer": "Yes"},
            {"question": "If one vehicle goes from Road 4 to Road 3, another goes from Road 1 to Road 4, will their paths at the intersection cross?", "answer": "No"},
            {"question": "If one vehicle goes from Road 0 to Road 3, another goes from Road 4 to Road 0, will their paths at the intersection cross?", "answer": "Yes"},
            {"question": "If one vehicle goes from Road 0 to Road 3, another goes from Road 1 to Road 0, will their paths at the intersection cross?", "answer": "No"},
            {"question": "If one vehicle goes from Road 1 to Road 3, another goes from Road 4 to Road 1, will their paths at the intersection cross?", "answer": "Yes"},
            {"question": "If one vehicle goes from Road 0 to Road 1, another goes from Road 4 to Road 3, will their paths at the intersection cross?", "answer": "No"},
        ]
    }

    try:
        interpreter = MapInterpretation(categorized_qa_pairs=categorized_sample_qa_pairs)
    except ValueError as e:
        print(f"Initialization Error: {e}")
        exit()


    map_image_location = "./data/processed/inD/map/01_bendplatz_graph.jpeg"
    if not os.path.exists(map_image_location):
        print(f"Error: map_image_location '{map_image_location}' does not exist.")
        exit()
    
    # 1. Verify map understanding with categorized questions
    print(f"\nStarting categorized map verification for: {map_image_location}")
    is_understood = interpreter.verify_map_understanding(
        image_path_or_url=map_image_location,
        questions_per_category=3, # Ask 2 questions from each category
        max_retries_per_category=1  # Allow 1 retry for each category (total 2 attempts)
    )

    if is_understood:
        print("\nOVERALL SUCCESS: LLM map understanding verified across all categories.")

        '''
        
        vehicle_actions_on_map_simple = [
            "Vehicle Red (not pictured) is approaching the house from the top of the map along a path parallel to the river on the left side.",
            "Vehicle Yellow (the one visible on the map) starts moving towards the river along the road.",
            "Vehicle Blue (not pictured) enters the map from the right, moving towards the forest edge."
        ]

        print("\n--- Now, analyzing vehicle interactions ---")
        analysis_result = interpreter.analyze_vehicle_interactions(vehicle_actions=vehicle_actions_on_map_simple)

        if analysis_result:
            print("\n--- Vehicle Interaction Analysis Complete (output already shown by LLM call) ---")
            with open("vehicle_interaction_analysis_categorized.txt", "w") as f:
                f.write(analysis_result)
            print("Analysis saved to vehicle_interaction_analysis_categorized.txt")
        else:
            print("\n--- Vehicle Interaction Analysis Failed ---")
            
        # Optional: print full history
        # print("\nFull conversation history:")
        # for i, msg in enumerate(interpreter.current_conversation_history):
        # ... (history printing logic from before)

        '''

    else:
        print("\nOVERALL FAILURE: LLM map understanding could not be verified across all categories.")