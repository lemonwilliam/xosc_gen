import openai
import base64
import requests
import json
import random
import os
import re # For robust JSON parsing
from typing import List, Dict, Any, Tuple, Optional, Union, Set

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
        self.current_map_description: Optional[str] = None # New attribute
        self.map_verified_successfully: bool = False
        
        print(f"MapInterpretation initialized with {len(categorized_qa_pairs)} QA categories, using model {self.model}.")

    def _read_map_description(self, description_path: Optional[str]) -> Optional[str]:
        """Reads map description from a file path."""
        if description_path:
            try:
                with open(description_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                print(f"Warning: Map description file not found at {description_path}. Proceeding without it.")
                return None
            except Exception as e:
                print(f"Warning: Error reading map description file {description_path}: {e}. Proceeding without it.")
                return None
        return None

    def _encode_image_to_base64(self, image_path_or_url: str) -> Optional[str]:
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
        except FileNotFoundError: # Changed to raise IOError for explicit handling in verify_map_understanding
            raise IOError(f"Image file not found at path: {image_path_or_url}")
        except Exception as e:
            print(f"Error processing image {image_path_or_url}: {e}")
            return None

    def _normalize_answer(self, answer: str) -> str:
        return answer.lower().strip()

    def _call_llm_with_history(self, messages_to_send: List[Dict[str, Any]], temperature: float = 0.2, max_tokens: int = 500) -> str:
        # Changed type hint for messages_to_send to List[Dict[str, Any]] for broader compatibility
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
        llm_answers_list: List[str] = []
        if "ERROR_API_CALL" in raw_llm_response or "ERROR_LLM_NO_CONTENT" in raw_llm_response :
                return ["ERROR_API_RESPONSE"] * num_expected_answers
        try:
            parsed_content = raw_llm_response
            match = re.search(r"```json\s*([\s\S]*?)\s*```", parsed_content)
            if match:
                parsed_content = match.group(1)
            else: # If no markdown, try to parse directly, stripping whitespace
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
                                 map_description_path: Optional[str] = None, # New parameter
                                 questions_per_category: int = 2, 
                                 max_retries_per_category: int = 2) -> bool:
        """
        Verifies LLM's understanding using image and optional text description.
        """
        self.current_conversation_history = []
        self.map_verified_successfully = False
        
        try:
            self.current_map_image_base64 = self._encode_image_to_base64(image_path_or_url)
        except IOError as e: # Catch IOError specifically from _encode_image_to_base64
            print(f"Critical error loading image: {e}")
            return False # Cannot proceed without image

        if not self.current_map_image_base64: # General check if encoding failed for other reasons
            print("Failed to load and encode map image.")
            return False

        self.current_map_description = self._read_map_description(map_description_path)

        system_prompt_content = (
            "You are an expert map interpreter. You will be provided with a map image and optionally a textual description of the map. Analyze these inputs carefully. "
            "You will then be asked questions about specific categories related to the map. "
            "For each set of questions, provide your answers *only* as a JSON list of strings, corresponding to the order of the questions."
            "Do not add any other explanatory text before or after the JSON list itself."
        )
        self.current_conversation_history.append({"role": "system", "content": system_prompt_content})
        
        # Initial user message with the image and description
        initial_user_prompt_parts = []
        initial_user_prompt_text_intro = "Here is a map image"
        if self.current_map_description:
            initial_user_prompt_text_intro += " and its textual description. Please analyze both."
            initial_user_prompt_parts.append({"type": "text", "text": initial_user_prompt_text_intro})
            initial_user_prompt_parts.append({"type": "text", "text": f"\nMap Description:\n---\n{self.current_map_description}\n---"})
        else:
            initial_user_prompt_text_intro += ". Red boxed numbers represent Road IDs and black boxed numbers represent lane ids. Please analyze it."
            initial_user_prompt_parts.append({"type": "text", "text": initial_user_prompt_text_intro})

        initial_user_prompt_parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.current_map_image_base64}"}}
        )
        initial_user_prompt_parts.append(
             {"type": "text", "text": "\nI will now ask you questions about it in categories."}
        )

        self.current_conversation_history.append({"role": "user", "content": initial_user_prompt_parts})
        self.current_conversation_history.append({"role": "assistant", "content": "Understood. I have analyzed the provided map information. I am ready for your questions."})

        # --- Category verification loop (largely same as before) ---
        all_categories_passed = True
        for category_name, qa_list_for_category in self.categorized_qa_pairs.items():
            print(f"\n--- Verifying Category: {category_name} ---")
            if not qa_list_for_category:
                print(f"Warning: No QA pairs for category '{category_name}'. Skipping.")
                continue

            passed_this_category = False
            asked_question_indices_in_category: Set[int] = set()


            for attempt in range(max_retries_per_category + 1):
                print(f"  Attempt {attempt + 1}/{max_retries_per_category + 1} for category '{category_name}'")

                available_indices = [i for i, _ in enumerate(qa_list_for_category) if i not in asked_question_indices_in_category]
                
                num_to_select = min(questions_per_category, len(available_indices))
                if num_to_select == 0 and len(qa_list_for_category) > 0 : 
                    print(f"    Note: All unique questions in '{category_name}' asked for this verification run. Re-sampling allowed.")
                    asked_question_indices_in_category.clear() 
                    available_indices = list(range(len(qa_list_for_category)))
                    num_to_select = min(questions_per_category, len(available_indices))

                if num_to_select == 0: 
                    print(f"    No questions to ask for '{category_name}' in this attempt. Marking as passed (vacuously true).")
                    passed_this_category = True 
                    break


                selected_indices = random.sample(available_indices, num_to_select)
                current_questions_being_asked = [qa_list_for_category[i] for i in selected_indices]
                
                for idx in selected_indices:
                    asked_question_indices_in_category.add(idx)


                question_texts = [qa['question'] for qa in current_questions_being_asked]
                
                prompt_lines = [
                    f"Now, focusing on the '{category_name}' category for the map information (image and text description, if provided) that you have already processed.",
                    "Please answer the following questions based *only* on the previously provided map image and its description (if any).",
                    "Provide your answers as a JSON list of strings, in the same order as the questions.\n",
                    "Questions:"
                ]
                for i, q_text in enumerate(question_texts):
                    prompt_lines.append(f"{i+1}. {q_text}")
                prompt_lines.append("\nYour JSON response:")
                category_user_prompt_text = "\n".join(prompt_lines)

                messages_for_this_call = self.current_conversation_history + [
                    {"role": "user", "content": category_user_prompt_text} # Text only for category Qs
                ]
                
                estimated_max_tokens = 70 + (len(current_questions_being_asked) * 35) # Increased estimate
                raw_llm_response = self._call_llm_with_history(messages_for_this_call, temperature=0.1, max_tokens=estimated_max_tokens)

                self.current_conversation_history.append({"role": "user", "content": category_user_prompt_text})
                self.current_conversation_history.append({"role": "assistant", "content": raw_llm_response})

                llm_answers_list = self._parse_llm_json_list_response(raw_llm_response, len(current_questions_being_asked))
                
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
                    break 
                else:
                    print(f"  --- Incorrect answer(s) in category '{category_name}' for this attempt. ---")
                    if attempt >= max_retries_per_category:
                        print(f"  --- Max retries for category '{category_name}' reached. This category FAILED. ---")
                        all_categories_passed = False 
                        break 
                    else:
                        print(f"    Will retry category '{category_name}' with new questions if available.")
            
            if not passed_this_category: 
                all_categories_passed = False 
                break 

        self.map_verified_successfully = all_categories_passed
        if self.map_verified_successfully:
            print("\nSUCCESS: All categories verified successfully using map image and description (if provided)!")
        else:
            print("\nFAILURE: Not all categories could be verified successfully.")
        return self.map_verified_successfully


    def analyze_vehicle_interactions(self, vehicle_actions: List[str]) -> Optional[str]:
        """
        Analyzes vehicle interactions, using established context (image and description).
        """
        if not self.map_verified_successfully: # Removed image check here, as it's part of context
            print("Error: Map understanding has not been successfully verified for the current map context.")
            print("Please call `verify_map_understanding()` successfully first.")
            return None
        
        if not self.current_map_image_base64: # Add a check here just in case, though it should be set if verified
            print("Error: Map image base64 not found in current context. This should not happen if verification was successful.")
            return None


        actions_text = "\n".join([f"- {action}" for action in vehicle_actions])
        
        analysis_prompt_text_parts_list = [] # Will build list of content dicts

        analysis_intro_text = (
            "You have previously demonstrated an understanding of the provided map context (which included an image and possibly a textual description) "
            "by correctly answering questions about its features across several categories. "
            "Now, considering that same map context and the following vehicle actions, "
            "please analyze potential interactions between vehicles.\n\n"
            "Focus on identifying and describing:\n"
            "1. Conflicts, 2. Potential Collisions, 3. Necessary Yielding Maneuvers, 4. Other safety-critical observations.\n\n"
            "If no significant interactions or safety concerns are found, please state that clearly.\n\n"
            "Vehicle Actions:\n"
            f"{actions_text}\n\n"
            "Your detailed analysis of interactions:"
        )
        analysis_prompt_text_parts_list.append({"type": "text", "text": analysis_intro_text})
        
        # Re-send image for robustness in this critical analysis step.
        analysis_prompt_text_parts_list.append(
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.current_map_image_base64}"}}
        )
        # Optionally, re-send description text if it's crucial and short, or a summary.
        # This reinforces the textual context directly with the task, rather than just relying on history.
        if self.current_map_description:
           summary_desc = self.current_map_description[:500] + ("..." if len(self.current_map_description) > 500 else "")
           analysis_prompt_text_parts_list.append({"type": "text", "text": f"\nReminder of Map Description Context:\n---\n{summary_desc}\n---"})


        messages_for_analysis_call = self.current_conversation_history + [
            {"role": "user", "content": analysis_prompt_text_parts_list} # Use the list of content dicts
        ]

        print("\n--- Requesting Vehicle Interaction Analysis (with map image & description context) ---")
        llm_analysis_response = self._call_llm_with_history(messages_for_analysis_call, temperature=0.4, max_tokens=1000)

        if "ERROR_API_CALL" not in llm_analysis_response and "ERROR_LLM_NO_CONTENT" not in llm_analysis_response:
            self.current_conversation_history.append({"role": "user", "content": analysis_prompt_text_parts_list})
            self.current_conversation_history.append({"role": "assistant", "content": llm_analysis_response})
            return llm_analysis_response
        else:
            print("Failed to get interaction analysis from LLM.")
            return None

# --- Example Usage ---
if __name__ == "__main__":
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
        # It's good practice to ensure OPENAI_API_KEY is set before this, or pass it explicitly.
        # Example: interpreter = MapInterpretation(categorized_qa_pairs=categorized_sample_qa_pairs, api_key="sk-...")
        interpreter = MapInterpretation(categorized_qa_pairs=categorized_sample_qa_pairs)
    except ValueError as e:
        print(f"Initialization Error: {e}")
        exit()

    # Define paths for your specific map image and its description
    # Ensure these paths are correct for your environment.
    map_image_location = "./data/processed/inD/map/01_bendplatz_graph.jpeg" # Your actual image path
    map_description_file_path = "./data/processed/inD/map/01_bendplatz_description.txt" # Your actual description file path

    # Check if files exist
    if not os.path.exists(map_image_location):
        print(f"Error: Map image file not found at '{map_image_location}'. Please check the path.")
        exit()
    if not os.path.exists(map_description_file_path):
        print(f"Warning: Map description file not found at '{map_description_file_path}'. Proceeding without description.")
        # Set to None if you want to proceed without it, or handle as an error if description is mandatory.
        # map_description_file_path = None


    print(f"\nStarting categorized map verification for: {map_image_location}")
    if map_description_file_path:
        print(f"Using map description from: {map_description_file_path}")
    
    is_understood = interpreter.verify_map_understanding(
        image_path_or_url=map_image_location,
        map_description_path=map_description_file_path, # Pass the description path
        questions_per_category=3, 
        max_retries_per_category=1  
    )

    if is_understood:
        print("\nOVERALL SUCCESS: LLM map understanding verified using image and description (if provided).")
        
        '''
        # Example vehicle actions for your specific map scenario (you'll need to define these)
        vehicle_actions_for_inD_map = [
            "Vehicle 1 (Car) enters from Road 0, intending to turn left onto Road 1.",
            "Vehicle 2 (Truck) enters from Road 3, intending to go straight onto Road 0.",
            "Vehicle 3 (Motorcycle) enters from Road 4, intending to turn right onto Road 0.",
            # Add more actions relevant to the inD dataset map's intersection
        ]

        print("\n--- Now, analyzing vehicle interactions (with description context) ---")
        analysis_result = interpreter.analyze_vehicle_interactions(vehicle_actions=vehicle_actions_for_inD_map)

        if analysis_result:
            print("\n--- Vehicle Interaction Analysis Complete ---")
            # The result is already printed within the method, but you can process it further.
            output_filename = "vehicle_interaction_analysis_inD_01.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(analysis_result)
            print(f"Analysis saved to {output_filename}")
        else:
            print("\n--- Vehicle Interaction Analysis Failed ---")
        '''
            
    else:
        print("\nOVERALL FAILURE: LLM map understanding could not be verified.")