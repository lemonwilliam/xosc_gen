import openai
import base64
import requests
import json
import random
import os
import re # For robust JSON parsing
from typing import List, Dict, Any, Optional, Tuple

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback

# Set OPENAI_API_KEY as an environment variable or pass it to the constructor.

class SceneInterpretation:
    def __init__(self, 
                 qa_json_path: str, # MODIFIED: Takes path to QA JSON file
                 api_key: Optional[str] = None, 
                 model: str = "gpt-4o"):
        """
        Initializes the MapInterpretation class.

        Args:
            qa_json_path: Path to a JSON file containing categorized QA pairs.
                          The JSON structure should be: 
                          {
                              "CategoryName1": [{"question": "...", "answer": "..."}],
                              "CategoryName2": [{"question": "...", "answer": "..."}]
                          }
            api_key: Your OpenAI API key.
            model: The OpenAI model to use.
        """
        self.categorized_qa_pairs = self._load_qa_from_json(qa_json_path)
        self.common_sense = self._read_text_file("./memos/common_sense.txt", "Common sense")   
        self.model_name = model
        
        # --- LangChain Setup ---
        # 1. Initialize the LLM
        self.llm = ChatOpenAI(
            model=self.model_name, 
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.1 # Set a default temperature
        )

        # 2. Create a prompt template with a placeholder for history
        self.prompt_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}")
        ])
        
        # 3. Create the main chain
        chain = self.prompt_template | self.llm | StrOutputParser()

        # 4. Set up the message history store
        # This dictionary will hold history objects for different sessions
        self.message_stores = {}

        # 5. Wrap the chain in RunnableWithMessageHistory
        self.conversational_chain = RunnableWithMessageHistory(
            chain,
            # A function to get the message history for a given session_id
            lambda session_id: self.message_stores.get(session_id, InMemoryChatMessageHistory()),
            input_messages_key="input",
            history_messages_key="history",
        )
        self.current_map_image_base64: Optional[str] = None
        self.current_map_description: Optional[str] = None
        self.map_verified_successfully: bool = False

        self.verification_token_usage: Dict[str, int] = {
            'prompt_tokens': 0, 
            'completion_tokens': 0, 
            'total_tokens': 0
        }
        
        print(f"MapInterpretation initialized with {len(self.categorized_qa_pairs)} QA categories from '{qa_json_path}', using model {self.model_name}.")

    def _load_qa_from_json(self, qa_json_path: str) -> Dict[str, List[Dict[str, str]]]: # NEW METHOD
        """Loads categorized QA pairs from a JSON file."""
        try:
            with open(qa_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"QA JSON file not found at: {qa_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from file: {qa_json_path}")
        
        # Basic validation of the loaded structure
        if not isinstance(data, dict):
            raise ValueError(f"QA JSON content in '{qa_json_path}' must be a dictionary (categories as keys).")
        
        validated_data: Dict[str, List[Dict[str, str]]] = {}
        for category, qas in data.items():
            if not isinstance(category, str):
                raise ValueError(f"Category names in '{qa_json_path}' must be strings.")
            if not isinstance(qas, list):
                raise ValueError(f"Value for category '{category}' in '{qa_json_path}' must be a list of QA pairs.")
            if not qas: # Allow empty categories if intended, or raise error:
                print(f"Warning: Category '{category}' in '{qa_json_path}' is empty.")
            
            validated_qas_for_category = []
            for qa_pair in qas:
                if not isinstance(qa_pair, dict) or \
                   'question' not in qa_pair or not isinstance(qa_pair['question'], str) or \
                   'answer' not in qa_pair or not isinstance(qa_pair['answer'], str):
                    raise ValueError(
                        f"Each item in category '{category}' in '{qa_json_path}' must be a dictionary "
                        "with 'question' (str) and 'answer' (str) keys."
                    )
                validated_qas_for_category.append({'question': qa_pair['question'], 'answer': qa_pair['answer']})
            validated_data[category] = validated_qas_for_category
            
        if not validated_data:
             raise ValueError(f"QA JSON file '{qa_json_path}' resulted in no valid categories being loaded.")
             
        return validated_data

    def _read_text_file(self, file_path: Optional[str], file_description_for_error: str) -> Optional[str]:
        """Reads text content from a file path."""
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                print(f"Warning: {file_description_for_error} file not found at {file_path}. Proceeding without it.")
                return None
            except Exception as e:
                print(f"Warning: Error reading {file_description_for_error} file {file_path}: {e}. Proceeding without it.")
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

    # Verify_map_understanding
    def verify_map_understanding(self, 
                                 image_path_or_url: str,
                                 map_description_path: Optional[str] = None,
                                 max_retries_per_category: int = 2) -> Tuple[bool, Dict[str, int]]:
        """
        Verifies LLM's understanding using image and optional text description.
        LLM answers ALL questions in a category. Retries on a per-category basis 
        if answers are incorrect, re-presenting all questions for that category.
        """
        self.map_verified_successfully = False

        # Reset token counter for this run
        self.verification_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        # Use the image path as a unique session ID for this conversation
        session_id = image_path_or_url 
        self.message_stores[session_id] = InMemoryChatMessageHistory()
        
        try:
            self.current_map_image_base64 = self._encode_image_to_base64(image_path_or_url)
        except IOError as e: 
            print(f"Critical error loading image: {e}")
            return False, self.verification_token_usage

        # --- Initial Briefing using LangChain message types ---
        system_prompt = self._read_text_file("./memos/system_prompt.txt", "System prompt")
        self.current_map_description = self._read_text_file(map_description_path, "Map description")
        
        initial_user_prompt_parts = [
            {"type": "text", "text": "Here is the common sense for a road user."},
            {"type": "text", "text": f"\nCommon Sense:\n---\n{self.common_sense}\n---"},
            {"type": "text", "text": "Here is the map diagram and its corrsponding structured description. Please utilize both to gain understanding on the map."},
            {"type": "text", "text": f"\nMap Description:\n---\n{self.current_map_description}\n---"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.current_map_image_base64}"}}
        ]

        # Manually add the initial messages to the history for this session
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=initial_user_prompt_parts),
            AIMessage(content="Understood. I have analyzed the provided information. I am ready for your questions.")
        ]
        self.message_stores[session_id].add_messages(initial_messages)


        all_categories_passed = True
        for category_name, qa_list_for_category in self.categorized_qa_pairs.items():
            print(f"\n--- Verifying Category: {category_name} ---")
            if not qa_list_for_category:
                print(f"    No questions to ask for '{category_name}'. Marking as passed (vacuously true).")
                continue

            passed_this_category = False
            incorrect_details_from_previous_attempt: List[Dict[str, str]] = [] 

            for attempt in range(max_retries_per_category + 1):
                print(f"  Attempt {attempt + 1}/{max_retries_per_category + 1} for category '{category_name}'")

                current_questions_being_asked = qa_list_for_category 
                
                question_texts = [qa['question'] for qa in current_questions_being_asked]
                
                # MODIFICATION: Construct retry prompt with details of previous errors
                prompt_lines = []
                if attempt > 0 and incorrect_details_from_previous_attempt: # This is a retry and there were errors
                    prompt_lines.extend([
                        f"You previously attempted questions for category '{category_name}'. Some answers were incorrect.",
                        "Please carefully review your reasoning for the following questions based on the map information provided earlier:",
                    ])
                    for i, detail in enumerate(incorrect_details_from_previous_attempt):
                        prompt_lines.append(f"  Incorrect Item {i+1}:")
                        prompt_lines.append(f"    Question: \"{detail['question']}\"")
                        prompt_lines.append(f"    Your (incorrect) answer: \"{detail['llm_answer']}\"")
                        prompt_lines.append(f"    The correct answer is: \"{detail['correct_answer']}\"")
                    prompt_lines.append("\nKeeping these corrections in mind, please re-answer ALL questions for this category:")
                else: # First attempt for this category
                    prompt_lines.append(
                        f"Now, focusing on the '{category_name}' category for the map information (image and text description, if provided) that you have already processed."
                    )

                prompt_lines.extend([
                    "Please answer the following questions based on the previously provided map image and its description.",
                    "Provide your answers *only* as a JSON list of strings, corresponding to the order of the questions.",
                    "Do not add any other explanatory text before or after the JSON list itself.\n",
                    "Questions:"
                ])
                for i, q_text in enumerate(question_texts):
                    prompt_lines.append(f"{i+1}. {q_text}")
                prompt_lines.append("\nYour JSON response:")

                category_user_prompt_text = "\n".join(prompt_lines)

                # Call the LLM using the LangChain runnable
                print(f"\n--- Sending to LLM via LangChain ({self.model_name}) ---")

                raw_llm_response = ""
                # Use the LangChain callback to track token usage
                with get_openai_callback() as cb:
                    response = self.conversational_chain.invoke(
                        {"input": category_user_prompt_text},
                        config={"configurable": {"session_id": session_id}}
                    )
                    raw_llm_response = response
                    
                    # Accumulate token usage
                    self.verification_token_usage['prompt_tokens'] += cb.prompt_tokens
                    self.verification_token_usage['completion_tokens'] += cb.completion_tokens
                    self.verification_token_usage['total_tokens'] += cb.total_tokens
                    print(f"    Tokens for this call: {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")
                
                llm_answers_list = self._parse_llm_json_list_response(raw_llm_response, len(qa_list_for_category))
                
                num_correct_in_this_attempt = 0
                total_questions_in_attempt = len(current_questions_being_asked)
                
                # MODIFICATION: Clear and repopulate incorrect_details for the *next* potential retry
                current_attempt_incorrect_details: List[Dict[str, str]] = [] 

                incorrect_in_this_attempt = False
                if len(llm_answers_list) != total_questions_in_attempt:
                    print(f"    Warning: LLM returned {len(llm_answers_list)} answers for category '{category_name}', but {total_questions_in_attempt} questions were asked.")
                    incorrect_in_this_attempt = True
                    # Mark all as incorrect if length mismatch
                    for i, qa_pair_expected in enumerate(current_questions_being_asked):
                        current_attempt_incorrect_details.append({
                            'question': qa_pair_expected['question'],
                            'llm_answer': "ERROR_LENGTH_MISMATCH" if i >= len(llm_answers_list) else llm_answers_list[i],
                            'correct_answer': qa_pair_expected['answer']
                        })
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
                            current_attempt_incorrect_details.append({ # MODIFICATION: Capture details
                                'question': qa_pair_expected['question'],
                                'llm_answer': llm_answers_list[i],
                                'correct_answer': qa_pair_expected['answer']
                            })
                        else:
                            print("      Result: CORRECT")
                            num_correct_in_this_attempt += 1
                
                # Update for the next potential retry
                incorrect_details_from_previous_attempt = current_attempt_incorrect_details

                correct_rate = 0.0
                if total_questions_in_attempt > 0:
                    correct_rate = (num_correct_in_this_attempt / total_questions_in_attempt) * 100
                    print(f"    Category '{category_name}' - Attempt {attempt + 1} Correct Rate: {num_correct_in_this_attempt}/{total_questions_in_attempt} ({correct_rate:.2f}%)")
                else:
                    print(f"    Category '{category_name}' - Attempt {attempt + 1}: No questions were processed to calculate a rate.")
                
                if not incorrect_in_this_attempt:
                    print(f"  +++ Category '{category_name}' passed this attempt! +++")
                    passed_this_category = True
                    break
                elif correct_rate >= 95.0:
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
                        print(f"    Will retry category '{category_name}' (all questions), providing corrections.") 
            
            if not passed_this_category: 
                all_categories_passed = False 
                break 

        self.map_verified_successfully = all_categories_passed
        
        # Print a final summary of token usage
        print("\n--- Verification Phase Token Usage Summary ---")
        print(f"Total Prompt Tokens:     {self.verification_token_usage['prompt_tokens']}")
        print(f"Total Completion Tokens: {self.verification_token_usage['completion_tokens']}")
        print(f"Total Tokens Consumed:   {self.verification_token_usage['total_tokens']}")
        print("--------------------------------------------")
        
        # Clean up the message store for this session if desired
        del self.message_stores[session_id]

        return self.map_verified_successfully, self.verification_token_usage


    def analyze_vehicle_interactions(self, 
                                     agent_actions_file_path: str,
                                     trigger_conditions_file_path: str) -> Optional[str]:
        """
        Analyzes agent interactions on the map, using established map context,
        agent actions from a file, and a list of trigger conditions from another file.
        """
        if not self.map_verified_successfully:
            print("Error: Map understanding has not been successfully verified.")
            return None
        
        if not self.current_map_image_base64:
            print("Error: Map image base64 not found in current context.")
            return None

        agent_actions_content = self._read_text_file(agent_actions_file_path, "Agent actions")
        trigger_conditions_content = self._read_text_file(trigger_conditions_file_path, "Trigger conditions definitions")

        if not agent_actions_content:
            print("Error: Cannot proceed without agent actions content.")
            return None
        if not trigger_conditions_content:
            print("Error: Cannot proceed without trigger conditions definitions.")
            return None

        # Build the detailed prompt for interaction analysis
        analysis_prompt_parts_list = []

        # Part 1: Introduction and Goal
        intro_text = (
            "You have previously demonstrated an understanding of the provided map context (image and possibly description).\n"
            "Your task now is to analyze a sequence of agent (vehicle, bicycle, pedestrian) actions occurring on this map "
            "to identify significant interactions between agents. For actions that are part of an interaction (e.g., yielding, "
            "emergency braking due to another agent, speeding up after a yielding agent passes), "
            "you must identify the most likely trigger condition from the provided list of 'Action Triggers' and explain your reasoning.\n\n"
            "Not every action will have an explicit trigger from the list, especially initial actions like 'Enters scenario'. Focus on "
            "actions that demonstrate a reaction or coordination between agents."
        )
        analysis_prompt_parts_list.append({"type": "text", "text": intro_text})

        # Part 2: Remind of Map Context (Image and Description if available)
        map_context_reminder_text = "\n--- Map Context Recap ---\n"
        map_context_reminder_text += "You have already analyzed the following map. Keep its structure in mind.\n"
        if self.current_map_description:
            summary_desc = self.current_map_description[:300] + ("..." if len(self.current_map_description) > 300 else "") # Shorter summary
            map_context_reminder_text += f"Key elements from description: {summary_desc}\n"
        
        analysis_prompt_parts_list.append({"type": "text", "text": map_context_reminder_text})
        analysis_prompt_parts_list.append(
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.current_map_image_base64}"}}
        )

        # Part 3: Agent Actions
        agent_actions_header_text = "\n--- Agent Actions Log ---\n"
        agent_actions_header_text += "Here are the actions performed by agents in the scenario:\n"
        analysis_prompt_parts_list.append({"type": "text", "text": agent_actions_header_text})
        analysis_prompt_parts_list.append({"type": "text", "text": agent_actions_content})
        
        # Part 4: Trigger Conditions Definitions
        trigger_conditions_header_text = "\n--- Possible Action Triggers (Choose from this list) ---\n"
        trigger_conditions_header_text += "When an agent's action is part of an interaction, select the most appropriate trigger from this list:\n"
        analysis_prompt_parts_list.append({"type": "text", "text": trigger_conditions_header_text})
        analysis_prompt_parts_list.append({"type": "text", "text": trigger_conditions_content})

        # Part 5: Output Format and Instructions
        output_instructions_text = (
            "\n--- Your Analysis Task & Output Format ---\n"
            "1. Identify sequences of actions that constitute an interaction between two or more agents.\n"
            "2. For each *reacting* agent in an interaction, pinpoint the specific action(s) that are a response.\n"
            "3. For these reacting actions, state the most likely 'Action Trigger' from the provided list by its name (e.g., 'Time Headway Condition', 'Storyboard Element State Condition').\n"
            "4. Briefly explain the interaction and why you chose that trigger for the specific action.\n\n"
            "Example Output Structure (for each identified interaction):\n"
            "Interaction between [Agent X] and [Agent Y]:\n"
            "  - Scenario: [Briefly describe what happened, e.g., Agent Y arrived at intersection first, Agent X approached intending to cross paths.]\n"
            "  - Agent [Reacting Agent ID]'s action: \"[The specific action string from the log, e.g., Significantly slows down at t=0.32 till stopped.]\"\n"
            "    - Trigger: [Name of Trigger Condition from the list]\n"
            "    - Reason: [Your explanation, e.g., To yield to Agent Y which had priority or was already in the conflicting path.]\n"
            "  - Agent [Reacting Agent ID]'s subsequent action (if part of same interaction event): \"[e.g., Moderately speeds up at t=7.84.]\"\n"
            "    - Trigger: [Name of Trigger Condition]\n"
            "    - Reason: [e.g., Agent Y has cleared the conflict zone, allowing Agent X to proceed.]\n\n"
            "If there are no significant interactions, state that clearly.\n\n"
            "Begin your analysis now:"
        )
        analysis_prompt_parts_list.append({"type": "text", "text": output_instructions_text})

        # Combine with existing conversation history
        messages_for_analysis_call = self.current_conversation_history + [
            {"role": "user", "content": analysis_prompt_parts_list}
        ]

        print("\n--- Requesting Agent Interaction Analysis ---")
        # Max tokens needs to be high for this kind of detailed analysis + input text
        llm_analysis_response = self._call_llm_with_history(messages_for_analysis_call, temperature=0.5, max_tokens=2000) # Increased max_tokens and slightly temp

        if "ERROR_API_CALL" not in llm_analysis_response and "ERROR_LLM_NO_CONTENT" not in llm_analysis_response:
            self.current_conversation_history.append({"role": "user", "content": analysis_prompt_parts_list}) # Add the complex user prompt
            self.current_conversation_history.append({"role": "assistant", "content": llm_analysis_response})
            return llm_analysis_response
        else:
            print("Failed to get interaction analysis from LLM.")
            return None

# --- Example Usage ---
if __name__ == "__main__":

    # Define paths for inputs
    qa_json_path = "./data/processed/inD/map/01_bendplatz_questions.json"
    map_image_path = "./data/processed/inD/map/01_bendplatz_graph.jpeg"
    map_description_path = "./data/processed/inD/map/01_bendplatz_description.txt"

    # Check if files exist
    if not os.path.exists(qa_json_path):
        print(f"Error: Map image file not found at '{qa_json_path}'. Please check the path.")
        exit()
    if not os.path.exists(map_image_path):
        print(f"Error: Map image file not found at '{map_image_path}'. Please check the path.")
        exit()
    if not os.path.exists(map_description_path):
        print(f"Warning: Map description file not found at '{map_description_path}'. Proceeding without description.")
        # Set to None if you want to proceed without it, or handle as an error if description is mandatory.
        # map_description_file_path = None

    try:
        interpreter = SceneInterpretation(
            qa_json_path=qa_json_path,
            # api_key="sk-..." # Optionally pass API key here if not in env
            model="gpt-4o"
        )
    except ValueError as e:
        print(f"Initialization Error: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        exit()

    is_understood, token_info = interpreter.verify_map_understanding(
        image_path_or_url=map_image_path,
        map_description_path=map_description_path, 
        max_retries_per_category=2  
    )

    print(f"\nVerification process finished. Final Token Usage: {token_info['total_tokens']} tokens.")

    if is_understood:
        print("\nOVERALL SUCCESS: LLM map understanding verified using image and description (if provided).")

        '''

        agent_actions_file = "./results/inD/description/08_1250_1600.txt"
        trigger_conditions_file = "./memos/condition_definition.txt"
        
        print("\n--- Now, analyzing agent interactions ---")
        analysis_result = interpreter.analyze_vehicle_interactions(
            agent_actions_file_path=agent_actions_file,
            trigger_conditions_file_path=trigger_conditions_file
        )

        if analysis_result:
            print("\n--- Agent Interaction Analysis Complete ---")
            # The result is already printed by the _call_llm_with_history method within analyze_vehicle_interactions
            output_filename = "agent_interaction_analysis_output.txt" 
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(analysis_result)
            print(f"Full analysis saved to {output_filename}")
        else:
            print("\n--- Agent Interaction Analysis Failed ---")
        '''
            
    else:
        print("\nOVERALL FAILURE: LLM map understanding could not be verified.")