import asyncio
import pandas as pd
import json
import os
import re
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import time

# --- Configuration ---
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_API_BASE")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

INPUT_CSV_PATH = 'gsm8k.csv'
OUTPUT_CSV_PATH = 'gsm8k_qwen2.5-32b_gov.csv'
MAX_CONCURRENT_REQUESTS = 4
API_MODEL = "Qwen/Qwen2.5-32B-Instruct"
REQUEST_TIMEOUT = 90
RETRY_ATTEMPTS = 5
RETRY_DELAY = 5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VERIFICATION_SYSTEM_PROMPT = ""

# Initialize the Async OpenAI Client
if API_BASE_URL:
    logging.info(f"Using custom API base URL: {API_BASE_URL}")
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT)
else:
    logging.info("Using default OpenAI API base URL.")
    client = AsyncOpenAI(api_key=API_KEY, timeout=REQUEST_TIMEOUT)

def parse_verification_response(response_text):
    """
    Parses the LLM's response.
    - The entire response is considered the 'analysis'.
    - The conclusion ('CORRECT'/'INCORRECT') is extracted from the *last*
      \\boxed{...} marker found in the text that *contains* 'correct' or
      'incorrect' (case-insensitive).
    - If no such \\boxed{...} marker provides a conclusion, it checks the
      entire response_text for 'incorrect'. If found, conclusion is 'INCORRECT'.
      Otherwise, conclusion is 'CORRECT'.
    """
    analysis = response_text if response_text is not None else ""
    conclusion = "ERROR_PARSING"
    found_valid_conclusion_in_box = False
    last_valid_conclusion_from_box = None

    if not response_text:
        logging.warning("Received empty response text.")
        analysis = "Received empty response text."
        conclusion = "CORRECT"
        logging.info("Fallback due to empty response: Setting conclusion to CORRECT.")
        return {"analysis": analysis, "conclusion": conclusion}

    logging.debug(f"Attempting to parse response: ...{response_text[-200:]}")

    try:
        matches = list(re.finditer(r"\\boxed{(.*?)}", response_text, re.IGNORECASE | re.DOTALL))
        logging.debug(f"Found {len(matches)} potential \\boxed{{...}} structures.")

        if not matches:
             logging.debug(f"No \\boxed{{...}} structures found at all. Will use fallback.")
        else:
            for i, match in enumerate(matches):
                extracted_text = match.group(1).strip()
                lower_extracted_text = extracted_text.lower()
                logging.debug(f"Checking box #{i+1}/{len(matches)}: Content='{extracted_text[:100]}...'")
                if "incorrect" in lower_extracted_text:
                    last_valid_conclusion_from_box = "INCORRECT"
                    found_valid_conclusion_in_box = True
                    logging.debug(f"  -> Found 'incorrect' in box #{i+1}.")
                elif "correct" in lower_extracted_text:
                    last_valid_conclusion_from_box = "CORRECT"
                    found_valid_conclusion_in_box = True
                    logging.debug(f"  -> Found 'correct' in box #{i+1}.")
                else:
                     logging.debug(f"  -> No keywords found in box #{i+1}.")
                     pass

        if found_valid_conclusion_in_box:
            conclusion = last_valid_conclusion_from_box
            logging.info(f"Successfully parsed conclusion from the last valid box: {conclusion}")
        else:
            logging.warning(
                f"No valid \\boxed{{...}} conclusion found (checked {len(matches)} boxes). "
                f"Falling back to raw text search for 'incorrect'."
            )
            if "incorrect" in response_text.lower():
                conclusion = "INCORRECT"
                logging.info(f"Fallback: Found 'incorrect' in raw text. Conclusion: {conclusion}")
            else:
                conclusion = "CORRECT"
                logging.info(f"Fallback: Did not find 'incorrect' in raw text. Conclusion: {conclusion}")

        analysis = analysis.strip()

    except Exception as e:
        logging.error(f"Error parsing LLM response during regex/iteration: {e}. Raw text: {response_text}")
        analysis = f"Parsing failed due to exception: {e}. Original text: {response_text}"
        conclusion = "ERROR_PARSING_EXCEPTION"

    return {"analysis": analysis, "conclusion": conclusion}


async def call_verifier_llm(problem, preceding_text, current_step_text, row_id, step_index):
    """
    Calls the LLM API for verification of a single step with retry logic.
    """
    user_message = f"""The following is a math problem, the verified preceding steps of its solution, and the current step to be evaluated.
[Math Problem]
{problem}

[Correct Preceding Steps]
{preceding_text if preceding_text else "No preceding steps."}

[Current Step]
{current_step_text}

Your task is to judge whether the [Current Step] is correct. Once you identify an error in the [Current Step], return "incorrect". Otherwise, return "correct."
Please think step by step, and put your final conclusion (i.e., 'correct' or 'incorrect') in \\boxed{{}}.
"""
    messages = []
    if VERIFICATION_SYSTEM_PROMPT:
         messages.append({"role": "system", "content": VERIFICATION_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_message})

    current_attempt = 0
    while current_attempt < RETRY_ATTEMPTS:
        try:
            logging.debug(f"Requesting verification for ID: {row_id}, Step Index: {step_index} (Attempt {current_attempt + 1})")
            response = await client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            logging.debug(f"Raw verification response for ID {row_id}, Step Index {step_index}: {content[:200]}...")
            parsed_result = parse_verification_response(content)
            parsed_result['error'] = None
            parsed_result['raw_response'] = content
            return parsed_result

        except RateLimitError as e:
            current_attempt += 1
            wait_time = RETRY_DELAY * (2 ** current_attempt)
            logging.warning(f"Rate limit for ID {row_id}, Step Index {step_index}. Retrying in {wait_time}s... ({current_attempt}/{RETRY_ATTEMPTS})")
            if current_attempt >= RETRY_ATTEMPTS:
                logging.error(f"Rate limit error persisted for ID {row_id}, Step Index {step_index}: {e}")
                return {"analysis": None, "conclusion": "ERROR_API", "error": f"Rate limit error: {e}", "raw_response": None}
            await asyncio.sleep(wait_time)
        except (APIError, APIConnectionError) as e:
             current_attempt += 1
             wait_time = RETRY_DELAY * (2 ** current_attempt)
             logging.warning(f"API/Connection Error for ID {row_id}, Step Index {step_index}: {e}. Retrying in {wait_time}s... ({current_attempt}/{RETRY_ATTEMPTS})")
             if current_attempt >= RETRY_ATTEMPTS:
                 logging.error(f"API/Connection error persisted for ID {row_id}, Step Index {step_index}: {e}")
                 return {"analysis": None, "conclusion": "ERROR_API", "error": f"API/Connection error: {e}", "raw_response": None}
             await asyncio.sleep(wait_time)
        except Exception as e:
            logging.error(f"Unexpected API error for ID {row_id}, Step Index {step_index}: {e}")
            if "Could not find model" in str(e) or "Invalid API key" in str(e) or "authentication" in str(e).lower():
                 return {"analysis": None, "conclusion": "ERROR_CONFIG", "error": f"Configuration Error: {e}", "raw_response": None}
            current_attempt += 1
            wait_time = RETRY_DELAY * (2 ** current_attempt)
            if current_attempt >= RETRY_ATTEMPTS:
                 logging.error(f"Unexpected API error persisted for ID {row_id}, Step Index {step_index}: {e}")
                 return {"analysis": None, "conclusion": "ERROR_UNEXPECTED", "error": f"Unexpected API error: {e}", "raw_response": None}
            logging.warning(f"Retrying unexpected error in {wait_time}s...")
            await asyncio.sleep(wait_time)
    logging.error(f"Failed verification for ID {row_id}, Step Index {step_index} after all retries.")
    return {"analysis": None, "conclusion": "ERROR_RETRY_FAILED", "error": "Failed after multiple retries", "raw_response": None}


async def verify_segments_for_row(row_data, semaphore):
    """
    Processes a single row: Splits model_response into steps, verifies them
    sequentially, and determines the final row status and the index of the first error step.
    Returns: (row_id, verification_results_list, final_status, error_step_index)
    """
    async with semaphore:
        row_id = row_data.get('id', 'N/A')
        problem = str(row_data.get('problem', ''))
        model_response = str(row_data.get('model_response', ''))

        try: hash(row_id)
        except TypeError: row_id = str(row_id)

        final_status = "ERROR_SETUP"
        error_step_index = -999
        verification_results = []

        if not problem:
            logging.warning(f"Skipping row {row_id}: Missing 'problem'.")
            return row_id, [{"error": "Missing problem statement"}], "ERROR_SETUP", error_step_index

        if not model_response:
            logging.warning(f"Skipping row {row_id}: Empty 'model_response'.")
            return row_id, [{"info": "No steps found in model_response"}], "CORRECT", -1

        steps = [s.strip() for s in model_response.strip().split('\n\n') if s.strip()]

        if not steps:
            logging.warning(f"Skipping row {row_id}: No non-empty steps found after splitting 'model_response'.")
            return row_id, [{"info": "No non-empty steps found in model_response"}], "CORRECT", -1

        preceding_text = ""
        final_status = "CORRECT"
        error_step_index = -1

        for i, current_step_text in enumerate(steps):
            step_index = i

            verification_data = await call_verifier_llm(
                problem, preceding_text, current_step_text, row_id, step_index
            )

            verification_results.append({
                "step_index": step_index,
                "step_text": current_step_text,
                "analysis": verification_data.get("analysis"),
                "conclusion": verification_data.get("conclusion"),
                "error": verification_data.get("error"),
                "raw_response": verification_data.get("raw_response")
            })

            conclusion_state = verification_data.get("conclusion")
            api_or_config_error = verification_data.get("error")

            if api_or_config_error:
                 stop_reason = f"API/config error ({api_or_config_error} / {conclusion_state})"
                 logging.warning(f"Stopping verification for row {row_id} at step {step_index} due to: {stop_reason}.")
                 final_status = "ERROR_API_CONFIG"
                 error_step_index = step_index
                 break
            elif conclusion_state == "INCORRECT":
                 stop_reason = "incorrect step"
                 logging.warning(f"Stopping verification for row {row_id} at step {step_index} due to: {stop_reason}.")
                 final_status = "INCORRECT"
                 error_step_index = step_index
                 break
            elif str(conclusion_state).startswith("ERROR_") and conclusion_state != "ERROR_API":
                 if final_status == "CORRECT":
                     stop_reason = f"verification error ({conclusion_state})"
                     logging.warning(f"Verification error found for row {row_id} at step {step_index}: {stop_reason}. Continuing verification.")
                     final_status = "ERROR_VERIFICATION"
                     error_step_index = step_index

            if not api_or_config_error and conclusion_state == "CORRECT":
                if preceding_text:
                    preceding_text += "\n\n" + current_step_text
                else:
                    preceding_text = current_step_text

        logging.info(f"Finished verification for row {row_id}. Steps processed: {len(verification_results)}. Final Status: {final_status}, Error Step Index: {error_step_index}")
        return row_id, verification_results, final_status, error_step_index


async def main():
    """
    Main function to read CSV, run verification tasks concurrently, and save results.
    """
    logging.info(f"Starting verification script. Reading CSV: {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        required_cols = ['id', 'problem', 'model_response']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             raise ValueError(f"Input CSV missing required columns: {missing_cols}")
        if 'id' in df.columns and not df['id'].is_unique:
            logging.warning("IDs in the 'id' column are not unique. Results for duplicate IDs might overwrite each other.")
    except FileNotFoundError:
        logging.error(f"Error: Input CSV file not found at {INPUT_CSV_PATH}")
        return
    except ValueError as e:
        logging.error(f"CSV Validation Error: {e}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return

    logging.info(f"Read {len(df)} rows from CSV.")
    logging.info(f"Using verification model: {API_MODEL}")
    logging.info(f"Max concurrent row processing tasks: {MAX_CONCURRENT_REQUESTS}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [verify_segments_for_row(row, semaphore) for row in df.to_dict('records')]

    results_data = {}
    logging.info("Starting verification API calls (processing rows concurrently)...")
    try:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Verifying Rows"):
            try:
                row_id, verification_list, final_status, error_step_idx = await future
                results_data[row_id] = {
                    'details': verification_list,
                    'status': final_status,
                    'error_step_idx': error_step_idx
                }
            except Exception as e:
                 logging.error(f"Error awaiting/unpacking row verification task result: {e}.")

    except Exception as e:
        logging.error(f"An error occurred during the main task execution loop (as_completed): {e}")

    logging.info("Verification processing finished.")
    logging.info(f"Results collected for {len(results_data)} rows out of {len(df)} attempted.")
    if results_data:
        status_counts = pd.Series([data['status'] for data in results_data.values()]).value_counts()
        logging.info(f"Final row statuses:\n{status_counts.to_string()}")
    else:
        logging.warning("No results were collected.")

    default_details = [{"error": "Processing did not yield result for this row"}]
    default_status = "ERROR_PROCESSING"
    default_error_idx = -999

    df['verification_results'] = df['id'].map(
        lambda x: json.dumps(results_data.get(x, {}).get('details', default_details))
    )
    df['final_verification_status'] = df['id'].map(
        lambda x: results_data.get(x, {}).get('status', default_status)
    )
    df['error_step_index'] = df['id'].map(
        lambda x: results_data.get(x, {}).get('error_step_idx', default_error_idx)
    )

    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        logging.info(f"Successfully saved verification results with status and error step index to {OUTPUT_CSV_PATH}")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {e}")

if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        is_running = loop.is_running()
    except RuntimeError:
        loop = None
        is_running = False

    if loop and is_running:
        logging.info("Asyncio loop already running. Adding main() to the loop.")
        tsk = loop.create_task(main())
    else:
        logging.info("Starting new asyncio event loop.")
        asyncio.run(main())