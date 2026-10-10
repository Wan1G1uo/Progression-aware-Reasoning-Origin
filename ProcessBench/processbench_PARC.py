import asyncio
import pandas as pd
import json
import os
import re
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError
from tqdm import tqdm
from dotenv import load_dotenv
import logging

# --- Configuration ---
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_API_BASE")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

INPUT_CSV_PATH = 'gsm8k.csv'
OUTPUT_CSV_PATH = 'gsm8k_qwen2.5-32b_parc.csv'
MAX_CONCURRENT_REQUESTS = 4
API_MODEL = "Qwen/Qwen2.5-32B-Instruct"
REQUEST_TIMEOUT = 120
RETRY_ATTEMPTS = 5
RETRY_DELAY = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if API_BASE_URL:
    logging.info(f"Using custom API base URL: {API_BASE_URL}")
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT)
else:
    logging.info("Using default OpenAI API base URL.")
    client = AsyncOpenAI(api_key=API_KEY, timeout=REQUEST_TIMEOUT)


# --- PARC Prompts ---

PREMISE_SYS_PROMPT = ""
PREMISE_USR_TEMPLATE = """You are provided with a question, a partial solution, and the next step in the solution. Your task is to identify the steps that serve as premises for the given next step.
A step qualifies as a premise if the next step directly relies on information from that step. Based on the identified premises, the correctness of the next step should be fully verifiable.

Question (Step 0):
{question}

Solution so far:
{solution}

Next step to analyze:
{step}

For the step above, identify which previous steps (including Step 0 - the question) are premises and explain why each one is necessary. Remember:
1. A step cannot be a premise to itself
2. The question (Step 0) can be a premise if used directly

Generate **ONLY** the premises and nothing else.
Format your response with one premise per line as:
Step X: [explanation of why this step is necessary for the current step]
If there are no required premises, output NONE."""

STANDALONE_SYS_PROMPT = """Your task is to determine whether a given sentence contains a clear mathematical error.
Check only the sentence itself. Mark it as "mathematical_error" only when the sentence explicitly contains an incorrect numerical or algebraic result.

Important rules:
1. Mathematical errors only come from wrong arithmetic or algebraic results in the sentence itself.
2. Do NOT mark an error just because the explanation is brief, skips obvious intermediate steps, uses different but equivalent notation, or could be more complete.
3. If the sentence contains no explicit numerical or algebraic calculation to verify, mark it as "correct".
4. If the sentence contains a concrete calculation or algebraic claim, verify it carefully before deciding."""

STANDALONE_USR_TEMPLATE = """
Statement to analyze:
{step}
Format your response as:

Reasoning: [detailed analysis of the statement's validity]
Verdict: [correct or mathematical_error]"""

CONTEXTUAL_SYS_PROMPT = """You are provided with a question, a statement that is one step in a solution, and the premises for that step. Your task is to decide whether the step follows from the premises and remains consistent with the quantities, entities, and constraints already established.
Mark the step as "logical_inconsistency" when it contradicts the premises or question, uses a quantity or entity incorrectly, makes an unsupported claim, or relies on a clearly invalid deduction.
Mark the step as "correct" when it is compatible with the premises and question and can be obtained from them by routine reasoning.

Important rules:
1. Assume the premises are correct. Only judge the current step.
2. Do NOT mark an error just because the step is terse, omits obvious intermediate reasoning, paraphrases earlier information, or is less detailed than you would prefer.
3. If the step is broadly supported by the premises and there is no clear contradiction, mark it as "correct".
4. Check carefully for unsupported jumps, invalid deductions, contradictions, wrong object counts, off-by-one setup mistakes, and misuse of quantities introduced earlier.
5. If a step carries forward or summarizes an already wrong quantity or relation, mark that step as "logical_inconsistency" rather than ignoring it as a mere restatement."""

CONTEXTUAL_USR_TEMPLATE = """
Given Premises:
Question:
{question}
Previous steps as premise:
{premises}
Statement to analyze:
{step}

Guidelines:
1. For logical_inconsistency, check whether the step misinterprets the premises, makes an invalid deduction, or introduces an unsupported assumption.
2. Do not check the correctness of the premises themselves.
3. Do not treat omitted obvious reasoning or harmless rephrasing as an error.
4. Focus on whether the step is actually supported by the premises and the question, not just whether it sounds plausible.
5. Pay special attention to counting mistakes, interval or spacing mistakes, wrong variable setup, and conclusions that reuse a value inconsistent with earlier established facts.


Format your response as:           
Reasoning: [detailed analysis of the statement's validity]
Verdict: [correct, logical_inconsistency]"""


# --- Parsing Utilities ---

def parse_premises_output(raw_response, step_index):
    """Parse LLM output to extract premise step indices."""
    if not raw_response:
        return []
    
    premises = []
    lines = raw_response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.upper() == "NONE":
            continue
        match = re.match(r'Step\s+(\d+)\s*:', line, re.IGNORECASE)
        if match:
            premise_idx = int(match.group(1))
            if premise_idx < step_index and premise_idx not in premises:
                premises.append(premise_idx)
    
    if not premises:
        nums = re.findall(r'\b(\d+)\b', raw_response)
        for num in nums:
            idx = int(num)
            if 0 <= idx < step_index and idx not in premises:
                premises.append(idx)
                if len(premises) >= 5:
                    break
    
    return sorted(premises)


def parse_standalone_verdict(raw_response):
    """Parse standalone error check output."""
    if not raw_response:
        return "correct"
    
    text = raw_response.lower()
    # Check for verdict line first
    verdict_match = re.search(r'verdict\s*:\s*(.+?)$', text, re.MULTILINE | re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).strip().lower()
        if "mathematical_error" in verdict:
            return "mathematical_error"
        if "correct" in verdict:
            return "correct"
    
    # Fallback: search whole text
    if "mathematical_error" in text:
        return "mathematical_error"
    if "incorrect" in text or "error" in text:
        return "mathematical_error"
    return "correct"


def parse_contextual_verdict(raw_response):
    """Parse contextual error check output."""
    if not raw_response:
        return "correct"
    
    text = raw_response.lower()
    verdict_match = re.search(r'verdict\s*:\s*(.+?)$', text, re.MULTILINE | re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).strip().lower()
        if "logical_inconsistency" in verdict:
            return "logical_inconsistency"
        if "correct" in verdict:
            return "correct"
    
    if "logical_inconsistency" in text:
        return "logical_inconsistency"
    if "inconsistency" in text or "incorrect" in text:
        return "logical_inconsistency"
    return "correct"


# --- API Calls ---

async def call_llm(messages, row_id, step_label, call_type):
    """General LLM API call with retry logic."""
    current_attempt = 0
    while current_attempt < RETRY_ATTEMPTS:
        try:
            logging.debug(f"Requesting {call_type} for ID: {row_id}, Step: {step_label} (Attempt {current_attempt + 1})")
            response = await client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            return content
        except RateLimitError as e:
            current_attempt += 1
            wait_time = RETRY_DELAY * (2 ** current_attempt)
            logging.warning(f"Rate limit for ID {row_id}, Step {step_label}. Retrying in {wait_time}s... ({current_attempt}/{RETRY_ATTEMPTS})")
            if current_attempt >= RETRY_ATTEMPTS:
                return None
            await asyncio.sleep(wait_time)
        except (APIError, APIConnectionError) as e:
            current_attempt += 1
            wait_time = RETRY_DELAY * (2 ** current_attempt)
            logging.warning(f"API/Connection Error for ID {row_id}, Step {step_label}: {e}. Retrying in {wait_time}s... ({current_attempt}/{RETRY_ATTEMPTS})")
            if current_attempt >= RETRY_ATTEMPTS:
                return None
            await asyncio.sleep(wait_time)
        except Exception as e:
            logging.error(f"Unexpected API error for ID {row_id}, Step {step_label}: {e}")
            if "Could not find model" in str(e) or "Invalid API key" in str(e) or "authentication" in str(e).lower():
                return None
            current_attempt += 1
            wait_time = RETRY_DELAY * (2 ** current_attempt)
            if current_attempt >= RETRY_ATTEMPTS:
                return None
            await asyncio.sleep(wait_time)
    return None


async def predict_premises(problem, steps, step_index, row_id):
    """Predict which previous steps are premises for the current step."""
    solution_text = "\n".join([f"Step {j}: {steps[j]}" for j in range(step_index)])
    step_text = steps[step_index]
    
    user_msg = PREMISE_USR_TEMPLATE.format(
        question=problem,
        solution=solution_text if solution_text else "No previous steps.",
        step=step_text,
    )
    
    messages = []
    if PREMISE_SYS_PROMPT:
        messages.append({"role": "system", "content": PREMISE_SYS_PROMPT})
    messages.append({"role": "user", "content": user_msg})
    
    raw = await call_llm(messages, row_id, f"Step{step_index}_PREMISE", "premise")
    return parse_premises_output(raw, step_index)


async def standalone_check(step_text, row_id, step_index):
    """Check if the step contains a standalone mathematical error."""
    user_msg = STANDALONE_USR_TEMPLATE.format(step=step_text)
    
    messages = [
        {"role": "system", "content": STANDALONE_SYS_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    
    raw = await call_llm(messages, row_id, f"Step{step_index}_STANDALONE", "standalone")
    return parse_standalone_verdict(raw)


async def contextual_check(problem, steps, step_index, premises, row_id):
    """Check if the step is logically consistent given its premises."""
    step_text = steps[step_index]
    
    # Build premises text from predicted premise indices
    if premises:
        premises_text = "\n".join([f"Step {p}: {steps[p]}" for p in premises])
    else:
        premises_text = "No premises identified."
    
    user_msg = CONTEXTUAL_USR_TEMPLATE.format(
        question=problem,
        premises=premises_text,
        step=step_text,
    )
    
    messages = [
        {"role": "system", "content": CONTEXTUAL_SYS_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    
    raw = await call_llm(messages, row_id, f"Step{step_index}_CONTEXTUAL", "contextual")
    return parse_contextual_verdict(raw)


# --- Row Processing ---

async def verify_row_parc(row_data, semaphore):
    """
    PARC pipeline for a single row:
    1. For each step: predict premises
    2. Standalone error check
    3. Contextual error check (using predicted premises)
    4. Fuse: step is correct only if BOTH checks pass
    """
    async with semaphore:
        row_id = row_data.get('id', 'N/A')
        problem = str(row_data.get('problem', ''))
        model_response = str(row_data.get('model_response', ''))

        try:
            hash(row_id)
        except TypeError:
            row_id = str(row_id)

        if not problem or not model_response:
            return row_id, [], "ERROR_SETUP", -1

        steps = [s.strip() for s in model_response.strip().split('\n\n') if s.strip()]
        if not steps:
            return row_id, [], "CORRECT", -1

        step_results = []
        error_step_index = -1

        for i, step_text in enumerate(steps):
            # Step 1: Predict premises
            premises = await predict_premises(problem, steps, i, row_id)

            # Step 2: Standalone error check
            standalone_verdict = await standalone_check(step_text, row_id, i)

            # Step 3: Contextual error check
            contextual_verdict = await contextual_check(problem, steps, i, premises, row_id)

            # Step 4: Fusion - correct only if both checks pass
            is_correct = (
                standalone_verdict == "correct"
                and contextual_verdict == "correct"
            )

            step_results.append({
                "step_index": i,
                "step_text": step_text,
                "premises": premises,
                "standalone_verdict": standalone_verdict,
                "contextual_verdict": contextual_verdict,
                "is_correct": is_correct,
            })

            # Stop at first incorrect step
            if not is_correct and error_step_index == -1:
                error_step_index = i

        final_status = "CORRECT" if error_step_index == -1 else "INCORRECT"
        logging.info(f"Finished PARC for row {row_id}. Steps: {len(step_results)}, "
                     f"Status: {final_status}, ErrorIdx: {error_step_index}")
        return row_id, step_results, final_status, error_step_index


async def main():
    logging.info(f"Starting PARC script. Reading CSV: {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        required_cols = ['id', 'problem', 'model_response']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input CSV missing required columns: {missing_cols}")
        if 'id' in df.columns and not df['id'].is_unique:
            logging.warning("IDs in the 'id' column are not unique.")
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
    logging.info(f"Using model: {API_MODEL}")
    logging.info(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [verify_row_parc(row, semaphore) for row in df.to_dict('records')]

    results_data = {}
    logging.info("Starting PARC verification (premise prediction + standalone + contextual)...")
    try:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="PARC Verifying"):
            try:
                row_id, step_results, final_status, error_step_idx = await future
                results_data[row_id] = {
                    'details': step_results,
                    'status': final_status,
                    'error_step_idx': error_step_idx
                }
            except Exception as e:
                logging.error(f"Error awaiting row result: {e}")
    except Exception as e:
        logging.error(f"Error in main task loop: {e}")

    logging.info("PARC verification finished.")
    logging.info(f"Results collected for {len(results_data)} rows out of {len(df)} attempted.")

    if results_data:
        status_counts = pd.Series([data['status'] for data in results_data.values()]).value_counts()
        logging.info(f"Final row statuses:\n{status_counts.to_string()}")
    else:
        logging.warning("No results collected.")

    default_details = [{"error": "Processing did not yield result"}]
    default_status = "ERROR_PROCESSING"
    default_error_idx = -999

    df['parc_verification_results'] = df['id'].map(
        lambda x: json.dumps(results_data.get(x, {}).get('details', default_details))
    )
    df['parc_final_status'] = df['id'].map(
        lambda x: results_data.get(x, {}).get('status', default_status)
    )
    df['error_step_index'] = df['id'].map(
        lambda x: results_data.get(x, {}).get('error_step_idx', default_error_idx)
    )

    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        logging.info(f"PARC results saved to {OUTPUT_CSV_PATH}")
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