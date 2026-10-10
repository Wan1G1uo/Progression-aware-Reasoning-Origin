import argparse
import ast
import asyncio
import json
import os
import re

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError
from tqdm.asyncio import tqdm

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
MAX_CONCURRENT_REQUESTS = 8
MAX_ACTIVE_ROWS = 8
MAX_API_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 5
TOP_K = 5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=120)

request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
row_semaphore = asyncio.Semaphore(MAX_ACTIVE_ROWS)


def build_prompt(problem, context_text, target_text, verification_type):
    if verification_type == "forward":
        return f"""The following is a math problem and preceding correct solution paragraphs:
[Math Problem]
{problem}

[Preceding Solution Paragraphs]
{context_text if context_text else "No preceding steps."}

Please verify the following paragraph:
[Current Paragraph]
{target_text}

Is this paragraph logically correct and computationally sound based on the problem and preceding paragraphs?
Return "incorrect" if the Current Paragraph contains an actual arithmetic mistake or logical error.
Please think step by step, and put your final conclusion (i.e., 'correct' or 'incorrect') in \\boxed{{}}.
"""

    if verification_type == "backward":
        return f"""The following is a math problem:
[Math Problem]
{problem}

Consider the following two consecutive paragraphs from a solution:
[Current Paragraph]
{target_text}

[Next Paragraph]
{context_text if context_text else "No next step available."}

Please perform a BACKWARD verification from the perspective of the [Next Paragraph]:
1. Does the [Current Paragraph] logically entail and provide necessary, uncontradicted premises for the [Next Paragraph]?
2. Critically check if the [Current Paragraph]'s calculation and logic are sound and semantic consistent.
3. If the [Current Paragraph] contains a calculation error that is carried over, or if the semantic logic is flawed, you MUST output INCORRECT.
4. Put your final conclusion (i.e., 'correct' or 'incorrect') in \\boxed{{}}.
"""

    if verification_type == "edge_refine":
        return f"""The following is a math problem:
[Math Problem]
{problem}

We are reviewing a specific step in the solution:
[Step Under Review]
{target_text}

[Context / Next Step]
{context_text}

Please verify the logical connection and calculation correctness between these two steps.
1. Recalculate any arithmetic in the [Step Under Review].
2. Assume the narrative flow might be misleading. Do not trust the transition just because it reads smoothly.
3. If there is ANY calculation error in the Step Under Review, even if the Next Step uses that wrong number consistently, you MUST output INCORRECT.
4. Put your final conclusion (i.e., 'correct' or 'incorrect') in \\boxed{{}}.
"""

    if verification_type == "top5_rank":
        return f"""You are given a math problem and its step-by-step solution.
[Math Problem]
{problem}

[Solution Steps]
{target_text}

Your task is to rank the five steps most likely to contain the first real error.
Rules:
1. Focus on the earliest step whose own arithmetic or logic is likely wrong.
2. Do not rank later propagated mistakes above their likely source step.
3. Return at most five distinct step indices, sorted from most likely to less likely.
4. Output only a Python-style list inside \\boxed{{}}, for example \\boxed{{[3, 4, 2]}}.
"""

    if verification_type == "rewrite_candidate":
        return f"""You are repairing one suspicious step in a math solution.
[Math Problem]
{problem}

[Verified Preceding Steps]
{context_text if context_text else "No preceding steps."}

[Suspicious Step]
{target_text}

Rewrite only the suspicious step so that it becomes mathematically and logically correct.
Requirements:
1. Keep the rewrite as a single paragraph.
2. Recompute any numbers yourself.
3. Do not mention that you are rewriting or correcting.
4. Output the rewritten paragraph only between <rewrite> and </rewrite>.
"""

    if verification_type == "repair_backward":
        return f"""The following is a math problem with a rewritten candidate step and the original next step.
[Math Problem]
{problem}

[Rewritten Candidate Step]
{target_text}

[Original Next Step]
{context_text if context_text else "No next step available."}

Judge whether the rewritten candidate step itself is a valid correction.
Important:
1. Recompute the rewritten candidate step yourself.
2. If the rewritten candidate step is mathematically correct, do NOT mark it incorrect merely because the original next step may still reflect the old wrong value.
3. Output INCORRECT only if the rewritten candidate step itself remains wrong or logically unsound as a corrected version.
4. Put your final conclusion (i.e., 'correct' or 'incorrect') in \\boxed{{}}.
"""

    if verification_type == "source_rerank":
        return f"""You are localizing the first erroneous step in a math solution.
[Math Problem]
{problem}

[Full Solution Steps]
{context_text}

[Candidate Analysis]
{target_text}

Your task is to choose the single step index that is most likely the FIRST original erroneous step.
Important rules:
1. Prefer the earliest source error, not a later step that only inherits a bad value.
2. A candidate whose rewritten version becomes correct is useful evidence, but does NOT automatically mean that candidate is the first error.
3. Use the rewrite-verification results only as auxiliary evidence for source localization.
4. Output exactly one integer index inside \\boxed{{}}, for example \\boxed{{3}}.
"""

    raise ValueError(f"Unknown verification type: {verification_type}")


def extract_conclusion(raw_response):
    if not raw_response:
        return "ERROR"

    matches = list(re.finditer(r"\\boxed\{(.*?)\}", raw_response, re.IGNORECASE | re.DOTALL))
    for match in reversed(matches):
        content = match.group(1).strip().upper()
        if "INCORRECT" in content:
            return "INCORRECT"
        if "CORRECT" in content:
            return "CORRECT"

    upper = raw_response.upper()
    if "INCORRECT" in upper:
        return "INCORRECT"
    if "CORRECT" in upper:
        return "CORRECT"
    return "UNCERTAIN"


def extract_ranked_indices(raw_response, max_index):
    if not raw_response:
        return []

    candidates = []
    boxed_matches = list(re.finditer(r"\\boxed\{(.*?)\}", raw_response, re.IGNORECASE | re.DOTALL))
    search_spaces = [match.group(1) for match in boxed_matches]
    search_spaces.append(raw_response)

    for text in search_spaces:
        list_match = re.search(r"\[(.*?)\]", text, re.DOTALL)
        if not list_match:
            continue
        try:
            parsed = ast.literal_eval("[" + list_match.group(1) + "]")
            if isinstance(parsed, list):
                candidates = parsed
                break
        except Exception:
            continue

    if not candidates:
        candidates = [int(num) for num in re.findall(r"-?\d+", raw_response)]

    cleaned = []
    for item in candidates:
        try:
            idx = int(item)
        except Exception:
            continue
        if 0 <= idx <= max_index and idx not in cleaned:
            cleaned.append(idx)
        if len(cleaned) >= TOP_K:
            break
    return cleaned


def extract_rewrite_text(raw_response):
    if not raw_response:
        return ""

    match = re.search(r"<rewrite>\s*(.*?)\s*</rewrite>", raw_response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    boxed_matches = list(re.finditer(r"\\boxed\{(.*?)\}", raw_response, re.IGNORECASE | re.DOTALL))
    if boxed_matches:
        return boxed_matches[-1].group(1).strip()

    return raw_response.strip()


def extract_single_index(raw_response, max_index):
    if not raw_response:
        return -1

    boxed_matches = list(re.finditer(r"\\boxed\{(.*?)\}", raw_response, re.IGNORECASE | re.DOTALL))
    search_spaces = [match.group(1) for match in boxed_matches]
    search_spaces.append(raw_response)

    for text in search_spaces:
        nums = re.findall(r"-?\d+", text)
        for num in nums:
            try:
                idx = int(num)
            except Exception:
                continue
            if 0 <= idx <= max_index:
                return idx
    return -1


async def call_llm(problem, context_text, target_text, row_id, step_label, verification_type, max_retries=MAX_API_RETRIES):
    messages = [
        {
            "role": "system",
            "content": "You are a strict mathematical reasoning assistant. Follow the requested output format exactly.",
        },
        {"role": "user", "content": build_prompt(problem, context_text, target_text, verification_type)},
    ]

    for attempt in range(max_retries):
        try:
            async with request_semaphore:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                )
                raw_response = response.choices[0].message.content
                return {
                    "conclusion": extract_conclusion(raw_response),
                    "raw_response": raw_response,
                    "row_id": row_id,
                    "step_label": step_label,
                }
        except RateLimitError as e:
            pass
        except (APIError, APIConnectionError) as e:
            pass
        except Exception as e:
            pass

        if attempt < max_retries - 1:
            wait_seconds = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            print(
                f"Retrying row={row_id} step={step_label} type={verification_type} "
                f"in {wait_seconds}s ({attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(wait_seconds)

    return {"conclusion": "ERROR", "raw_response": "retry_exhausted"}


def fuse_verdicts(fwd_conclusion, bwd_conclusion):
    fwd_clean = fwd_conclusion.strip() if fwd_conclusion else "NONE"
    bwd_clean = bwd_conclusion.strip() if bwd_conclusion else "NONE"

    if "ERROR" in fwd_clean or "ERROR" in bwd_clean:
        return "UNCERTAIN"
    if fwd_clean == "INCORRECT":
        return "INCORRECT"
    if fwd_clean == "CORRECT" and bwd_clean == "CORRECT":
        return "CORRECT"
    if fwd_clean == "CORRECT" and bwd_clean == "INCORRECT":
        return "SUSPICIOUS"
    return "UNCERTAIN"


def parse_steps(row, row_id):
    if "steps" in row and pd.notna(row["steps"]):
        try:
            steps = ast.literal_eval(row["steps"])
            return [str(step).strip() for step in steps if str(step).strip()]
        except Exception as exc:
            print(f"Error parsing steps for row {row_id}: {exc}")

    model_response = str(row.get("model_response", "") or "").strip()
    if model_response:
        return [step.strip() for step in model_response.split("\n\n") if step.strip()]
    return []


def normalize_index(value):
    if pd.isna(value):
        return -1
    try:
        return int(value)
    except Exception:
        return -1


def format_steps_for_ranking(steps):
    return "\n".join([f"Step {idx}: {step}" for idx, step in enumerate(steps)])


def format_candidate_analysis(candidate_details):
    blocks = []
    for detail in candidate_details:
        blocks.append(
            "\n".join(
                [
                    f"Candidate Index: {detail['candidate_index']}",
                    f"Candidate Rank: {detail['candidate_rank']}",
                    f"Original Step: {detail['original_step']}",
                    f"Rewritten Step: {detail['rewritten_step']}",
                    f"Rewrite Success: {detail['rewrite_success']}",
                    f"Rewrite Forward Conclusion: {detail['rewrite_forward_conclusion']}",
                    f"Rewrite Backward Conclusion: {detail['rewrite_backward_conclusion']}",
                    f"Rewrite Fused Verdict: {detail['rewrite_fused_verdict']}",
                    f"Rewrite Edge Conclusion: {detail['rewrite_edge_conclusion']}",
                ]
            )
        )
    return "\n\n".join(blocks)


async def run_pro_detector(problem, steps, row_id):
    for i, current_step_text in enumerate(steps):
        preceding_text = "\n".join(steps[:i])
        fwd_data = await call_llm(problem, preceding_text, current_step_text, row_id, f"PRO_Step{i}_FWD", "forward")
        fwd_conclusion = fwd_data.get("conclusion", "ERROR")

        bwd_conclusion = "CORRECT"
        if i < len(steps) - 1:
            next_step_text = steps[i + 1]
            bwd_data = await call_llm(problem, next_step_text, current_step_text, row_id, f"PRO_Step{i}_BWD", "backward")
            bwd_conclusion = bwd_data.get("conclusion", "ERROR")

        fusion_result = fuse_verdicts(fwd_conclusion, bwd_conclusion)
        if fusion_result == "INCORRECT":
            return i, "PRO_HARD_FLAG"
        if fusion_result == "UNCERTAIN":
            return -2, "PRO_UNCERTAIN"
        if fusion_result == "SUSPICIOUS" and i < len(steps) - 1:
            edge_data = await call_llm(problem, steps[i + 1], current_step_text, row_id, f"PRO_Step{i}_EDGE", "edge_refine")
            edge_conclusion = edge_data.get("conclusion")
            if edge_conclusion == "INCORRECT":
                return i, "PRO_EDGE_FLAG"
            if edge_conclusion == "UNCERTAIN" or "ERROR" in str(edge_conclusion):
                return -2, "PRO_UNCERTAIN"

    return -1, "PRO_CLEAR"


async def rank_top5_candidates(problem, steps, row_id):
    ranking_data = await call_llm(
        problem,
        "",
        format_steps_for_ranking(steps),
        row_id,
        "TOP5_RANK",
        "top5_rank",
    )
    indices = extract_ranked_indices(ranking_data.get("raw_response", ""), len(steps) - 1)
    return indices, ranking_data.get("raw_response", "")


async def rewrite_candidate(problem, steps, row_id, candidate_index):
    preceding_text = "\n".join(steps[:candidate_index])
    rewrite_data = await call_llm(
        problem,
        preceding_text,
        steps[candidate_index],
        row_id,
        f"REWRITE_Step{candidate_index}",
        "rewrite_candidate",
    )
    rewritten_step = extract_rewrite_text(rewrite_data.get("raw_response", ""))
    return rewritten_step, rewrite_data.get("raw_response", "")


async def verify_rewrite(problem, steps, row_id, candidate_index, rewritten_step):
    preceding_text = "\n".join(steps[:candidate_index])
    next_step_text = steps[candidate_index + 1] if candidate_index + 1 < len(steps) else ""

    fwd_data = await call_llm(
        problem,
        preceding_text,
        rewritten_step,
        row_id,
        f"REPAIR_FWD_Step{candidate_index}",
        "forward",
    )
    fwd_conclusion = fwd_data.get("conclusion", "ERROR")

    if next_step_text:
        bwd_data = await call_llm(
            problem,
            next_step_text,
            rewritten_step,
            row_id,
            f"REPAIR_BWD_Step{candidate_index}",
            "repair_backward",
        )
        bwd_conclusion = bwd_data.get("conclusion", "ERROR")
        bwd_raw = bwd_data.get("raw_response", "")
    else:
        bwd_conclusion = "CORRECT"
        bwd_raw = "LAST_STEP_ASSUME_CORRECT"

    fused = fuse_verdicts(fwd_conclusion, bwd_conclusion)
    edge_conclusion = ""
    success = fused == "CORRECT"

    if fused == "SUSPICIOUS" and next_step_text:
        edge_data = await call_llm(
            problem,
            next_step_text,
            rewritten_step,
            row_id,
            f"REPAIR_EDGE_Step{candidate_index}",
            "edge_refine",
        )
        edge_conclusion = edge_data.get("conclusion", "ERROR")
        success = edge_conclusion == "CORRECT"

    return {
        "rewrite_forward_conclusion": fwd_conclusion,
        "rewrite_backward_conclusion": bwd_conclusion,
        "rewrite_backward_raw": bwd_raw,
        "rewrite_fused_verdict": fused,
        "rewrite_edge_conclusion": edge_conclusion,
        "rewrite_success": int(success),
    }


async def rerank_first_error(problem, steps, row_id, candidate_details):
    rerank_data = await call_llm(
        problem,
        format_steps_for_ranking(steps),
        format_candidate_analysis(candidate_details),
        row_id,
        "SOURCE_RERANK",
        "source_rerank",
    )
    selected_index = extract_single_index(rerank_data.get("raw_response", ""), len(steps) - 1)
    return selected_index, rerank_data.get("raw_response", "")


async def process_row(row, row_id):
    async with row_semaphore:
        problem = str(row.get("problem", ""))
        steps = parse_steps(row, row_id)
        if not problem or not steps:
            return {
                "row_id": row_id,
                "final_index": -1,
                "status": "NO_STEPS",
                "pro_detector_index": -1,
                "pro_detector_status": "NO_STEPS",
                "top5_indices": [],
                "top5_raw": "",
                "selected_index": -1,
                "selected_rank": -1,
                "selected_rewrite": "",
                "rerank_raw": "",
                "details": [],
            }

        pro_index, pro_status = await run_pro_detector(problem, steps, row_id)

        if pro_status == "PRO_UNCERTAIN":
            return {
                "row_id": row_id,
                "final_index": -1,
                "status": "PRO_DETECTOR_UNCERTAIN",
                "pro_detector_index": -1,
                "pro_detector_status": pro_status,
                "top5_indices": [],
                "top5_raw": "",
                "selected_index": -1,
                "selected_rank": -1,
                "selected_rewrite": "",
                "rerank_raw": "",
                "details": [],
            }
        if pro_index == -1:
            return {
                "row_id": row_id,
                "final_index": -1,
                "status": "PRO_CLEARED",
                "pro_detector_index": pro_index,
                "pro_detector_status": pro_status,
                "top5_indices": [],
                "top5_raw": "",
                "selected_index": -1,
                "selected_rank": -1,
                "selected_rewrite": "",
                "rerank_raw": "",
                "details": [],
            }

        top5_indices, top5_raw = await rank_top5_candidates(problem, steps, row_id)
        if pro_index not in top5_indices:
            top5_indices = [pro_index] + top5_indices
        top5_indices = top5_indices[:TOP_K]

        candidate_details = []
        for rank, candidate_index in enumerate(top5_indices, start=1):
            rewritten_step, rewrite_raw = await rewrite_candidate(problem, steps, row_id, candidate_index)
            verification_result = await verify_rewrite(problem, steps, row_id, candidate_index, rewritten_step)

            detail = {
                "candidate_rank": rank,
                "candidate_index": candidate_index,
                "original_step": steps[candidate_index],
                "rewritten_step": rewritten_step,
                "rewrite_raw": rewrite_raw,
            }
            detail.update(verification_result)
            candidate_details.append(detail)

        rerank_index, rerank_raw = await rerank_first_error(problem, steps, row_id, candidate_details)
        rerank_candidates = {detail["candidate_index"]: detail for detail in candidate_details}

        if rerank_index in rerank_candidates:
            selected_detail = rerank_candidates[rerank_index]
            selected_rank = selected_detail["candidate_rank"]
            selected_index = selected_detail["candidate_index"]
            selected_rewrite = selected_detail["rewritten_step"]
            final_index = selected_index
            status = "TOP5_RERANK_LOCALIZED"
        else:
            selected_rank = -1
            selected_index = -1
            selected_rewrite = ""
            rerank_raw = rerank_raw or "INVALID_RERANK_OUTPUT"
            final_index = pro_index
            status = "TOP5_RERANK_FALLBACK_TO_PRO"

        return {
            "row_id": row_id,
            "final_index": final_index,
            "status": status,
            "pro_detector_index": pro_index,
            "pro_detector_status": pro_status,
            "top5_indices": top5_indices,
            "top5_raw": top5_raw,
            "selected_index": selected_index,
            "selected_rank": selected_rank,
            "selected_rewrite": selected_rewrite,
            "rerank_raw": rerank_raw,
            "details": candidate_details,
        }


async def main():
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(BASE_DIR, "gsm8k.csv"))
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "processbench_pro.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.iloc[: args.limit]
    df = df.reset_index(drop=True)

    tasks = [process_row(row, index) for index, row in df.iterrows()]
    results = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="PRO"):
        results.append(await future)

    results.sort(key=lambda item: item["row_id"])
    df["first_error_paragraph_index_top5_rewrite"] = [res["final_index"] for res in results]
    df["top5_rewrite_status"] = [res["status"] for res in results]
    df["pro_detector_index"] = [res["pro_detector_index"] for res in results]
    df["pro_detector_status"] = [res["pro_detector_status"] for res in results]
    df["top5_candidate_indices"] = [json.dumps(res["top5_indices"], ensure_ascii=True) for res in results]
    df["top5_rank_raw_response"] = [res["top5_raw"] for res in results]
    df["selected_candidate_index"] = [res["selected_index"] for res in results]
    df["selected_candidate_rank"] = [res["selected_rank"] for res in results]
    df["selected_rewrite_text"] = [res["selected_rewrite"] for res in results]
    df["top5_rerank_raw_response"] = [res["rerank_raw"] for res in results]
    df["top5_rewrite_details"] = [json.dumps(res["details"], ensure_ascii=True) for res in results]

    df.to_csv(args.output, index=False)
    print(f"PRO localization complete. Saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())