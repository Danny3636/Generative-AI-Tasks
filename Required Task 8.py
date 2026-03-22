# -*- coding: utf-8 -*-
"""
Risk Divergence Analysis: Block (SQ) vs PayPal (PYPL) — 10-K 2023
==================================================================
This script analyzes the "Risk Factors" sections from the 2023 10-K filings
of Block, Inc. and PayPal Holdings, Inc. using multiple LLM prompting
strategies and an LLM-as-Judge evaluation framework.

Base Question: Is Block's "AI & Bitcoin" focus creating undisclosed
operational risks compared to PayPal's conservative approach?

Instructions:
  1. Upload Block_10K_2023.pdf and PayPal_10K_2023.pdf to your Colab runtime.
  2. Get a free Gemini API key from https://aistudio.google.com/app/apikey
  3. Paste the key when prompted (or set GEMINI_API_KEY env variable).
  4. Run all cells.
"""

# ── 0. Install dependencies ─────────────────────────────────────────────────
# !pip install -q google-generativeai pypdf

import os, re, textwrap, json, time
from pypdf import PdfReader

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Run:  pip install -q google-generativeai pypdf\n"
        "then restart the runtime."
    )

# ── 1. API key setup ────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    try:
        from google.colab import userdata
        API_KEY = userdata.get("GEMINI_API_KEY")
    except Exception:
        pass
if not API_KEY:
    API_KEY = input("Enter your Gemini API key: ").strip()

genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-1.5-flash"       # fast + cheap; swap to "gemini-1.5-pro" for deeper analysis
model = genai.GenerativeModel(MODEL_NAME)

print(f"✅ Configured model: {MODEL_NAME}")

# ── 2. PDF text extraction ──────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    print(f"   Extracted {len(text):,} chars from {os.path.basename(pdf_path)}")
    return text


def extract_risk_factors(full_text: str, company: str) -> str:
    """
    Pull the 'Item 1A. Risk Factors' section from a 10-K filing.
    Falls back to a generous window if exact markers aren't found.
    """
    # Try standard SEC markers
    start_patterns = [
        r"ITEM\s+1A[\.\s]+RISK\s+FACTORS",
        r"Item\s+1A[\.\s]+Risk\s+Factors",
    ]
    end_patterns = [
        r"ITEM\s+1B[\.\s]",
        r"Item\s+1B[\.\s]",
        r"ITEM\s+2[\.\s]",
    ]

    start_idx = None
    for pat in start_patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            start_idx = m.start()
            break

    end_idx = None
    if start_idx is not None:
        for pat in end_patterns:
            m = re.search(pat, full_text[start_idx + 100:], re.IGNORECASE)
            if m:
                end_idx = start_idx + 100 + m.start()
                break

    if start_idx is not None:
        section = full_text[start_idx: end_idx] if end_idx else full_text[start_idx: start_idx + 200_000]
    else:
        print(f"   ⚠️  Could not locate Risk Factors for {company}; using first 150k chars")
        section = full_text[:150_000]

    print(f"   {company} Risk Factors: {len(section):,} chars")
    return section

# ── Load the files ───────────────────────────────────────────────────────────
# Update these paths if your files are in a different location
BLOCK_PDF  = "Block_10K_2023.pdf"
PAYPAL_PDF = "PayPal_10K_2023.pdf"

# Check for files in common Colab upload locations
for candidate_dir in [".", "/content", "/content/drive/MyDrive"]:
    bp = os.path.join(candidate_dir, BLOCK_PDF)
    pp = os.path.join(candidate_dir, PAYPAL_PDF)
    if os.path.exists(bp) and os.path.exists(pp):
        BLOCK_PDF, PAYPAL_PDF = bp, pp
        break

print("📄 Extracting text from PDFs...")
block_full   = extract_text_from_pdf(BLOCK_PDF)
paypal_full  = extract_text_from_pdf(PAYPAL_PDF)

block_risks  = extract_risk_factors(block_full,  "Block")
paypal_risks = extract_risk_factors(paypal_full, "PayPal")

# Truncate to fit context windows (keep the most important first ~60k chars)
MAX_CHARS = 60_000
block_risks_trunc  = block_risks[:MAX_CHARS]
paypal_risks_trunc = paypal_risks[:MAX_CHARS]

print("✅ Risk Factors extracted and ready.\n")

# ── 3. Define multiple prompting strategies ──────────────────────────────────
#
# We use three distinct techniques:
#   Prompt A – Direct Comparative (zero-shot)
#   Prompt B – Chain-of-Thought (CoT) with structured reasoning
#   Prompt C – Role-Play Forensic Analyst (persona-based)
#
# Each receives the same data but is designed to surface different insights.

PROMPT_A_TEMPLATE = textwrap.dedent("""\
    You are a financial analyst. Compare the Risk Factors sections of Block, Inc.
    and PayPal from their 2023 10-K filings provided below.

    Focus specifically on:
    1. Risks unique to Block that stem from its Bitcoin/cryptocurrency strategy.
    2. Risks unique to Block from its AI initiatives or ecosystem expansion (Tidal, TBD).
    3. Any language in Block's filing that hints at undisclosed or emerging operational risks
       that PayPal does NOT face.
    4. How PayPal's risk profile reflects a "stable incumbent" posture vs. Block's aggressive
       diversification.

    Provide a structured comparison with specific evidence (quote short phrases where helpful).

    === BLOCK RISK FACTORS (TRUNCATED) ===
    {block_risks}

    === PAYPAL RISK FACTORS (TRUNCATED) ===
    {paypal_risks}
""")

PROMPT_B_TEMPLATE = textwrap.dedent("""\
    You are an SEC filing analyst. I will provide Risk Factors from the 2023 10-K
    filings of Block, Inc. and PayPal Holdings, Inc.

    Think step by step:

    Step 1: List every risk category mentioned by Block that is ABSENT from PayPal's filing.
    Step 2: For each Block-only risk, assess whether it relates to (a) Bitcoin/crypto,
            (b) AI/machine learning, (c) ecosystem expansion (Tidal, TBD, Afterpay), or
            (d) other strategic bets.
    Step 3: Identify any hedging language, vague disclaimers, or new risk disclosures in
            Block's filing that could signal emerging legal, regulatory, or operational
            trouble that is not yet fully transparent.
    Step 4: Rate each unique Block risk on a scale of 1-5 for "potential to become a
            material undisclosed liability" and explain your reasoning.
    Step 5: Summarize whether Block's AI & Bitcoin focus is creating a meaningfully
            different risk profile vs PayPal.

    Be specific. Cite short phrases from the filings.

    === BLOCK RISK FACTORS ===
    {block_risks}

    === PAYPAL RISK FACTORS ===
    {paypal_risks}
""")

PROMPT_C_TEMPLATE = textwrap.dedent("""\
    You are a forensic financial investigator hired by a hedge fund to find hidden
    red flags in Block, Inc.'s 2023 10-K Risk Factors that a casual reader would miss.
    Your benchmark is PayPal's 2023 10-K — a mature, conservative fintech.

    Your investigation framework:
    • LANGUAGE SHIFTS: Find phrases in Block's filing that use unusually vague,
      aspirational, or hedging language around Bitcoin, AI, or new ventures
      (Tidal, TBD/tbDEX). Compare to PayPal's more direct risk language.
    • CONCENTRATION RISK: Does Block's revenue dependency on Bitcoin or Cash App
      create risks PayPal avoids through diversification?
    • REGULATORY GAPS: Are there regulatory risks Block acknowledges that PayPal
      does not — especially around crypto custody, money transmission, or
      decentralized finance?
    • TALENT & EXECUTION: Does Block's filing reveal concerns about integrating
      disparate businesses (music streaming, crypto, BNPL, banking) that PayPal
      avoids?
    • HIDDEN LIABILITIES: Any mention of contingent liabilities, pending
      investigations, or compliance costs unique to Block's strategy?

    Deliver your findings as a structured intelligence brief.

    === BLOCK RISK FACTORS ===
    {block_risks}

    === PAYPAL RISK FACTORS ===
    {paypal_risks}
""")

# ── 4. Run all three prompts ────────────────────────────────────────────────

def call_llm(prompt: str, label: str) -> str:
    """Send a prompt to Gemini and return the response text."""
    print(f"🔄 Running {label}...")
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=4096,
            ),
        )
        result = response.text
        print(f"   ✅ {label} complete — {len(result):,} chars returned")
        return result
    except Exception as e:
        print(f"   ❌ {label} failed: {e}")
        return f"[ERROR] {e}"


results = {}

results["Prompt_A_Direct_Comparative"] = call_llm(
    PROMPT_A_TEMPLATE.format(
        block_risks=block_risks_trunc,
        paypal_risks=paypal_risks_trunc
    ),
    "Prompt A — Direct Comparative"
)
time.sleep(2)  # rate-limit courtesy

results["Prompt_B_Chain_of_Thought"] = call_llm(
    PROMPT_B_TEMPLATE.format(
        block_risks=block_risks_trunc,
        paypal_risks=paypal_risks_trunc
    ),
    "Prompt B — Chain-of-Thought"
)
time.sleep(2)

results["Prompt_C_Forensic_Analyst"] = call_llm(
    PROMPT_C_TEMPLATE.format(
        block_risks=block_risks_trunc,
        paypal_risks=paypal_risks_trunc
    ),
    "Prompt C — Forensic Analyst"
)
time.sleep(2)

# ── 5. Print intermediate results ───────────────────────────────────────────

for name, text in results.items():
    print("\n" + "=" * 80)
    print(f"  {name}")
    print("=" * 80)
    print(text[:3000] + ("\n... [truncated for display]" if len(text) > 3000 else ""))

# ── 6. LLM-as-Judge evaluation ──────────────────────────────────────────────
#
# The judge rates each prompt's output on:
#   • Specificity (1-10): How concrete and evidence-backed are the findings?
#   • Insight depth (1-10): Does it surface non-obvious risks?
#   • Actionability (1-10): Could an investor act on these findings?
#   • Accuracy (1-10): Are claims well-grounded in the actual filings?

JUDGE_PROMPT = textwrap.dedent("""\
    You are an expert evaluator assessing the quality of financial risk analyses.

    BASE QUESTION: "Is Block's AI & Bitcoin focus creating undisclosed operational
    risks compared to PayPal's conservative approach?"

    Below are three analyst responses generated from different prompting strategies,
    all analyzing the same 2023 10-K Risk Factors of Block, Inc. vs PayPal.

    Rate EACH response on four criteria (1–10 scale) and provide a brief justification
    for each score. Then declare which response is BEST overall and explain why.

    Finally, synthesize the strongest insights from ALL three responses into a single
    paragraph identifying the most critical risk divergences.

    CRITERIA:
    1. Specificity (1-10): Concrete evidence and direct references to filing language
    2. Insight Depth (1-10): Non-obvious risks surfaced beyond surface-level comparison
    3. Actionability (1-10): Findings an investor could act on
    4. Accuracy (1-10): Claims well-grounded in actual filing content

    FORMAT your response as:

    ## Evaluation of Response A (Direct Comparative)
    - Specificity: X/10 — [justification]
    - Insight Depth: X/10 — [justification]
    - Actionability: X/10 — [justification]
    - Accuracy: X/10 — [justification]

    ## Evaluation of Response B (Chain-of-Thought)
    - Specificity: X/10 — [justification]
    - Insight Depth: X/10 — [justification]
    - Actionability: X/10 — [justification]
    - Accuracy: X/10 — [justification]

    ## Evaluation of Response C (Forensic Analyst)
    - Specificity: X/10 — [justification]
    - Insight Depth: X/10 — [justification]
    - Actionability: X/10 — [justification]
    - Accuracy: X/10 — [justification]

    ## Best Response: [A/B/C]
    [Explanation]

    ## Synthesized Key Risk Divergences
    [Single paragraph combining the strongest findings from all three]

    === RESPONSE A (Direct Comparative) ===
    {response_a}

    === RESPONSE B (Chain-of-Thought) ===
    {response_b}

    === RESPONSE C (Forensic Analyst) ===
    {response_c}
""")

judge_result = call_llm(
    JUDGE_PROMPT.format(
        response_a=results["Prompt_A_Direct_Comparative"][:8000],
        response_b=results["Prompt_B_Chain_of_Thought"][:8000],
        response_c=results["Prompt_C_Forensic_Analyst"][:8000],
    ),
    "LLM Judge Evaluation"
)

print("\n" + "=" * 80)
print("  LLM JUDGE EVALUATION")
print("=" * 80)
print(judge_result)

# ── 7. Generate final 200-word Executive Summary ────────────────────────────

SUMMARY_PROMPT = textwrap.dedent("""\
    You are a senior financial analyst writing an executive summary.

    Based on the judge evaluation and synthesized findings below, write a FINAL
    200-word Executive Summary answering this question:

    "Is Block's AI & Bitcoin focus creating undisclosed operational risks compared
    to PayPal's conservative approach?"

    Requirements:
    - Exactly ~200 words (hard limit: 180–220 words)
    - Start with a clear thesis statement
    - Reference specific risk categories found in the 10-K filings
    - Compare Block's risk posture to PayPal's
    - End with a forward-looking risk assessment

    JUDGE EVALUATION AND SYNTHESIS:
    {judge_result}
""")

executive_summary = call_llm(
    SUMMARY_PROMPT.format(judge_result=judge_result[:6000]),
    "Final Executive Summary"
)

print("\n" + "=" * 80)
print("  FINAL EXECUTIVE SUMMARY (≈200 words)")
print("=" * 80)
print(executive_summary)

# ── 8. Save all outputs ─────────────────────────────────────────────────────

output = {
    "task": "Risk Divergence Analysis — Block vs PayPal (10-K 2023)",
    "model": MODEL_NAME,
    "prompt_results": results,
    "judge_evaluation": judge_result,
    "executive_summary": executive_summary,
}

OUTPUT_FILE = "risk_divergence_output.json"
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n💾 Full output saved to {OUTPUT_FILE}")

# Also save a clean text report
REPORT_FILE = "risk_divergence_report.txt"
with open(REPORT_FILE, "w") as f:
    f.write("RISK DIVERGENCE ANALYSIS: BLOCK vs PAYPAL (2023 10-K)\n")
    f.write("=" * 60 + "\n\n")

    for name, text in results.items():
        f.write(f"\n{'─' * 60}\n")
        f.write(f"ANALYSIS: {name}\n")
        f.write(f"{'─' * 60}\n")
        f.write(text + "\n")

    f.write(f"\n{'─' * 60}\n")
    f.write("LLM JUDGE EVALUATION\n")
    f.write(f"{'─' * 60}\n")
    f.write(judge_result + "\n")

    f.write(f"\n{'═' * 60}\n")
    f.write("FINAL EXECUTIVE SUMMARY (~200 words)\n")
    f.write(f"{'═' * 60}\n")
    f.write(executive_summary + "\n")

print(f"📝 Report saved to {REPORT_FILE}")
print("\n✅ Analysis complete!")
