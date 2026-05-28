# Evaluation/golden_tests.py
# Real Q&A pairs extracted from Infoservices Employee Handbook (Ver 7)
# Run this after any code change to validate the full RAG pipeline.

from Evaluation.Evaluator import evaluate_response

# ──────────────────────────────────────────
# GOLDEN DATASET
# Every answer is directly from the handbook.
# ──────────────────────────────────────────

GOLDEN_TESTS = [
    {
        "id":      "GT-01",
        "topic":   "Notice Period",
        "query":   "What is the notice period for resignation?",
        "answer":  "One month. Employees are expected to serve a notice period "
                   "of one month excluding Weekly Offs and National Holidays.",
        "must_contain": ["one month"],
        "must_not_contain": ["two months", "three months", "two weeks"],
        "context": "Once an employee resigns, he/she is expected to serve a notice "
                   "period of one month excluding Weekly Offs and National Holidays. "
                   "The resignation can be accepted with immediate effect or within "
                   "any number of days up to one month as per the sole discretion of "
                   "the Management Committee."
    },
    {
        "id":      "GT-02",
        "topic":   "Vacation Leave",
        "query":   "How many vacation leaves do employees get per year?",
        "answer":  "Employees get 6 days of Vacation Leave (casual leave) per calendar year.",
        "must_contain": ["6"],
        "must_not_contain": ["12", "18", "10", "15"],
        "context": "An employee may be granted Vacation Leaves of not more than 6 days "
                   "in a calendar year. Vacation leave is nothing but Casual Leave (CL). "
                   "Unspent Vacation Leaves shall not be carried forward to the next calendar year."
    },
    {
        "id":      "GT-03",
        "topic":   "Sick Leave",
        "query":   "How many sick leaves are employees entitled to?",
        "answer":  "Employees are entitled to 6 days of Sick Leave per calendar year.",
        "must_contain": ["6"],
        "must_not_contain": ["12", "10", "15"],
        "context": "An employee may be granted Sick Leaves of not more than 6 days in a "
                   "calendar year. Sick Leave can be taken for half a day also. "
                   "Unspent Sick Leaves shall not be carried forward to the next calendar year."
    },
    {
        "id":      "GT-04",
        "topic":   "Earned Leave",
        "query":   "How many earned leaves does an employee get?",
        "answer":  "Employees get 6 Earned Leave days per calendar year, credited at 1 day "
                   "per bi-monthly. Maximum accumulation is 20 days.",
        "must_contain": ["6"],
        "must_not_contain": ["12", "18"],
        "context": "Every Full Time Permanent employee is entitled for 6 days Earned Leaves "
                   "during a calendar year. An employee is entitled to earn 1 leave on a "
                   "bi-monthly basis. The Earned Leave credits carried forward plus the current "
                   "credit does not exceed the maximum limit of 20 days."
    },
    {
        "id":      "GT-05",
        "topic":   "Probation Period",
        "query":   "What is the probation period for new employees?",
        "answer": "All trainees joining the organization have to serve a probation period "
          "of three months unless otherwise stated in the Joining Letter.",
        "must_contain": ["three months"],
        "must_not_contain": ["six months", "one month", "two months"],
        "context": "All trainees joining the organization have to serve a probation period "
                   "of three months unless otherwise stated in the Joining Letter. "
                   "A newly employed person will be on a probation period of three months, "
                   "unless otherwise stated in the Offer Letter."
    },
    {
        "id":      "GT-06",
        "topic":   "Salary Payment",
        "query":   "When is monthly salary paid to employees?",
        "answer":  "Monthly salary is paid on the first of every month.",
        "must_contain": ["first"],
        "must_not_contain": ["last day", "15th", "5th"],
        "context": "Salary is calculated from the first day to the last day of every month. "
                   "Monthly salary is paid to employees on the first of every month."
    },
    {
        "id":      "GT-07",
        "topic":   "Working Hours",
        "query":   "How many working hours per day and per week?",
        "answer":  "8 working hours per day with an additional hour for lunch break. "
                   "Maximum 40 working hours per week. 5 days a week.",
        "must_contain": ["8", "40"],
        "must_not_contain": ["9 hours", "45 hours", "50 hours"],
        "context": "Typically, all work locations have 8 working hours per day with an "
                   "additional hour for lunch break. For any full-time employee, the maximum "
                   "number of working hours per week is 40. We have the same work week of "
                   "5 days in a week."
    },
    {
        "id":      "GT-08",
        "topic":   "Appraisal Cycle",
        "query":   "When is the performance appraisal and salary hike?",
        "answer":  "The hike review period is in the month of April every year. "
                   "Promotions and salary increments take effect from April irrespective "
                   "of date of joining.",
        "must_contain": ["april", "April"],
        "must_not_contain": ["january", "october", "december"],
        "context": "The hike review period for all employees will be in the month of APRIL "
                   "irrespective of their Date Of Joining (DOJ). Promotions and salary "
                   "increments will take effect starting from the April month each year "
                   "irrespective of employee's work anniversary."
    },
    {
        "id":      "GT-09",
        "topic":   "Work From Home",
        "query":   "Who is eligible for Work From Home?",
        "answer":  "Only Full-Time Permanent Employees who have served the organization "
                   "for more than 6 months are entitled to apply for Work From Home.",
        "must_contain": ["6 months"],
        "must_not_contain": ["3 months", "1 year", "immediately"],
        "context": "Only Full-Time Permanent Employees who have served the organization "
                   "for more than 6 months are entitled to apply for Work From Home facility. "
                   "The approval for the same is purely at the discretion of the Management Committee."
    },
    {
        "id":      "GT-10",
        "topic":   "PF Deduction",
        "query":   "How much is deducted for Provident Fund?",
        "answer":  "12% of Basic salary is deducted as Provident Fund per month. "
                   "If basic salary is Rs. 15,000 or more, PF is fixed at Rs. 1800.",
        "must_contain": ["12%", "1800"],
        "must_not_contain": ["10%", "15%", "2000"],
        "context": "12% of Basic salary is deducted as Provident Fund per month. However, "
                   "if basic salary is greater than or equal to Rs. 15000 per month, then "
                   "PF amount deducted would be fixed at Rs. 1800."
    },
    {
        "id":      "GT-11",
        "topic":   "Foreign Tour Per Diem",
        "query":   "What is the daily allowance for foreign tours?",
        "answer":  "The company pays $100 (100 Dollars) per diem for foreign tours "
                   "to cover daily necessities such as food and travel to and from office.",
        "must_contain": ["100", "$100"],
        "must_not_contain": ["$50", "$200", "50 dollars"],
        "context": "The company will pay $100 (100 Dollars) per diem which is an allowance "
                   "given on daily basis for your daily basis necessities such as for food, "
                   "travel to and fro office."
    },
    {
        "id":      "GT-12",
        "topic":   "General Shift Timings",
        "query":   "What are the general shift timings?",
        "answer":  "General shift is from 9 AM to 6 PM for Admin, Recruiting and "
                   "other Operations departments.",
        "must_contain": ["9", "6"],
        "must_not_contain": ["10 am", "8 am", "7 pm"],
        "context": "General Shift: General shift is from 9 am to 6 pm for Admin, "
                   "Recruiting and other Operations departments."
    },
]


# ──────────────────────────────────────────
# TEST RUNNER
# ──────────────────────────────────────────

def run_golden_tests(verbose: bool = True) -> dict:
    """
    Runs all golden tests against the evaluator.
    Returns summary of pass/fail results.
    """
    print("\n══════════════════════════════════════")
    print("  GOLDEN TEST SUITE — Infoservices RAG")
    print("══════════════════════════════════════\n")

    passed       = 0
    failed       = 0
    results      = []
    total_quality = 0.0

    for test in GOLDEN_TESTS:
        result = evaluate_response(
            query    = test["query"],
            response = test["answer"],
            context  = test["context"]
        )

        quality = result["quality_score"]
        total_quality += quality

        # Content checks
        answer_lower  = test["answer"].lower()
        content_pass  = True
        content_notes = []

        for phrase in test.get("must_contain", []):
            if phrase.lower() not in answer_lower:
                content_pass = False
                content_notes.append(f"MISSING: '{phrase}'")

        for phrase in test.get("must_not_contain", []):
            if phrase.lower() in answer_lower:
                content_pass = False
                content_notes.append(f"SHOULD NOT CONTAIN: '{phrase}'")

        # Pass = quality > 0.70 AND content checks pass
        overall_pass = quality >= 0.70 and content_pass

        if overall_pass:
            passed += 1
            icon = "✅"
        else:
            failed += 1
            icon = "❌"

        results.append({
            "id":      test["id"],
            "topic":   test["topic"],
            "passed":  overall_pass,
            "quality": quality,
            "hallucinations": result.get("hallucination_flags", []),
            "notes":   content_notes
        })

        if verbose:
            print(f"{icon} [{test['id']}] {test['topic']}")
            print(f"   Query:          {test['query']}")
            print(f"   Quality Score:  {quality}")
            print(f"   Hallucinations: {result.get('hallucination_flags', [])}")
            print(f"   Reasoning:      {result.get('reasoning', '')}")
            if content_notes:
                for note in content_notes:
                    print(f"   ⚠️  {note}")
            print()

    avg_quality = total_quality / len(GOLDEN_TESTS) if GOLDEN_TESTS else 0

    print("══════════════════════════════════════")
    print(f"  RESULTS: {passed} passed / {failed} failed")
    print(f"  AVG QUALITY SCORE: {avg_quality:.3f}")
    print("══════════════════════════════════════\n")

    return {
        "passed":      passed,
        "failed":      failed,
        "total":       len(GOLDEN_TESTS),
        "avg_quality": round(avg_quality, 4),
        "results":     results
    }


if __name__ == "__main__":
    run_golden_tests(verbose=True)
