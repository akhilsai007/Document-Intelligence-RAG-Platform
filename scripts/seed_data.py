"""Generate a small labeled sample corpus across categories so the platform
can be ingested, the router trained, and the API demonstrated end-to-end.
Four documents per category gives the XGBoost router enough signal to learn
real class boundaries (and to support a stratified train/test split)."""
from __future__ import annotations

import json
import os

SAMPLE = {
    # ---- HR ----
    "hr_pto_policy": ("hr",
        "Paid time off policy. Full-time employees accrue 20 days of paid "
        "vacation per year, accruing monthly. Unused vacation of up to 5 days "
        "may carry over into the next calendar year. Sick leave is separate and "
        "provides 10 days annually. Submit vacation requests in the HR portal at "
        "least two weeks in advance."),
    "hr_remote_work": ("hr",
        "Remote work guidelines. Employees may work remotely up to three days "
        "per week with manager approval. Core collaboration hours are 10am to "
        "3pm local time. A home office stipend of 500 dollars is available once "
        "per year for equipment and furniture."),
    "hr_benefits": ("hr",
        "Employee benefits overview. The company offers health, dental, and "
        "vision insurance, a 401k retirement plan with employer matching up to "
        "four percent, and twelve weeks of paid parental leave. Open enrollment "
        "for benefits happens every November."),
    "hr_onboarding": ("hr",
        "New hire onboarding. On day one, new employees complete payroll and "
        "tax paperwork in the HR portal, receive their laptop, and meet their "
        "manager and onboarding buddy. Mandatory compliance training must be "
        "finished within the first two weeks of employment."),

    # ---- Finance ----
    "finance_expense": ("finance",
        "Expense reimbursement procedure. Submit receipts within 30 days of "
        "purchase. Meals during travel are reimbursed up to 75 dollars per day. "
        "Airfare should be booked in economy class. Reimbursements are paid in "
        "the next payroll cycle after your cost center owner approves."),
    "finance_quarterly": ("finance",
        "Quarterly close process. Revenue is recognized when goods or services "
        "are delivered. The finance team locks the ledger five business days "
        "after quarter end. Variance analysis compares actuals against the "
        "approved budget for each department."),
    "finance_procurement": ("finance",
        "Procurement and purchase orders. Any purchase above 5000 dollars "
        "requires a purchase order approved by finance before the vendor is "
        "engaged. Preferred vendors offer negotiated pricing. Invoices are paid "
        "on net 30 terms unless otherwise agreed in the contract."),
    "finance_budget": ("finance",
        "Annual budgeting cycle. Department heads submit budget proposals in "
        "September for the following fiscal year. Finance consolidates the "
        "forecasts, reviews headcount and capital expenditure, and presents the "
        "consolidated budget to the executive team for approval in November."),

    # ---- Engineering ----
    "engineering_oncall": ("engineering",
        "On-call runbook. The primary on-call engineer acknowledges pages within "
        "five minutes. Severity one incidents require a status update every 30 "
        "minutes. Roll back the most recent deployment if error rates exceed two "
        "percent. Post-incident reviews are blameless and due within 48 hours."),
    "engineering_deploy": ("engineering",
        "Deployment pipeline. All changes go through pull request review and "
        "continuous integration. Images are built with Docker and deployed to "
        "Kubernetes via a rolling update. Canary traffic starts at five percent "
        "and increases if latency and error metrics stay healthy."),
    "engineering_code_review": ("engineering",
        "Code review standards. Every pull request needs at least one approving "
        "review before merge. Reviewers check for tests, readability, and "
        "security issues. Large changes should be split into smaller pull "
        "requests. The continuous integration suite must pass before merging."),
    "engineering_architecture": ("engineering",
        "Service architecture overview. Microservices communicate over REST and "
        "an event bus. Each service owns its database and exposes health and "
        "metrics endpoints. Infrastructure is provisioned with Terraform and "
        "runs on a managed Kubernetes cluster in the cloud."),

    # ---- Legal ----
    "legal_nda": ("legal",
        "Non-disclosure agreement summary. Confidential information must not be "
        "shared with third parties without written consent. The confidentiality "
        "obligation survives termination of the agreement for three years. "
        "Breach may result in injunctive relief and monetary damages."),
    "legal_data_retention": ("legal",
        "Data retention standard. Customer personal data is retained only as "
        "long as necessary for the stated purpose. Deletion requests are honored "
        "within 30 days. Audit logs are retained for one year and then purged "
        "according to the records schedule."),
    "legal_privacy": ("legal",
        "Privacy policy summary. We collect only the personal data needed to "
        "provide the service. Users may request access to, correction of, or "
        "deletion of their data. Data is not sold to third parties. Processing "
        "complies with applicable privacy regulations."),
    "legal_contracts": ("legal",
        "Contract review guidelines. All vendor and customer contracts are "
        "reviewed by the legal team before signature. Pay attention to liability "
        "caps, indemnification clauses, governing law, and termination rights. "
        "Only authorized signatories may execute a binding agreement."),
}


# Short, query-like labeled phrases for training the router. Training the
# classifier on short text that matches the inference distribution (user
# queries) works far better than training only on long documents.
ROUTER_PHRASES = {
    "hr": [
        "how many vacation days", "paid time off policy", "sick leave allowance",
        "request pto in the portal", "remote work from home policy",
        "home office stipend", "parental leave weeks", "employee benefits and 401k",
        "health and dental insurance", "open enrollment benefits",
        "new hire onboarding", "onboarding buddy and laptop",
        "manager approval to work remotely", "carry over unused vacation",
        "vacation accrual per year", "sick days off work",
    ],
    "finance": [
        "expense reimbursement receipts", "meal limit while traveling",
        "book airfare economy class", "quarterly close ledger",
        "revenue recognition rules", "variance against budget",
        "purchase order approval", "preferred vendor pricing",
        "net 30 invoice terms", "annual budgeting cycle",
        "capital expenditure forecast", "submit budget proposal",
        "cost center owner approval", "reimbursed in next payroll",
        "travel expense receipts", "vendor invoice payment",
    ],
    "engineering": [
        "on call incident response", "severity one status update",
        "roll back deployment error rate", "docker image build pipeline",
        "kubernetes rolling update", "canary traffic percentage",
        "pull request code review", "continuous integration must pass",
        "microservice rest event bus", "terraform infrastructure provisioning",
        "service health metrics endpoint", "blameless post incident review",
        "split large pull request", "deploy to production",
        "rollback on high error rate", "review pull request before merge",
    ],
    "legal": [
        "non disclosure agreement", "confidential information third parties",
        "data retention period", "delete my personal data",
        "privacy policy data collection", "audit log retention",
        "contract review liability cap", "indemnification clause review",
        "governing law and termination", "authorized signatory only",
        "records schedule purge", "request access to my data",
        "breach injunctive relief", "vendor contract signature",
        "customer data deletion request", "confidentiality survives termination",
    ],
}


def main(
    out_dir: str = "data/sample_docs",
    labels_path: str = "data/labels.json",
    router_path: str = "data/router_training.json",
):
    os.makedirs(out_dir, exist_ok=True)
    labels = {}
    for doc_id, (category, text) in SAMPLE.items():
        with open(os.path.join(out_dir, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        labels[doc_id] = category
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    router_examples = [
        {"text": phrase, "category": cat}
        for cat, phrases in ROUTER_PHRASES.items()
        for phrase in phrases
    ]
    with open(router_path, "w", encoding="utf-8") as f:
        json.dump(router_examples, f, indent=2)

    print(f"Wrote {len(SAMPLE)} documents to {out_dir}, labels to {labels_path}, "
          f"and {len(router_examples)} router training phrases to {router_path}")


if __name__ == "__main__":
    main()
