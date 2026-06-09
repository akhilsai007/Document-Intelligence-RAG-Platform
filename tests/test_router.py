from src.router.train_xgb import train
from src.router.classifier import RouterClassifier


def test_router_trains_and_predicts(tmp_path):
    examples = [
        ("paid vacation pto sick leave hr portal", "hr"),
        ("remote work stipend manager approval hr", "hr"),
        ("expense reimbursement receipts payroll finance", "finance"),
        ("quarterly close ledger budget variance finance", "finance"),
        ("on call incident rollback deployment engineering", "engineering"),
        ("docker kubernetes canary pipeline engineering", "engineering"),
    ]
    model_path = tmp_path / "router.joblib"
    result = train(examples, model_path=str(model_path))
    assert model_path.exists()
    assert set(result["labels"]) == {"hr", "finance", "engineering"}

    clf = RouterClassifier.load(str(model_path))
    label, conf = clf.predict("how much vacation pto do I get")
    assert label == "hr"
    assert 0.0 <= conf <= 1.0


def test_router_without_model_returns_general():
    clf = RouterClassifier()
    assert clf.predict("anything") == ("general", 0.0)
