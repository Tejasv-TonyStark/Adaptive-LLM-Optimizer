"""Offline regression/integration tests. Provider calls are replaced by fixtures."""
import os
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["JWT_SECRET_KEY"] = "offline-test-secret-that-is-at-least-32-bytes"
os.environ["EVALUATION_SAMPLE_RATE"] = "1"
os.environ["ADMIN_USERNAMES"] = ""
os.environ["JUDGE_CALIBRATION_PATH"] = ""
os.environ["JUDGE_MODEL"] = "llama3-70b"
os.environ["RAG_ACCESS_POLICY"] = ""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from database.connection import Base, get_db
from database.models import User, Query, Evaluation, Probability, Usage
from database.seed import seed_probabilities
from Backend.main import app
from Backend.dependencies import limiter
from core.orchestrator import orchestrate
from core.complexity_analyzer import analyze_complexity
from core.intent_detector import detect_intent
from core.decision_engine import score_model, select_best_model
from core.service import process_query, PipelineUnavailable
from Execution.Execution_layer import execute_query, ExecutionError, ABSTENTION
from Evaluation.Evaluator import extract_json, evaluate_response, calculate_quality_score
from Evaluation.worker import process_one
from Learning.learning import update_model_probabilities
from Security.Security_Guard import inspect_query, sanitize

GOOD = dict(text="A grounded answer [1].", input_tokens=20, output_tokens=8)
SCORES = dict(success=True, relevance=.9, correctness=.9, completeness=.9,
              quality_score=.9, reasoning="supported", hallucination_flags=[],
              retrieval_score=.8, retrieval_warning=False, usage=[])
CHUNK = dict(text="Employees have six sick leave days.", source="handbook.pdf",
             page=2, chunk_id="handbook:p2:c0", score=.9)
RAG_GOOD = {**GOOD, "text": json.dumps({"answerable": True, "evidence": [
    {"source_id": 1, "quote": CHUNK["text"]}]})}

class PureTests(unittest.TestCase):
    def test_topic_is_not_difficulty(self):
        self.assertEqual(analyze_complexity("Define theorem", "general")["complexity"], "low")
        self.assertEqual(analyze_complexity("Write a distributed scheduler with retries and deadlock detection", "general")["complexity"], "high")
    def test_intent_word_boundaries(self):
        for query in ("What is a web server?", "What is a user-defined function?", "Describe power supplies"):
            self.assertEqual(detect_intent(query)["intent"], "general")
    def test_policy_retrieval(self):
        self.assertTrue(orchestrate("What is our sick leave entitlement?")["retrieval_needed"])
        self.assertTrue(orchestrate("What is Sankalpa's remote work policy?")["retrieval_needed"])
        self.assertTrue(orchestrate("Explain about our company Sankalpa")["retrieval_needed"])
        self.assertFalse(orchestrate("In general, explain Sankalpa as a word")["retrieval_needed"])
        self.assertFalse(orchestrate("Explain notice period in general")["retrieval_needed"])
    def test_same_utility_independent_of_sample_count(self):
        self.assertEqual(score_model(.5,.8,.4,"low",0), score_model(.5,.8,.4,"low",50))
    def test_no_context_never_calls_provider(self):
        with patch("Execution.Execution_layer.invoke_model") as provider:
            result = execute_query("Our leave policy?", "nova-micro", "fast", retrieval_required=True)
        provider.assert_not_called()
        self.assertEqual(result["response"], ABSTENTION)
    def test_fallback_records_all_attempts(self):
        with patch("Execution.Execution_layer.invoke_model", side_effect=[TimeoutError(), GOOD]):
            result = execute_query("What is Python?", "nova-micro", "fast")
        self.assertEqual(result["model_used"], "llama3-8b")
        self.assertTrue(result["fallback_used"])
        self.assertIsNone(result["attempts"][0]["estimated_cost"])
        self.assertEqual(len(result["attempts"]), 2)
    def test_all_models_fail_explicitly(self):
        with patch("Execution.Execution_layer.invoke_model", side_effect=TimeoutError()):
            with self.assertRaises(ExecutionError) as error:
                execute_query("What is Python?", "nova-micro", "fast")
        self.assertEqual(len(error.exception.attempts), 3)
    def test_empty_answer_has_recorded_cost(self):
        with patch("Execution.Execution_layer.invoke_model", return_value={**GOOD, "text": ""}):
            with self.assertRaises(ExecutionError) as error:
                execute_query("Python?", "nova-micro", "fast")
        self.assertIsNotNone(error.exception.attempts[0]["estimated_cost"])
    def test_general_answers_have_a_hard_display_limit(self):
        long_answer = "word " * 200
        with patch("Execution.Execution_layer.invoke_model", return_value={**GOOD, "text": long_answer}):
            result = execute_query("Explain Python", "nova-micro", "fast")
        self.assertLessEqual(len(result["response"]), 600)
        self.assertTrue(result["response"].endswith("…"))
    def test_json_strings_and_bad_output(self):
        self.assertEqual(extract_json('prefix {"reasoning":"a } brace"} suffix'), {"reasoning": "a } brace"})
        self.assertIsNone(extract_json("not JSON"))
    def test_scores_reject_nan_and_out_of_range(self):
        for value in (8, float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                calculate_quality_score(value, .5, .5)
    def test_invalid_judge_output_is_not_learnable(self):
        with patch("Evaluation.Evaluator.invoke_model", return_value={**GOOD, "text": '{"relevance":8}'}):
            result = evaluate_response("q", "a")
        self.assertFalse(result["success"])
        self.assertIsNone(result["quality_score"])
        self.assertEqual(len(result["usage"]), 1)
    def test_judge_has_deterministic_setting(self):
        answer = {k: SCORES[k] for k in ("relevance","correctness","completeness","hallucination_flags","reasoning")}
        with patch("Evaluation.Evaluator.invoke_model", return_value={**GOOD, "text":json.dumps(answer)}) as call:
            self.assertTrue(evaluate_response("q","a")["success"])
        self.assertEqual(call.call_args.kwargs["temperature"], 0)
    def test_preserve_code_and_legitimate_policy_questions(self):
        self.assertEqual(sanitize("Explain for(i=0;i<3;i++)"), "Explain for(i=0;i<3;i++)")
        self.assertTrue(inspect_query("What is our company policy on fraud reporting?", True)["safe"])
        self.assertFalse(inspect_query("<script>x</script>", False)["safe"])
    def test_character_estimate_uses_full_prompt(self):
        with patch("Execution.Execution_layer.invoke_model", return_value={"text":RAG_GOOD["text"]}):
            result = execute_query("q", "nova-micro", "rag", context="[1] handbook.pdf, page 2\n"+CHUNK["text"], retrieval_required=True)
        self.assertGreater(result["input_tokens"], 50)

class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread":False}, poolclass=StaticPool)
        @event.listens_for(self.engine, "connect")
        def foreign_keys(conn, record):
            conn.execute("PRAGMA foreign_keys=ON")
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.factory() as db:
            db.add(User(id=1,username="owner",email="owner@test.example",hashed_password="unused",is_active=True))
            db.add(User(id=2,username="other",email="other@test.example",hashed_password="unused",is_active=True))
            seed_probabilities(db)
            db.commit()
        def override():
            with self.factory() as db:
                yield db
        app.dependency_overrides[get_db] = override
        limiter.enabled = False
        self.client = TestClient(app)
        self.client.__enter__()
        from auth.auth_handler import create_token
        self.headers = {"Authorization":"Bearer "+create_token(1,"owner")}
        self.other_headers = {"Authorization":"Bearer "+create_token(2,"other")}
    def tearDown(self):
        self.client.__exit__(None,None,None)
        app.dependency_overrides.clear()
        self.engine.dispose()
    def query(self, **kwargs):
        with self.factory() as db, patch("Execution.Execution_layer.invoke_model", return_value=GOOD):
            return process_query(db,1,"What is Python?","test-session", evaluation_rate=1,**kwargs)
    def test_register_login_persist_across_sessions(self):
        body=dict(username="alice",email="alice@test.example",password="strong-password")
        response=self.client.post("/api/auth/register",json=body)
        self.assertEqual(response.status_code,201,response.text)
        with self.factory() as db:
            self.assertIsNotNone(db.query(User).filter_by(username="alice").first())
        response=self.client.post("/api/auth/login",json=dict(username="ALICE",password=body["password"]))
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.client.post("/api/auth/register",json=body).status_code,409)
    def test_long_utf8_password_rejected(self):
        result=self.client.post("/api/auth/register",json=dict(username="alice",email="a@b.example",password="\u00e9"*40))
        self.assertEqual(result.status_code,422)
    def test_inactive_user_token_rejected(self):
        with self.factory() as db:
            db.get(User,1).is_active=False
            db.commit()
        self.assertEqual(self.client.get("/api/auth/me",headers=self.headers).status_code,401)
    def test_feedback_and_status_are_public_in_demo_mode(self):
        row=self.query()
        payload=dict(query_id=row.id,rating=4)
        self.assertEqual(self.client.get(f"/api/queries/{row.id}/evaluation").status_code,200)
        self.assertEqual(self.client.post("/api/feedback",json=payload).status_code,200)
        self.assertEqual(self.client.post("/api/feedback",headers=self.headers,json=payload).status_code,409)
    def test_dashboard_metrics_are_public(self):
        self.assertEqual(self.client.get("/api/metrics").status_code,200)
        self.assertEqual(self.client.get("/api/probabilities").status_code,200)
    def test_api_contract_and_status_transition(self):
        with patch("Execution.Execution_layer.invoke_model",return_value=GOOD):
            response=self.client.post("/api/chat",headers=self.headers,
                json=dict(query="What is Python?",session_id="test-session"))
        self.assertEqual(response.status_code,200,response.text)
        data=response.json()
        self.assertIsNone(data["quality_score"])
        self.assertEqual(data["evaluation_status"],"pending")
        self.assertIn("selected_model",data)
        self.assertIn("fallback_used",data)
        with patch("Evaluation.worker.evaluate_response",return_value=SCORES):
            self.assertTrue(process_one(self.factory))
        status=self.client.get(f'/api/queries/{data["query_id"]}/evaluation',headers=self.headers).json()
        self.assertEqual(status["status"],"completed")
        self.assertEqual(status["quality_score"],.9)
    def test_real_pipeline_with_fixture_retrieval(self):
        with self.factory() as db, patch("core.service.retrieve",return_value=dict(
            context="[1] handbook.pdf, page 2\n"+CHUNK["text"],chunks=[CHUNK],found=True)), patch(
                "Execution.Execution_layer.invoke_model",return_value=RAG_GOOD) as provider:
            row=process_query(db,1,"What is our sick leave entitlement?","test-session",evaluation_rate=0)
        prompt=provider.call_args.args[1]
        self.assertIn(CHUNK["text"],prompt)
        self.assertIn("ONLY",prompt)
        self.assertEqual(row.sources[0]["page"],2)
        self.assertEqual(row.evaluation_status,"skipped")
        self.assertEqual(row.status,"completed")
        self.assertEqual(row.response, CHUNK["text"]+" [1]")
    def test_api_abstention(self):
        with patch("core.service.retrieve",return_value=dict(context="",chunks=[],found=False)), patch(
                "Execution.Execution_layer.invoke_model") as provider:
            response=self.client.post("/api/chat",headers=self.headers,
                json=dict(query="What is our sick leave entitlement?",session_id="test-session"))
        self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(response.json()["abstained"])
        self.assertEqual(response.json()["evaluation_status"],"skipped")
        provider.assert_not_called()
    def test_api_provider_failure_not_answer(self):
        with patch("Execution.Execution_layer.invoke_model",side_effect=TimeoutError()):
            response=self.client.post("/api/chat",headers=self.headers,
                json=dict(query="What is Python?",session_id="test-session"))
        self.assertEqual(response.status_code,503,response.text)
        with self.factory() as db:
            row=db.query(Query).one()
            self.assertEqual(row.status,"failed")
            self.assertEqual(row.evaluation_status,"skipped")
            self.assertIsNone(row.response)
            self.assertEqual(db.query(Usage).count(),3)
    def test_retrieval_failure_never_generates(self):
        with self.factory() as db, patch("core.service.retrieve",side_effect=FileNotFoundError()), patch(
                "Execution.Execution_layer.invoke_model") as provider:
            with self.assertRaises(PipelineUnavailable):
                process_query(db,1,"What is our leave policy?","test-session")
        provider.assert_not_called()
    def test_worker_idempotent_and_seed_retained(self):
        row=self.query()
        with patch("Evaluation.worker.evaluate_response",return_value=SCORES) as judge, patch(
                "Evaluation.worker.judge_learning_allowed", return_value=True):
            self.assertTrue(process_one(self.factory))
            self.assertFalse(process_one(self.factory))
        judge.assert_called_once()
        with self.factory() as db:
            self.assertEqual(db.query(Evaluation).count(),1)
            p=db.query(Probability).filter_by(model=row.model_used,complexity=row.complexity).one()
            self.assertEqual(p.sample_count,1)
            self.assertAlmostEqual(p.p_quality, .7+(.9-.7)/6)
    def test_worker_retry_exhaustion(self):
        row=self.query()
        with patch("Evaluation.worker.evaluate_response",return_value=dict(success=False,usage=[])):
            for n in range(3):
                self.assertTrue(process_one(self.factory))
                with self.factory() as db:
                    job=db.get(Query,row.id)
                    job.evaluation_retry_at=datetime.now(timezone.utc)-timedelta(seconds=1)
                    db.commit()
        with self.factory() as db:
            self.assertEqual(db.get(Query,row.id).evaluation_status,"failed")
            self.assertEqual(db.query(Evaluation).count(),0)
            self.assertEqual(db.query(Probability).filter_by(model=row.model_used,complexity=row.complexity).one().sample_count,0)
    def test_worker_learning_rollback(self):
        row=self.query()
        with patch("Evaluation.worker.evaluate_response",return_value=SCORES), patch(
                "Evaluation.worker.update_model_probabilities",side_effect=RuntimeError()), patch(
                "Evaluation.worker.judge_learning_allowed", return_value=True):
            with self.assertRaises(RuntimeError):
                process_one(self.factory)
        with self.factory() as db:
            self.assertEqual(db.get(Query,row.id).evaluation_status,"pending")
            self.assertEqual(db.query(Evaluation).count(),0)
    def test_no_confidence_inversion_and_missing_metrics(self):
        with self.factory() as db:
            static=select_best_model(db,"high",policy="static")
            self.assertEqual(static["selected_model"],"llama3-70b")
            db.query(Probability).filter_by(model="nova-micro",complexity="high").delete()
            self.assertTrue(select_best_model(db,"high",policy="adaptive")["fallback"])
    def test_atomic_mean_multiple_observations(self):
        with self.factory() as db:
            for value in [.9,.1,.8]:
                update_model_probabilities(db,"nova-micro","low",value,100)
            db.commit()
        with self.factory() as db:
            p=db.query(Probability).filter_by(model="nova-micro",complexity="low").one()
            self.assertEqual(p.sample_count,3)
            self.assertAlmostEqual(p.p_quality,(5*.7+.9+.1+.8)/8)
    def test_migration_idempotent(self):
        from database.migrate_v3 import migrate
        migrate(self.engine)
        migrate(self.engine)
        with self.engine.connect() as conn:
            self.assertEqual(conn.execute(text("SELECT count(*) FROM schema_migrations")).scalar(),1)

class RetrievalTests(unittest.TestCase):
    def test_json_index_search_roundtrip(self):
        import RAG.vector_store as store
        vector=[1.0]+[0.0]*1023
        with tempfile.TemporaryDirectory() as path, patch.object(store,"DIRECTORY",Path(path)):
            store.build_index([{**CHUNK,"embedding":vector}])
            result=store.search_index(vector,5)
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]["page"],2)
            self.assertAlmostEqual(result[0]["score"],1.0)
            with self.assertRaises(ValueError):
                store.search_index([0.0]*1024)
            self.assertFalse(any(Path(path).glob("*.pkl")))
    def test_threshold_and_embedding_usage(self):
        from RAG.retriever import retrieve
        records=[]
        with patch("RAG.retriever.get_embedding_result",return_value=dict(
            embedding=[1.0]*1024,input_tokens=5,output_tokens=0,text="")), patch(
            "RAG.retriever.search_index",return_value=[{**CHUNK,"score":.5}]):
            # A deployment may still select a stricter cutoff explicitly.
            result=retrieve("our policy",usage=records,threshold=.75)
        self.assertFalse(result["found"])
        self.assertEqual(records[0]["stage"],"embedding")
        self.assertGreater(records[0]["estimated_cost"],0)

if __name__ == "__main__":
    unittest.main()
