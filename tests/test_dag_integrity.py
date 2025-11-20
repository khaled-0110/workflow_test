# tests/test_dag_integrity.py

from airflow.models import DagBag
import os

def test_dag_integrity():
    root_dir = os.environ.get("ROOT_DIR", ".")
    dags_path = os.path.join(root_dir, "dags")
    dag_bag = DagBag(dag_folder=dags_path, include_examples=False)
    print("DAGs found:", dag_bag.dags.keys())
    print("Import errors:", dag_bag.import_errors)
    assert len(dag_bag.import_errors) == 0, f"DAG import failures: {dag_bag.import_errors}"
    assert 'tweets_classification_pipeline' in dag_bag.dags, "DAG 'tweets_classification_pipeline' is missing"