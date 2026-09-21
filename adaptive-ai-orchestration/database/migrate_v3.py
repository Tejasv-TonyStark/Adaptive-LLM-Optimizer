"""Additive, transactional v3 migration. Review and back up before deployment.

Existing queries retain NULL ownership; no new user receives their access.
Old token columns remain generation-only. Historical total costs cannot be
reconstructed. Legacy in-memory accounts cannot be migrated.
"""
from sqlalchemy import inspect, text
from sqlalchemy.schema import CreateColumn
from database.connection import engine, Base
from database import models

def migrate(target=engine):
    with target.begin() as connection:
        Base.metadata.create_all(connection)
        for name in ("queries", "evaluations"):
            present = {c["name"] for c in inspect(connection).get_columns(name)}
            for column in Base.metadata.tables[name].columns:
                if column.name not in present:
                    ddl = str(CreateColumn(column).compile(dialect=connection.dialect))
                    connection.execute(text(f'ALTER TABLE "{name}" ADD COLUMN {ddl}'))
        connection.execute(text("UPDATE queries SET status='completed' WHERE status IS NULL"))
        connection.execute(text("UPDATE queries SET evaluation_attempts=0 WHERE evaluation_attempts IS NULL"))
        connection.execute(text("UPDATE queries SET evaluation_status=CASE WHEN EXISTS "
            "(SELECT 1 FROM evaluations WHERE evaluations.query_id=queries.id) "
            "THEN 'completed' ELSE 'skipped' END WHERE evaluation_status IS NULL"))
        # Do not silently merge independently learned distributions.
        conflicts = connection.execute(text("SELECT 1 FROM probabilities a JOIN probabilities b "
            "ON a.complexity=b.complexity WHERE a.model='haiku' AND b.model='llama3-70b'")).first()
        if conflicts:
            raise RuntimeError("Both legacy and new model metrics exist; reconcile them before migration.")
        connection.execute(text("UPDATE probabilities SET model='llama3-70b' WHERE model='haiku'"))
        connection.execute(text("UPDATE queries SET model_used='llama3-70b' WHERE model_used='haiku'"))
        connection.execute(text("UPDATE queries SET selected_model='llama3-70b' WHERE selected_model='haiku'"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_queries_user_id ON queries(user_id)"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_queries_evaluation_status ON queries(evaluation_status)"))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_username_ci ON users(lower(username))"))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email_ci ON users(lower(email))"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations "
                                "(version VARCHAR(30) PRIMARY KEY)"))
        if not connection.execute(text("SELECT 1 FROM schema_migrations WHERE version='v3'")).first():
            connection.execute(text("INSERT INTO schema_migrations(version) VALUES ('v3')"))
if __name__ == "__main__":
    migrate()
    print("v3 migration complete. Legacy queries remain unowned.")
