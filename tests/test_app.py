"""Pruebas locales de reglas que no requieren clave ni llamadas a Groq."""

import io
import sqlite3
import unittest

import pandas as pd

from app import load_csv, run_safe_query, validate_query, schema_description


class TestCSVApp(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {"Genero": ["F", "M", "F", "M"], "Edad": [21, 36, 30, 33], "Gasto": [10, 20, 50, 20]}
        )

    def test_load_csv(self):
        parsed = load_csv(self.df.to_csv(index=False).encode("utf-8"))
        self.assertEqual(len(parsed), 4)
        self.assertEqual(list(parsed.columns), ["Genero", "Edad", "Gasto"])

    def test_empty_csv(self):
        with self.assertRaises(ValueError):
            load_csv(b"")

    def test_schema(self):
        description = schema_description(self.df)
        self.assertIn("Genero", description)
        self.assertIn("F", description)

    def test_count_all_rows(self):
        result = run_safe_query(self.df, 'SELECT COUNT(*) AS total FROM datos')
        self.assertEqual(result.iat[0, 0], 4)

    def test_grouping_is_exact(self):
        result = run_safe_query(
            self.df,
            'SELECT "Genero", COUNT(*) AS total FROM datos GROUP BY "Genero" ORDER BY "Genero"',
        )
        self.assertEqual(result["total"].tolist(), [2, 2])

    def test_average(self):
        result = run_safe_query(self.df, 'SELECT AVG("Edad") FROM datos')
        self.assertEqual(result.iat[0, 0], 30.0)

    def test_reject_writes(self):
        for statement in ['DROP TABLE datos', 'DELETE FROM datos', 'SELECT * FROM datos; DROP TABLE datos']:
            with self.assertRaises(ValueError):
                validate_query(statement)

    def test_reject_other_tables(self):
        with self.assertRaises((sqlite3.DatabaseError, pd.errors.DatabaseError)):
            run_safe_query(self.df, 'SELECT name FROM sqlite_master')


if __name__ == "__main__":
    unittest.main()

# Estas pruebas utilizan un cliente ficticio: no hacen peticiones reales a la API.
class TestGroqPlanContract(unittest.TestCase):
    def test_plan_parser_handles_out_of_scope(self):
        from types import SimpleNamespace
        from app import plan_sql

        class FakeCompletions:
            def create(self, **kwargs):
                assert kwargs["model"] == "openai/gpt-oss-20b"
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content='{"answerable": false, "sql": "", "reason": "No hay datos geográficos"}'
                    ))]
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        frame = pd.DataFrame({"Edad": [20, 30]})
        plan = plan_sql(fake_client, frame, "¿Cuál es la capital de Francia?")
        self.assertFalse(plan["answerable"])
        self.assertEqual(plan["sql"], "")


class TestGroqErrors(unittest.TestCase):
    def test_rate_limit_is_not_payment_prompt(self):
        from app import explain_api_error
        RateLimitError = type("RateLimitError", (Exception,), {})
        text = explain_api_error(RateLimitError())
        self.assertIn("plan gratuito", text)
        self.assertNotIn("paga", text.lower())

    def test_authentication_message_has_no_key(self):
        from app import explain_api_error
        AuthenticationError = type("AuthenticationError", (Exception,), {})
        self.assertIn("GROQ_API_KEY", explain_api_error(AuthenticationError()))
