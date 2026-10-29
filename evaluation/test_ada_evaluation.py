"""Local integrity tests. Fixtures live only in temporary directories, never results/."""
import json
from pathlib import Path
import tempfile
import types
import unittest

from evaluation.ada_core import Journal, ObservedLLM, RUBRIC, UNAVAILABLE, load_existing_dataset, parse_judge, price_usage, provider_usage
from evaluation.ada_reports import eligible, reduction, summarize, total, validate_route


class EvaluationIntegrityTests(unittest.TestCase):
    def test_exact_existing_queries_and_models(self):
        queries, models = load_existing_dataset()
        self.assertEqual(len(queries), 10)
        self.assertEqual(len({q['query_id'] for q in queries}), 10)
        self.assertEqual([m['provider'] for m in models], ['openrouter', 'groq', 'gemini', 'cohere'])

    def test_missing_usage_stays_unknown(self):
        self.assertEqual(provider_usage(types.SimpleNamespace())['total_tokens'], UNAVAILABLE)
        message = types.SimpleNamespace(usage_metadata={'input_tokens': 3, 'output_tokens': 4}, response_metadata={})
        self.assertEqual(provider_usage(message)['total_tokens'], 7)
        message.usage_metadata = {'input_tokens': -1, 'output_tokens': 4}
        self.assertEqual(provider_usage(message)['total_tokens'], UNAVAILABLE)

    def test_verified_rates_do_not_imply_verified_charges(self):
        obs = {'provider': 'p', 'model_id': 'm', 'input_tokens': 100, 'output_tokens': 20}
        prices = {'p:m': {'verified': True, 'input_per_million': 2, 'output_per_million': 5}}
        cost = price_usage(obs, prices)
        self.assertAlmostEqual(cost['standard_tariff_cost'], 0.0003)
        self.assertEqual(cost['api_cost'], UNAVAILABLE)
        prices['p:m']['billing_condition'] = 'verified_paid_standard'
        self.assertAlmostEqual(price_usage(obs, prices)['api_cost'], .0003)
        obs['input_tokens'] = UNAVAILABLE
        self.assertEqual(price_usage(obs, prices)['api_cost'], UNAVAILABLE)

    def test_no_zero_fill_and_zero_denominator(self):
        self.assertEqual(total([1, UNAVAILABLE]), UNAVAILABLE)
        self.assertEqual(total([]), UNAVAILABLE)
        self.assertEqual(reduction(0, 1), 'NOT_COMPUTED')
        stats = summarize([{'execution_status': 'FAILED', 'latency_ms': 999},
                           {'execution_status': 'SUCCESS', 'latency_ms': 10, 'overall_quality_score': 4}])
        self.assertEqual(stats['average_latency_ms'], 10)
        self.assertEqual(stats['latency_observations'], 1)
        self.assertEqual(stats['total_api_cost'], UNAVAILABLE)

    def test_strict_judge_schema(self):
        with self.assertRaises(ValueError):
            parse_judge('{}')
        from evaluation.ada_core import SCORES
        data = {s: 4 for s in SCORES}
        data['reasons'] = {s: 'fixture rationale' for s in SCORES}
        self.assertEqual(parse_judge(json.dumps(data))['overall_quality_score'], 4)
        data[SCORES[0]] = True
        with self.assertRaises(ValueError):
            parse_judge(json.dumps(data))

    def test_routing_missing_candidates_is_unvalidated(self):
        row = {'model_id': 'a', 'execution_status': 'SUCCESS', 'overall_quality_score': 4,
               'faithfulness_score': 4, 'safety_score': 4, 'api_cost': .1}
        self.assertEqual(validate_route('a', [row], ['a', 'b'])['routing_status'], 'UNVALIDATED')

    def test_routing_cost_and_safety_thresholds(self):
        rows = [{'model_id': name, 'execution_status': 'SUCCESS', 'overall_quality_score': 4,
                 'faithfulness_score': 4, 'safety_score': 4, 'api_cost': cost} for name, cost in [('a', .1), ('b', .2)]]
        self.assertEqual(validate_route('a', rows, ['a', 'b'])['routing_suitability'], 'APPROPRIATE')
        self.assertEqual(validate_route('b', rows, ['a', 'b'])['routing_suitability'], 'OVER_ROUTING')
        rows[1]['faithfulness_score'] = 2
        self.assertEqual(validate_route('b', rows, ['a', 'b'])['routing_suitability'], 'UNDER_ROUTING')
        rows[0]['incident_severity'] = 'L3'
        self.assertFalse(eligible(rows[0]))

    def test_journal_retains_redacts_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(directory, ['unit-test-secret-value'])
            journal.append('attempt_start', attempt_id='x', error='unit-test-secret-value')
            self.assertNotIn('unit-test-secret-value', journal.path.read_text())
            self.assertEqual(len(Journal(directory).events), 1)
            with journal.path.open('a') as stream:
                stream.write('{broken')
            with self.assertRaises(ValueError):
                Journal(directory)

    def test_failed_call_has_start_end_and_no_invented_usage(self):
        class FailingProvider:
            def invoke(self, *args, **kwargs):
                raise RuntimeError('fixture provider failure')
        with tempfile.TemporaryDirectory() as directory:
            rec = types.SimpleNamespace(journal=Journal(directory), prices={}, experiment_id='UNIT_ONLY', query_id='q', scope='candidate', stage='candidate')
            llm = ObservedLLM(rec, FailingProvider(), 'p', 'm', {})
            with self.assertRaises(RuntimeError):
                llm.invoke([])
            self.assertEqual([e['kind'] for e in rec.journal.events], ['attempt_start', 'attempt_end'])
            self.assertEqual(llm.last['execution_status'], 'FAILED')
            self.assertEqual(llm.last['input_tokens'], UNAVAILABLE)
            self.assertGreaterEqual(llm.last['latency_ms'], 0)
            self.assertTrue(llm.last['error_message'])


if __name__ == '__main__':
    unittest.main()
