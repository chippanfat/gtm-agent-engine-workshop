"""Offline tests for prospect tool data boundaries."""

import importlib
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

with patch.dict(os.environ, {
    "OPENAI_API_KEY": "test-key",
    "LANGSMITH_TRACING": "false",
}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.record = deepcopy(agent.data_service.PROSPECTS[self.prospect_id])
        self.enterContext(patch.object(agent.data_service, "_PROFILES", {}))

    def safe_profile(self):
        return {
            "prospect_id": self.prospect_id,
            **{k: v for k, v in self.record.items() if k != "billing_qualification"},
        }

    def test_lookup_returns_only_contact_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {
            "prospect": {
                "prospect_id": self.prospect_id,
                "name": self.record["name"],
                "email": self.record["email"],
                "disqualified": self.record["disqualified"],
            },
            "found": True,
        })

    def test_new_profile_is_safe_in_result_and_cache(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"prospect_profile": self.safe_profile(), "found": True})
        self.assertEqual(agent.data_service._PROFILES[self.prospect_id], self.safe_profile())
        self.assertEqual(agent.data_service.PROSPECTS[self.prospect_id], self.record)
        cached = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(cached, result)

    def test_existing_profile_is_cleaned_without_rebuilding(self):
        existing = {"prospect_id": self.prospect_id, **deepcopy(self.record)}
        existing["enrichment_source"] = "cached enrichment"
        agent.data_service._PROFILES[self.prospect_id] = existing
        expected = {**self.safe_profile(), "enrichment_source": "cached enrichment"}
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertEqual(agent.data_service._PROFILES[self.prospect_id], expected)
        self.assertIn("billing_qualification", existing)

    def test_scoring_excludes_billing_and_preserves_safe_fields(self):
        offering = next(iter(agent.data_service.OFFERINGS.values()))
        for include_id in (True, False):
            with self.subTest(include_id=include_id):
                profile = deepcopy(self.record)
                expected = self.safe_profile()
                if include_id:
                    profile["prospect_id"] = self.prospect_id
                    profile["tech_stack"] = ["outdated technology"]
                else:
                    expected.pop("prospect_id")
                original = deepcopy(profile)
                score = {
                    "score": 75, "max_score": 100, "justification": "Fit",
                    "rubric_breakdown": {
                        "revenue_fit": 80, "tech_stack_match": 70,
                        "segment_fit": 75, "component_max": 100,
                    },
                }
                model = Mock()
                model.invoke.return_value = agent.ProspectScore(**score)
                with patch.object(agent, "_scoring_llm", model):
                    result = agent.score_prospect.invoke({
                        "prospect_profile": profile, "offering": offering,
                    })
                messages = model.invoke.call_args.args[0]
                prompt = messages[1]["content"]
                serialized_offering, serialized_profile = prompt.split("\n\nProspect profile:\n")
                self.assertEqual(json.loads(serialized_offering.removeprefix("Offering:\n")), offering)
                self.assertEqual(json.loads(serialized_profile), expected)
                self.assertNotIn("billing_qualification", prompt)
                for value in self.record["billing_qualification"].values():
                    self.assertNotIn(value, prompt)
                self.assertEqual(result, score)
                self.assertEqual(profile, original)

    def test_missing_prospect_return_shapes(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })

    def test_invalid_offering_does_not_score(self):
        model = Mock()
        with patch.object(agent, "_scoring_llm", model):
            result = agent.score_prospect.invoke({"prospect_profile": self.record})
        self.assertEqual(result, {
            "score": None, "error": "Cannot score without a valid offering.",
        })
        model.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
