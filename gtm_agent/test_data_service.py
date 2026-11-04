import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from . import data_service


with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    with patch("dotenv.load_dotenv"):
        agent = importlib.import_module("gtm_agent.gtm_agent")


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-39002"
        record = data_service.get_prospect_record(self.prospect_id)
        self.addCleanup(patch.stopall)
        patch.object(data_service, "PROSPECTS", {
            self.prospect_id: {**record, "tech_stack": list(record["tech_stack"])},
        }).start()
        patch.object(data_service, "_PROFILES", {}).start()

    def test_update_persists_technology_and_returns_stored_stack(self):
        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        stored = data_service.fetch_tech_stack(self.prospect_id)
        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Terraform", stored)
        self.assertEqual(result["tech_stack"], stored)

    def test_update_rebuilds_cached_profile(self):
        cached = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Terraform", cached["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, "Terraform")
        rebuilt = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertTrue(rebuilt["found"])
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertEqual(result["tech_stack"].count("Terraform"), 1)
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))

    def test_unknown_prospect_is_not_updated(self):
        result = data_service.update_prospect_info("unknown", "Terraform")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertIsNone(data_service.get_prospect_record("unknown"))

    def test_score_reads_updated_stack_even_with_old_profile(self):
        old = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        agent.update_prospect_info.invoke({
            "prospect_id": self.prospect_id, "technology": "Terraform",
        })
        rebuilt = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertIn("Terraform", data_service.fetch_tech_stack(self.prospect_id))
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])
        offering = data_service.get_offering("OFFER-10004")

        def score(messages):
            profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
            self.assertIn("Terraform", profile["tech_stack"])
            self.assertEqual(profile["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))
            return agent.ProspectScore(
                score=100, justification="All required technologies are present.",
                rubric_breakdown=agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100, segment_fit=100,
                ),
            )

        with patch.object(agent, "_scoring_llm", Mock(invoke=score)):
            result = agent.score_prospect.invoke({
                "prospect_profile": old["prospect_profile"], "offering": offering,
            })

        self.assertEqual(result["score"], 100)
