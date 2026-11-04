import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.rep = agent.data_service.get_rep("rep_amills")
        self.score = agent.ProspectScore(
            score=75, justification="Good fit", rubric_breakdown={
                "revenue_fit": 80, "tech_stack_match": 70, "segment_fit": 75,
            },
        )

    def test_scoring_loads_records_and_refreshes_cached_tech_stack(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                if cached:
                    agent.data_service._PROFILES["LEAD-12853"]["tech_stack"] = ["stale"]
                with patch.object(agent, "_scoring_llm") as llm:
                    llm.invoke.return_value = self.score
                    result = agent.score_prospect.invoke({
                        "prospect_id": "LEAD-12853", "offering_id": "OFFER-10001",
                    })
                self.assertEqual(result, self.score.model_dump())
                content = llm.invoke.call_args.args[0][1]["content"]
                offering_text, profile_text = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
                self.assertEqual(json.loads(offering_text), agent.data_service.get_offering("OFFER-10001"))
                profile = json.loads(profile_text)
                self.assertEqual(profile["tech_stack"], agent.data_service.fetch_tech_stack("LEAD-12853"))
                self.assertEqual(profile["engagement_history"], agent.data_service.fetch_engagement_history("LEAD-12853"))
                self.assertEqual(profile["account_details"], agent.data_service.fetch_account_details("LEAD-12853"))

    def test_scoring_missing_records_and_invalid_offerings(self):
        agent.data_service._PROFILES["missing"] = {"prospect_id": "missing"}
        cases = [("missing", "OFFER-10001", None), ("LEAD-12853", "missing", None)]
        offering = agent.data_service.get_offering("OFFER-10001")
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            cases.append(("LEAD-12853", "invalid", {**offering, field: None}))
        for prospect_id, offering_id, invalid in cases:
            with self.subTest(prospect_id=prospect_id, offering=invalid):
                with patch.dict(agent.data_service.OFFERINGS, {"invalid": invalid}), patch.object(agent, "_scoring_llm") as llm:
                    result = agent.score_prospect.invoke({"prospect_id": prospect_id, "offering_id": offering_id})
                self.assertIsNone(result["score"])
                self.assertTrue(result["error"])
                llm.invoke.assert_not_called()

    def send(self, body="Can we schedule a demo?", prospect_id="LEAD-12853"):
        return agent.send_prospect_email.func(prospect_id, "Demo", body, self.runtime)

    def test_email_resolves_contacts_and_signs_without_duplication(self):
        for signature in ("", self.rep["name"], self.rep["email"], self.rep["name"] + "\n" + self.rep["email"]):
            with self.subTest(signature=signature):
                body = "Can we schedule a demo?\n\n" + signature
                result = self.send(body)
                self.assertEqual(result["status"], "sent")
                self.assertEqual(result["to"], agent.data_service.get_prospect_record("LEAD-12853")["email"])
                self.assertEqual(result["from"], self.rep["email"])
                self.assertEqual(result["from_name"], self.rep["name"])
                self.assertEqual(result["subject"], "Demo")
                for part in (self.rep["name"], self.rep["email"]):
                    self.assertEqual(result["body"].count(part), 1)
                if signature == self.rep["name"] + "\n" + self.rep["email"]:
                    self.assertEqual(result["body"], body)

    def test_email_missing_prospect_or_email(self):
        self.assertEqual(self.send(prospect_id="missing")["status"], "failed")
        with patch.dict(agent.data_service.PROSPECTS, {"no-email": {"name": "Prospect"}}):
            self.assertEqual(self.send(prospect_id="no-email")["status"], "failed")

    def test_email_missing_or_unusable_sender(self):
        for config in ({}, {"metadata": None}, {"metadata": {"user_id": "missing"}}, {"metadata": {"user_id": 123}}):
            with self.subTest(config=config):
                self.runtime.config = config
                self.assertEqual(self.send()["status"], "failed")
        self.runtime.config = {"metadata": {"user_id": "rep_amills"}}
        for record in ({"name": "Rep"}, {"email": "rep@example.com"}, {"name": " ", "email": "rep@example.com"}, {"name": "Rep", "email": "invalid"}, {"name": "Rep", "email": "rep@"}, {"name": "Rep", "email": "rep @example.com"}):
            with self.subTest(record=record), patch.object(agent.data_service, "get_rep", return_value=record):
                self.assertEqual(self.send()["status"], "failed")

    def test_graph_executes_direct_id_calls_and_rep_identity_lookup(self):
        cases = [
            ("Score LEAD-12853 against OFFER-10001", agent.score_prospect, {"prospect_id": "LEAD-12853", "offering_id": "OFFER-10001"}),
            ("Email LEAD-12853 to schedule a demo", agent.send_prospect_email, {"prospect_id": "LEAD-12853", "subject": "Demo", "body": "Can we schedule a demo?"}),
            ("Who am I signed in as?", agent.get_current_rep, {}),
        ]
        for request, tool, args in cases:
            with self.subTest(tool=tool.name):
                model = ToolCallingModel(responses=[
                    AIMessage(content="", tool_calls=[{"name": tool.name, "args": args, "id": "call-1"}]),
                    AIMessage(content="Done"),
                ])
                graph = agent.create_deep_agent(
                    model=model,
                    tools=[agent.lookup_offering, agent.build_prospect_profile, agent.get_prospect,
                           agent.send_prospect_email, agent.score_prospect,
                           agent.update_prospect_info, agent.get_current_rep],
                    system_prompt=agent.SYSTEM_PROMPT,
                )
                with patch.object(agent, "_scoring_llm") as llm:
                    llm.invoke.return_value = self.score
                    result = graph.invoke({"messages": [("user", request)]}, config=self.runtime.config)
                outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual(len(outputs), 1)
                payload = json.loads(outputs[0].content)
                if tool is agent.score_prospect:
                    self.assertEqual(payload["score"], 75)
                elif tool is agent.send_prospect_email:
                    self.assertEqual(payload["status"], "sent")
                    self.assertIn(self.rep["email"], payload["body"])
                else:
                    self.assertEqual(payload["rep"], self.rep)


if __name__ == "__main__":
    unittest.main()
