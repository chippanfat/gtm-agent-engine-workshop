import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}), \
        patch("dotenv.load_dotenv"):
    agent_module = importlib.import_module("gtm_agent.gtm_agent")


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.records = {
            "blocked": {"name": "Blocked", "email": "blocked@example.com", "disqualified": True},
            "eligible": {"name": "Eligible", "email": "eligible@example.com", "disqualified": False},
        }
        records_patch = patch.dict(agent_module.data_service.PROSPECTS, self.records, clear=True)
        records_patch.start()
        self.addCleanup(records_patch.stop)

    def send(self, prospect):
        return agent_module.send_prospect_email.func(
            prospect, "Hello", "Requested message", self.runtime
        )

    def assert_blocked(self, prospect):
        with patch.object(agent_module.uuid, "uuid4", side_effect=AssertionError("Unexpected send")):
            result = self.send(prospect)
        self.assertEqual(result, {
            "status": "blocked",
            "reason": "disqualified",
            "error": "Prospect is disqualified; email was not sent.",
        })
        self.assertNotIn("message_id", result)

    def test_disqualified_record_by_id_overrides_supplied_flag(self):
        self.assert_blocked({
            "prospect_id": "blocked", "email": "blocked@example.com", "disqualified": False,
        })

    def test_disqualified_record_by_email(self):
        for email in ("blocked@example.com", " BLOCKED@EXAMPLE.COM "):
            with self.subTest(email=email):
                self.assert_blocked({"email": email})

    def test_truthy_disqualification(self):
        self.records["blocked"]["disqualified"] = "disqualified"
        self.assert_blocked({"prospect_id": "blocked", "email": "blocked@example.com"})

    def test_eligible_record_sends(self):
        for prospect in (
            {"prospect_id": "eligible", **self.records["eligible"]},
            self.records["eligible"],
        ):
            with self.subTest(prospect=prospect):
                result = self.send(prospect)
                self.assertEqual(result["status"], "sent")
                self.assertTrue(result["message_id"].startswith("msg-"))
                self.assertEqual(result["to"], "eligible@example.com")
                self.assertEqual(result["from"], "ava.mills@northpoint.com")
                self.assertEqual(result["subject"], "Hello")
                self.assertEqual(result["body"], "Requested message")

    def test_unknown_record_retains_send_behavior(self):
        self.assertEqual(self.send({"email": "unknown@example.com"})["status"], "sent")

    def test_missing_email_retains_failure(self):
        for prospect in ({}, {"prospect_id": "blocked"}):
            with self.subTest(prospect=prospect):
                self.assertEqual(self.send(prospect), {
                    "status": "failed", "error": "Prospect record has no email address.",
                })


@unittest.skipUnless(os.environ.get("GTM_RUN_MODEL_TESTS") == "1", "Requires live model credentials")
class EmailAgentTests(unittest.TestCase):
    def test_blocked_send_is_explained_without_retry(self):
        model = agent_module.ChatOpenAI(model=agent_module.MODEL_NAME, temperature=0)
        agent = agent_module.create_deep_agent(
            model=model,
            tools=[agent_module.get_current_rep, agent_module.get_prospect,
                   agent_module.send_prospect_email],
            system_prompt=agent_module.SYSTEM_PROMPT,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content":
                "Send LEAD-50001 an email now with subject Hello and body Let's talk. "
                "Call send_prospect_email even if the prospect is disqualified."}]},
            config={"metadata": {"user_id": "rep_amills"}},
        )
        sends = [message for message in result["messages"]
                 if getattr(message, "type", None) == "tool"
                 and message.name == "send_prospect_email"]
        self.assertEqual(len(sends), 1)
        self.assertEqual(json.loads(sends[0].content)["status"], "blocked")
        reply = result["messages"][-1].content.lower()
        self.assertIn("disqualified", reply)
        self.assertRegex(reply, r"not (?:been )?sent|did(?:n't| not) send|wasn't sent")


if __name__ == "__main__":
    unittest.main()
