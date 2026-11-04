import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class EmailToolTests(unittest.TestCase):
    def test_plain_text_email_sends(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") or "test-key"}):
            agent = importlib.import_module("gtm_agent.gtm_agent")
        body = "Are you available for a pricing review next week?"
        result = agent.send_prospect_email.func(
            prospect={"name": "Prospect", "email": "prospect@example.com"},
            subject="Pricing review",
            body=body,
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_kpatel"}}),
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)
        self.assertEqual(result["to"], "prospect@example.com")
        self.assertEqual(result["from"], "kavya.patel@northpoint.com")
        self.assertNotIn("attachments", result)


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "Live agent tests require OPENAI_API_KEY")
class EmailAgentTests(unittest.TestCase):
    def send_email(self, request, user_id):
        agent = importlib.import_module("gtm_agent.gtm_agent")
        result = agent.run_agent(request, user_id=user_id)
        emails = [
            json.loads(message.content)
            for message in result["messages"]
            if getattr(message, "type", None) == "tool"
            and getattr(message, "name", None) == "send_prospect_email"
        ]
        self.assertTrue(emails, "The requested email must be sent")
        for email in emails:
            self.assertEqual(email["status"], "sent")
            self.assertNotRegex(
                email["body"].lower(),
                r"\b(?:please find attached|i(?:'ve| have) attached|attached (?:is|are)|deck is attached|document is attached)\b",
            )
        return emails, result["reply"]

    def test_pricing_deck_is_not_claimed_as_attached(self):
        emails, reply = self.send_email(
            "Email LEAD-50003 to send over our pricing deck.", "rep_sbrown"
        )
        self.assertIn("pricing", emails[0]["body"].lower())
        self.assertRegex(
            reply.lower(),
            r"(?:not attached|no attachment|without (?:an? |the )?attach|cannot attach|can't attach|unable to attach|could not attach|couldn't attach)",
        )
        self.assertNotRegex(
            reply.lower(), r"\b(?:i(?:'ve| have) attached|deck (?:was|is) included)\b"
        )

    def test_ordinary_text_email_still_sends(self):
        emails, reply = self.send_email(
            "Please email LEAD-70001 to confirm their availability for a pricing review.",
            "rep_kpatel",
        )
        self.assertIn("pricing", emails[0]["body"].lower())
        self.assertNotRegex(emails[0]["body"].lower(), r"\b(?:attached|enclosed)\b")
        self.assertNotRegex(reply.lower(), r"\b(?:attached|enclosed)\b")


if __name__ == "__main__":
    unittest.main()
