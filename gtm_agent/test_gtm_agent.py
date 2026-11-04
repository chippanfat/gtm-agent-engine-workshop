"""Offline integration checks for the GTM agent graph."""

import json
import os
import re
import unittest
import uuid
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "offline-test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from gtm_agent import data_service
from gtm_agent import gtm_agent as app
from run import EXAMPLES


BUSINESS_TOOLS = {
    "lookup_offering", "build_prospect_profile", "get_prospect",
    "send_prospect_email", "score_prospect", "update_prospect_info",
    "get_current_rep",
}


class AgentTests(unittest.TestCase):
    def test_examples_with_scripted_model(self):
        """Execute example tool plans through the real graph without provider calls."""
        for user_id, request in EXAMPLES:
            with self.subTest(request=request):
                prospect_id = re.search(r"LEAD-\d+", request).group()
                offering_match = re.search(r"OFFER-\d+", request)
                scoring = offering_match is not None
                plan = [("get_current_rep", {})]
                if scoring:
                    if app.classify_intent(request) == "update_prospect":
                        technology = re.search(r"Kafka|Terraform|Okta", request).group()
                        plan.append(("update_prospect_info", {
                            "prospect_id": prospect_id, "technology": technology,
                        }))
                    plan.extend([
                        ("build_prospect_profile", {"prospect_id": prospect_id}),
                        ("lookup_offering", {"offering_id": offering_match.group()}),
                        ("score_prospect", None),
                    ])
                else:
                    plan.extend([
                        ("get_prospect", {"prospect_id": prospect_id}),
                        ("send_prospect_email", None),
                    ])
                remaining = iter(plan)
                model_calls = []

                def respond(messages, **kwargs):
                    model_calls.append((messages, kwargs))
                    results = {
                        m.name: json.loads(m.content)
                        for m in messages if isinstance(m, ToolMessage)
                    }
                    step = next(remaining, None)
                    if step is None:
                        message = AIMessage(content="Done")
                    else:
                        name, args = step
                        if name == "score_prospect":
                            args = {
                                "prospect_profile": results["build_prospect_profile"]["prospect_profile"],
                                "offering": results["lookup_offering"]["offering"],
                            }
                        elif name == "send_prospect_email":
                            args = {
                                "prospect": results["get_prospect"]["prospect"],
                                "subject": "Follow up", "body": request,
                            }
                        message = AIMessage(content="", tool_calls=[{
                            "name": name, "args": args, "id": str(uuid.uuid4()),
                        }])
                    return ChatResult(generations=[ChatGeneration(message=message)])

                score = app.ProspectScore(
                    score=50, justification="Offline scoring response",
                    rubric_breakdown=app.RubricBreakdown(
                        revenue_fit=50, tech_stack_match=50, segment_fit=50,
                    ),
                )
                with (
                    patch.object(ChatOpenAI, "_generate", side_effect=respond),
                    patch.object(app, "_scoring_llm") as scoring_model,
                    patch.object(app.gtm_agent, "invoke", wraps=app.gtm_agent.invoke) as invoke,
                ):
                    scoring_model.invoke.return_value = score
                    result = app.run_agent(
                        request, user_id=user_id, environment="test", thread_id="example-thread",
                    )
                self.assertEqual(result["reply"], "Done")
                config = invoke.call_args.kwargs["config"]
                self.assertEqual(config["run_name"], "GTM Assistant")
                self.assertEqual(str(config["run_id"]), result["run_id"])
                self.assertEqual(config["metadata"], {
                    "thread_id": "example-thread", "user_id": user_id,
                    "environment": "test", "request_intent": app.classify_intent(request),
                })
                outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual([m.name for m in outputs], [name for name, _ in plan])
                self.assertTrue(all(m.status == "success" for m in outputs))
                rep = data_service.get_rep(user_id)
                self.assertEqual(json.loads(outputs[0].content), {"rep": rep, "found": True})
                if scoring:
                    self.assertEqual(json.loads(outputs[-1].content), score.model_dump())
                else:
                    email = json.loads(outputs[-1].content)
                    self.assertEqual(email["status"], "sent")
                    self.assertEqual(email["from"], rep["email"])
                    self.assertEqual(email["to"], data_service.get_prospect_record(prospect_id)["email"])
                for messages, kwargs in model_calls:
                    self.assertEqual([m.content for m in messages if m.type == "system"], [app.SYSTEM_PROMPT])
                    tools = {t["function"]["name"]: t["function"] for t in kwargs["tools"]}
                    self.assertEqual(set(tools), BUSINESS_TOOLS)
                    for name in ("get_current_rep", "send_prospect_email"):
                        self.assertNotIn("runtime", tools[name]["parameters"]["properties"])


if __name__ == "__main__":
    unittest.main()
