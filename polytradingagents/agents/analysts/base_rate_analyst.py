"""Base Rate Analyst — historical frequency analysis for Polymarket questions.

Researches the historical base rate for the event type being predicted.
Examples:
  - "How often does the Fed cut rates at the next meeting given current conditions?"
  - "What % of similar geopolitical events escalated further?"
  - "What is the historical accuracy of prediction markets at this time horizon?"

The base rate grounds the probability estimate before any current-event evidence
is layered on top.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from polytradingagents.agents.utils.agent_utils import get_news, get_language_instruction
from polytradingagents.agents.utils.tool_utils import dispatch_tool_calls
from polytradingagents.agents.prompts import load_prompt


def create_base_rate_analyst(llm):

    def base_rate_analyst_node(state):
        current_date = state["trade_date"]
        market_question = state["market_question"]

        tools = [get_news]
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        system_message = load_prompt("base_rate_analyst") + get_language_instruction()

        messages = [
            SystemMessage(
                content=(
                    "You are a superforecaster specializing in base rate research. "
                    "Your job is to find the historical frequency of events similar to the market question. "
                    "Use search tools to find comparable past events and their outcomes. "
                    f"You have access to: {', '.join(tool_map)}.\n{system_message}"
                    f" The current date is {current_date}."
                )
            ),
            HumanMessage(content=market_question),
        ]

        while True:
            result = llm_with_tools.invoke(messages)
            messages.append(result)
            if not result.tool_calls:
                break
            messages.extend(dispatch_tool_calls(tool_map, result.tool_calls))

        return {"analyst_reports": {"base_rate": result.content}}

    return base_rate_analyst_node
