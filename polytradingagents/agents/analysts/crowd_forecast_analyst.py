"""Crowd Forecast Analyst — aggregates external prediction market signals.

Collects probability estimates from:
  - Polymarket current market price (already in state)
  - Metaculus community predictions
  - Manifold Markets
  - Prediction market consensus aggregators

Provides a "wisdom of crowds" probability estimate and notes any significant
divergence between platforms (divergence itself is a signal).
"""
from langchain_core.messages import HumanMessage, SystemMessage

from polytradingagents.agents.utils.agent_utils import get_news, get_language_instruction
from polytradingagents.agents.utils.tool_utils import dispatch_tool_calls
from polytradingagents.agents.prompts import load_prompt


def create_crowd_forecast_analyst(llm):

    def crowd_forecast_analyst_node(state):
        current_date = state["trade_date"]
        market_question = state["market_question"]
        current_probability = state.get("current_probability", 0.5)

        tools = [get_news]
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        system_message = load_prompt("crowd_forecast_analyst") + get_language_instruction()

        messages = [
            SystemMessage(
                content=(
                    "You are a crowd forecast aggregator. Your job is to find what other "
                    "prediction markets and forecasters think about this question. "
                    "Look for Metaculus, Manifold Markets, Good Judgment Project, or other "
                    "forecasting platforms that may have similar questions. "
                    f"You have access to: {', '.join(tool_map)}.\n{system_message}"
                    f" The current date is {current_date}. "
                    f" The Polymarket current probability is {current_probability:.1%}."
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

        return {"analyst_reports": {"crowd_forecast": result.content}}

    return crowd_forecast_analyst_node
