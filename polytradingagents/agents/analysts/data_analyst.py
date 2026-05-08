"""Data Analyst — quantitative and domain-specific data analysis.

Depending on the market type, this analyst fetches and interprets:
  - Polling data (political markets)
  - On-chain / DeFi data (crypto markets)
  - Economic indicators (macro markets)
  - Sports statistics (sports markets)
  - Company filings / earnings (financial markets)

The Data Analyst complements the News Analyst: where News focuses on recent
events and narratives, Data focuses on measurable quantitative signals.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from polytradingagents.agents.utils.agent_utils import (
    get_news,
    get_global_news,
    get_language_instruction,
)
from polytradingagents.agents.utils.tool_utils import dispatch_tool_calls
from polytradingagents.agents.prompts import load_prompt


def create_data_analyst(llm):

    def data_analyst_node(state):
        current_date = state["trade_date"]
        market_question = state["market_question"]
        resolution_criteria = state.get("resolution_criteria", "")

        tools = [get_news, get_global_news]
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        system_message = load_prompt("data_analyst") + get_language_instruction()

        messages = [
            SystemMessage(
                content=(
                    "You are a quantitative data analyst. Your job is to find and interpret "
                    "measurable signals relevant to this prediction market question. "
                    "Focus on polls, statistics, surveys, on-chain data, economic indicators, "
                    "or any quantitative evidence that bears on the resolution. "
                    f"You have access to: {', '.join(tool_map)}.\n{system_message}"
                    f" The current date is {current_date}."
                    + (f" Resolution criteria: {resolution_criteria}" if resolution_criteria else "")
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

        return {"analyst_reports": {"data": result.content}}

    return data_analyst_node
