import operator
from typing import Annotated, Dict, Optional
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


# Researcher team state
class InvestDebateState(TypedDict):
    bull_history: Annotated[str, "YES-side conversation history"]
    bear_history: Annotated[str, "NO-side conversation history"]
    history: Annotated[str, "Conversation history"]
    current_response: Annotated[str, "Latest response"]
    judge_decision: Annotated[str, "Final judge decision"]
    count: Annotated[int, "Length of the current conversation"]


# Risk management team state
class RiskDebateState(TypedDict):
    aggressive_history: Annotated[str, "Aggressive Agent's Conversation history"]
    conservative_history: Annotated[str, "Conservative Agent's Conversation history"]
    neutral_history: Annotated[str, "Neutral Agent's Conversation history"]
    history: Annotated[str, "Conversation history"]
    latest_speaker: Annotated[str, "Analyst that spoke last"]
    current_aggressive_response: Annotated[str, "Latest response by the aggressive analyst"]
    current_conservative_response: Annotated[str, "Latest response by the conservative analyst"]
    current_neutral_response: Annotated[str, "Latest response by the neutral analyst"]
    judge_decision: Annotated[str, "Judge's decision"]
    count: Annotated[int, "Length of the current conversation"]


class AgentState(MessagesState):
    # --- Polymarket market identity ---
    condition_id: Annotated[str, "Polymarket condition ID (market identifier)"]
    market_question: Annotated[str, "The full resolution question for this market"]
    resolution_criteria: Annotated[str, "How and when the market resolves"]
    resolution_date: Annotated[str, "ISO date the market resolves"]
    current_probability: Annotated[float, "Current market mid price (implied probability 0–1)"]

    # --- Liquidity gate (TradeSniper) ---
    liquidity_summary: Annotated[Dict, "TradeSniper liquidity assessment dict"]
    liquidity_ok: Annotated[bool, "True if market passed liquidity thresholds"]

    # --- Legacy compatibility fields ---
    # 'company_of_interest' maps to market_question for prompt compatibility
    company_of_interest: Annotated[str, "Market question passed to analyst prompts"]
    trade_date: Annotated[str, "Date of this analysis run (YYYY-MM-DD)"]

    sender: Annotated[str, "Agent that sent this message"]

    # Analyst reports — keyed by analyst type.
    # operator.or_ merges dicts so parallel analyst nodes each contribute their own key
    # without overwriting each other's output.
    analyst_reports: Annotated[Dict[str, str], operator.or_]

    # Researcher team discussion
    investment_debate_state: Annotated[
        InvestDebateState, "Current state of the YES/NO debate"
    ]
    investment_plan: Annotated[str, "Probability estimate and thesis from Research Manager"]

    trader_investment_plan: Annotated[str, "Position sizing plan from Position Sizer"]

    # Risk management team discussion
    risk_debate_state: Annotated[
        RiskDebateState, "Current state of the risk debate"
    ]
    final_trade_decision: Annotated[str, "Final position decision (YES/NO/SKIP + sizing)"]
    past_context: Annotated[str, "Memory log context: past decisions + calibration lessons"]
