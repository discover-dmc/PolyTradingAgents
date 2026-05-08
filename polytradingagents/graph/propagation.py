# polytradingagents/graph/propagation.py

from typing import Any, Dict, List, Optional

from polytradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit: int = 100):
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        condition_id: str,
        trade_date: str,
        market_question: str = "",
        resolution_criteria: str = "",
        resolution_date: str = "",
        current_probability: float = 0.5,
        past_context: str = "",
    ) -> Dict[str, Any]:
        """Create the initial state for the PolyTradingAgents graph.

        Args:
            condition_id: Polymarket condition ID (market identifier).
            trade_date: ISO date of this analysis run.
            market_question: Full resolution question text.
            resolution_criteria: How and when the market resolves.
            resolution_date: ISO date the market resolves.
            current_probability: Current market mid price (0.0–1.0).
            past_context: Memory log context from previous runs.
        """
        # company_of_interest is kept for backward compat with analyst prompts
        # that use state["company_of_interest"] — map it to market_question.
        question_display = market_question or condition_id

        return {
            "messages": [("human", question_display)],
            "condition_id": condition_id,
            "market_question": question_display,
            "resolution_criteria": resolution_criteria,
            "resolution_date": resolution_date,
            "current_probability": current_probability,
            # Backward compat alias used by analyst node prompts
            "company_of_interest": question_display,
            "trade_date": str(trade_date),
            "past_context": past_context,
            # TradeSniper gate — initialised to False; TradeSniper sets the real value
            "liquidity_summary": {},
            "liquidity_ok": False,
            # Analyst reports accumulate via operator.or_ as analysts complete
            "analyst_reports": {},
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
        }

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation."""
        config: Dict[str, Any] = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
