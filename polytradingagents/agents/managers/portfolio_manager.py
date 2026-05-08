"""Portfolio Manager: synthesises the risk-analyst debate into a PositionDecision.

Produces a typed PositionDecision (YES/NO/SKIP + probability + Kelly sizing)
via structured output, then renders it to markdown for memory log and reports.
When liquidity_ok=False the market was SKIP'd by TradeSniper and this node
emits a SKIP decision immediately without reading the debate.
"""
from __future__ import annotations

from polytradingagents.agents.schemas import PositionDecision, render_position_decision
from polytradingagents.agents.utils.agent_utils import get_language_instruction
from polytradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PositionDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        # Fast-path: TradeSniper vetoed this market
        if not state.get("liquidity_ok", True):
            liq = state.get("liquidity_summary", {})
            reason = (
                f"Market failed liquidity check — "
                f"volume=${liq.get('volume_24h', 0):,.0f}, "
                f"liquidity=${liq.get('liquidity', 0):,.0f}, "
                f"spread={liq.get('yes_spread') or 0:.1%}, "
                f"depth=${liq.get('yes_depth_usd', 0):,.0f}"
            )
            skip_decision = render_position_decision(PositionDecision(
                direction="SKIP",
                estimated_probability=state.get("current_probability", 0.5),
                market_probability=state.get("current_probability", 0.5),
                edge=0.0,
                kelly_fraction=0.0,
                confidence="Low",
                reasoning=reason,
            ))
            return {"final_trade_decision": skip_decision}

        market_question = state.get("market_question", state.get("company_of_interest", ""))
        current_prob = state.get("current_probability", 0.5)
        history = state["risk_debate_state"]["history"]
        research_plan = state.get("investment_plan", "")
        trader_plan = state.get("trader_investment_plan", "")
        past_context = state.get("past_context", "")
        resolution_date = state.get("resolution_date", "")

        lessons_line = f"Prior calibration lessons:\n{past_context}\n" if past_context else ""
        resolution_line = f"Resolves: {resolution_date}\n" if resolution_date else ""

        prompt = f"""You are the Portfolio Manager for a Polymarket prediction market trading system.

Market: {market_question}
{resolution_line}Current market probability (mid price): {current_prob:.1%}
{lessons_line}
Research Manager's probability estimate and plan: {research_plan}
Position Sizer's proposal: {trader_plan}

Risk Analysts Debate:
{history}

---

Produce a final PositionDecision:
- direction: YES (buy YES token), NO (buy NO token), or SKIP (insufficient edge)
- estimated_probability: your best estimate of true probability (0.0–1.0)
- market_probability: the current market price above ({current_prob:.3f})
- edge: estimated_probability − market_probability (positive = market underprices YES)
- kelly_fraction: half-Kelly sizing capped at 0.25 — formula: |edge| / (1 − losing_prob)
- confidence: High/Medium/Low based on evidence quality
- reasoning: key factors driving the estimate

SKIP if |edge| < 0.03 or confidence is Low with thin evidence.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm, llm, prompt, render_position_decision, "Portfolio Manager"
        )

        risk_debate_state = state["risk_debate_state"]
        new_risk_debate_state = {**risk_debate_state, "judge_decision": final_trade_decision, "latest_speaker": "Judge"}

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
