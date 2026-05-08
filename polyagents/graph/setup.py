# polyagents/graph/setup.py

from typing import Any, List

from langgraph.graph import END, START, StateGraph

from polyagents.agents import *
from polyagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic
from .node_names import ANALYST_NODE_NAMES, NodeNames


class GraphSetup:
    """Handles the setup and configuration of the PolyAgents graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        conditional_logic: ConditionalLogic,
    ):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.conditional_logic = conditional_logic

    def setup_graph(self, selected_analysts: List[str] = None):
        """Set up and compile the PolyAgents workflow graph.

        Graph flow:
          START → TradeSniper (liquidity gate)
            ├─ illiquid → PortfolioManager (immediate SKIP) → END
            └─ liquid   → [analysts in parallel, fan-in] → YES Researcher ←→ NO Researcher
                          → Research Manager → Position Sizer
                          → Risk Debate → Portfolio Manager → END

        Args:
            selected_analysts: Analyst types to include. Options:
                "news", "base_rate", "crowd_forecast", "data".
        """
        if selected_analysts is None:
            selected_analysts = ["news", "base_rate", "crowd_forecast", "data"]
        if not selected_analysts:
            raise ValueError("PolyAgents Graph Setup Error: no analysts selected!")

        unknown = set(selected_analysts) - set(ANALYST_NODE_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown analyst type(s): {sorted(unknown)}. "
                f"Valid options: {sorted(ANALYST_NODE_NAMES)}"
            )

        analyst_factories = {
            "news": create_news_analyst,
            "base_rate": create_base_rate_analyst,
            "crowd_forecast": create_crowd_forecast_analyst,
            "data": create_data_analyst,
        }

        workflow = StateGraph(AgentState)

        # TradeSniper — liquidity gate (always first)
        workflow.add_node(NodeNames.TRADE_SNIPER, create_trade_sniper(self.quick_thinking_llm))

        # Analyst nodes (parallel — only reached if TradeSniper passes)
        for key in selected_analysts:
            node_name = ANALYST_NODE_NAMES[key]
            workflow.add_node(node_name, analyst_factories[key](self.quick_thinking_llm))

        # Researcher + manager nodes
        workflow.add_node(NodeNames.BULL_RESEARCHER, create_bull_researcher(self.quick_thinking_llm))
        workflow.add_node(NodeNames.BEAR_RESEARCHER, create_bear_researcher(self.quick_thinking_llm))
        workflow.add_node(NodeNames.RESEARCH_MANAGER, create_research_manager(self.deep_thinking_llm))
        workflow.add_node(NodeNames.TRADER, create_trader(self.quick_thinking_llm))

        # Risk analysis nodes
        workflow.add_node(NodeNames.AGGRESSIVE_ANALYST, create_aggressive_debator(self.quick_thinking_llm))
        workflow.add_node(NodeNames.NEUTRAL_ANALYST, create_neutral_debator(self.quick_thinking_llm))
        workflow.add_node(NodeNames.CONSERVATIVE_ANALYST, create_conservative_debator(self.quick_thinking_llm))
        workflow.add_node(NodeNames.PORTFOLIO_MANAGER, create_portfolio_manager(self.deep_thinking_llm))

        # START → TradeSniper
        workflow.add_edge(START, NodeNames.TRADE_SNIPER)

        # TradeSniper → analysts (liquid, parallel fan-out) OR → PortfolioManager (skip)
        # The router returns a list of node names for fan-out, or a single string for skip.
        analyst_names = [ANALYST_NODE_NAMES[k] for k in selected_analysts]
        liquidity_router = self.conditional_logic.build_liquidity_router(analyst_names)

        sniper_targets = {name: name for name in analyst_names}
        sniper_targets[NodeNames.PORTFOLIO_MANAGER] = NodeNames.PORTFOLIO_MANAGER

        workflow.add_conditional_edges(
            NodeNames.TRADE_SNIPER,
            liquidity_router,
            sniper_targets,
        )

        # Fan-in: each analyst → YES Researcher
        for key in selected_analysts:
            workflow.add_edge(ANALYST_NODE_NAMES[key], NodeNames.BULL_RESEARCHER)

        # Research debate loop
        workflow.add_conditional_edges(
            NodeNames.BULL_RESEARCHER,
            self.conditional_logic.should_continue_debate,
            {
                NodeNames.BEAR_RESEARCHER: NodeNames.BEAR_RESEARCHER,
                NodeNames.RESEARCH_MANAGER: NodeNames.RESEARCH_MANAGER,
            },
        )
        workflow.add_conditional_edges(
            NodeNames.BEAR_RESEARCHER,
            self.conditional_logic.should_continue_debate,
            {
                NodeNames.BULL_RESEARCHER: NodeNames.BULL_RESEARCHER,
                NodeNames.RESEARCH_MANAGER: NodeNames.RESEARCH_MANAGER,
            },
        )

        workflow.add_edge(NodeNames.RESEARCH_MANAGER, NodeNames.TRADER)
        workflow.add_edge(NodeNames.TRADER, NodeNames.AGGRESSIVE_ANALYST)

        # Risk debate loop
        workflow.add_conditional_edges(
            NodeNames.AGGRESSIVE_ANALYST,
            self.conditional_logic.should_continue_risk_analysis,
            {
                NodeNames.CONSERVATIVE_ANALYST: NodeNames.CONSERVATIVE_ANALYST,
                NodeNames.PORTFOLIO_MANAGER: NodeNames.PORTFOLIO_MANAGER,
            },
        )
        workflow.add_conditional_edges(
            NodeNames.CONSERVATIVE_ANALYST,
            self.conditional_logic.should_continue_risk_analysis,
            {
                NodeNames.NEUTRAL_ANALYST: NodeNames.NEUTRAL_ANALYST,
                NodeNames.PORTFOLIO_MANAGER: NodeNames.PORTFOLIO_MANAGER,
            },
        )
        workflow.add_conditional_edges(
            NodeNames.NEUTRAL_ANALYST,
            self.conditional_logic.should_continue_risk_analysis,
            {
                NodeNames.AGGRESSIVE_ANALYST: NodeNames.AGGRESSIVE_ANALYST,
                NodeNames.PORTFOLIO_MANAGER: NodeNames.PORTFOLIO_MANAGER,
            },
        )

        workflow.add_edge(NodeNames.PORTFOLIO_MANAGER, END)

        return workflow
