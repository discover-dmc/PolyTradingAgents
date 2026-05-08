# polytradingagents/graph/trading_graph.py

import logging
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

from polytradingagents.llm_clients import create_llm_client
from polytradingagents.agents import *
from polytradingagents.agents.schemas import build_run_snapshot
from polytradingagents.default_config import DEFAULT_CONFIG
from polytradingagents.agents.utils.memory import TradingMemoryLog
from polytradingagents.dataflows.utils import safe_ticker_component
from polytradingagents.agents.utils.agent_states import AgentState, InvestDebateState, RiskDebateState
from polytradingagents.dataflows.config import set_config

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor
from polytradingagents.dataflows.interface import clear_session_cache


class PolyTradingAgentsGraph:
    """Orchestrates the PolyTradingAgents framework for Polymarket prediction markets."""

    def __init__(
        self,
        selected_analysts: List[str] = None,
        debug: bool = False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        if selected_analysts is None:
            selected_analysts = ["news", "base_rate", "crowd_forecast", "data"]

        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        set_config(self.config)
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        llm_kwargs = self._get_provider_kwargs()
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        self.memory_log = TradingMemoryLog(self.config)
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.conditional_logic,
        )
        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        self.curr_state = None
        self.condition_id: Optional[str] = None
        self.log_states_dict: Dict[str, Any] = {}
        # In-session cache for _fetch_resolution: keyed by condition_id.
        self._resolution_cache: Dict[str, Optional[bool]] = {}

        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        provider = self.config.get("llm_provider", "").lower()
        if provider == "google":
            v = self.config.get("google_thinking_level")
            if v: kwargs["thinking_level"] = v
        elif provider == "openai":
            v = self.config.get("openai_reasoning_effort")
            if v: kwargs["reasoning_effort"] = v
        elif provider == "anthropic":
            v = self.config.get("anthropic_effort")
            if v: kwargs["effort"] = v
        max_retries = self.config.get("llm_max_retries")
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        return kwargs

    def _fetch_resolution(self, condition_id: str) -> Optional[bool]:
        """Check if a market has resolved and return the outcome (True=YES, False=NO, None=unresolved).

        Results cached in-session; only successful (non-None) fetches are stored
        so unresolved markets are re-checked on the next call.
        """
        if condition_id in self._resolution_cache:
            return self._resolution_cache[condition_id]
        try:
            from polytradingagents.dataflows.polymarket import get_market
            market = get_market(condition_id)
            if not market.get("closed", False):
                return None  # still open — don't cache
            tokens = market.get("tokens", [])
            for token in tokens:
                if token.get("winner", False):
                    outcome = token.get("outcome", "").upper() == "YES"
                    self._resolution_cache[condition_id] = outcome
                    return outcome
            return None
        except Exception as e:
            logger.warning("Could not fetch resolution for %s: %s", condition_id, e)
            return None

    def _resolve_pending_entries(self) -> None:
        """Resolve all pending memory log entries whose markets have now closed."""
        pending = self.memory_log.get_pending_entries()
        if not pending:
            return

        updates = []
        for entry in pending:
            cid = entry.get("ticker")  # stored as ticker for compat
            outcome = self._fetch_resolution(cid)
            if outcome is None:
                continue
            # Map binary outcome to raw_return: +1.0 if correct, -1.0 if wrong
            decision_text = entry.get("decision", "").upper()
            predicted_yes = "YES" in decision_text and "NO" not in decision_text.split("YES")[0]
            correct = (outcome == predicted_yes)
            raw_return = 1.0 if correct else -1.0
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw_return,
                alpha_return=raw_return,  # no benchmark for binary markets
            )
            updates.append({
                "ticker": cid,
                "trade_date": entry["date"],
                "raw_return": raw_return,
                "alpha_return": raw_return,
                "holding_days": 0,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def propagate(
        self,
        condition_id: str,
        trade_date: str,
        market_question: str = "",
        resolution_criteria: str = "",
        resolution_date: str = "",
        current_probability: float = 0.5,
    ) -> Tuple[Dict[str, Any], str]:
        """Run the PolyTradingAgents graph for a Polymarket market.

        Args:
            condition_id: Polymarket condition ID.
            trade_date: ISO date of this analysis run (YYYY-MM-DD).
            market_question: Full resolution question text.
            resolution_criteria: How the market resolves.
            resolution_date: ISO date the market resolves.
            current_probability: Current market mid price (0.0–1.0).

        Returns:
            Tuple of (final_state dict, signal string — one of YES/NO/SKIP).
        """
        self.condition_id = condition_id
        clear_session_cache()
        self._resolve_pending_entries()

        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], condition_id
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)
            step = checkpoint_step(self.config["data_cache_dir"], condition_id, trade_date)
            if step is not None:
                logger.info("Resuming from step %d for %s on %s", step, condition_id, trade_date)
            else:
                logger.info("Starting fresh for %s on %s", condition_id, trade_date)

        try:
            return self._run_graph(
                condition_id, trade_date, market_question,
                resolution_criteria, resolution_date, current_probability,
            )
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def _run_graph(
        self,
        condition_id: str,
        trade_date: str,
        market_question: str,
        resolution_criteria: str,
        resolution_date: str,
        current_probability: float,
    ) -> Tuple[Dict[str, Any], str]:
        past_context = self.memory_log.get_past_context(condition_id)
        init_state = self.propagator.create_initial_state(
            condition_id=condition_id,
            trade_date=trade_date,
            market_question=market_question,
            resolution_criteria=resolution_criteria,
            resolution_date=resolution_date,
            current_probability=current_probability,
            past_context=past_context,
        )
        args = self.propagator.get_graph_args()

        if self.config.get("checkpoint_enabled"):
            tid = thread_id(condition_id, trade_date)
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            trace = []
            for chunk in self.graph.stream(init_state, **args):
                if chunk.get("messages"):
                    chunk["messages"][-1].pretty_print()
                trace.append(chunk)
            final_state = trace[-1]
        else:
            final_state = self.graph.invoke(init_state, **args)

        self.curr_state = final_state
        self._log_state(trade_date, final_state)

        self.memory_log.store_decision(
            ticker=condition_id,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(self.config["data_cache_dir"], condition_id, trade_date)

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date: str, final_state: Dict[str, Any]) -> None:
        snapshot = build_run_snapshot(trade_date, final_state)
        snapshot_dict = snapshot.model_dump()
        self.log_states_dict[str(trade_date)] = snapshot_dict

        safe_id = safe_ticker_component(self.condition_id)
        directory = Path(self.config["results_dir"]) / safe_id / "PolyTradingStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_dict, f, indent=4)

    def process_signal(self, full_signal: str) -> str:
        """Extract YES / NO / SKIP from the Portfolio Manager's decision."""
        return self.signal_processor.process_signal(full_signal)
