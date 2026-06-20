"""Interactive demo of the Doubly Optimistic create-or-reuse agent.

Run with::

    pip install streamlit
    streamlit run app/streamlit_demo.py

You type a support query and pick a creation cost; the agent decides whether to
REUSE an existing action from its library or CREATE a new one, and shows the
confidence-bound reasoning behind the decision. The library and cumulative cost
persist across queries within a session, so you can watch it learn.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genaction import (  # noqa: E402
    Action,
    CreateOrReuseEnv,
    DoublyOptimisticPolicy,
    Embedder,
    LossModel,
    Query,
)

CATEGORIES = [
    "billing", "refund", "account_login", "troubleshooting", "pricing",
    "onboarding", "subscription_cancel", "sales_followup", "crm_update",
    "meeting_summary",
]

EXAMPLES = [
    ("How do I cancel my subscription?", "subscription_cancel"),
    ("Why was I charged twice this month?", "billing"),
    ("Can you summarize our meeting today?", "meeting_summary"),
    ("How do I reset my password?", "account_login"),
]


@st.cache_resource
def load_engine():
    """Build a fitted embedder + seed library once per session."""
    faq = pd.read_csv(ROOT / "data" / "sample_faq_library.csv")
    stream = pd.read_csv(ROOT / "data" / "sample_stream.csv")
    env = CreateOrReuseEnv(faq, stream, Embedder(backend="auto"), LossModel(),
                           creation_cost=0.2, seed=0)
    return env, faq


def init_state(env):
    if "library" not in st.session_state:
        st.session_state.library = [a for a in env.library]
        st.session_state.policy = DoublyOptimisticPolicy(creation_cost=0.2)
        st.session_state.policy.reset()
        st.session_state.cum_loss = 0.0
        st.session_state.cum_mismatch = 0.0
        st.session_state.cum_creation = 0.0
        st.session_state.n_created = 0
        st.session_state.n_rounds = 0
        st.session_state.history = []


def reset_session():
    for k in ["library", "policy", "cum_loss", "cum_mismatch", "cum_creation",
              "n_created", "n_rounds", "history"]:
        st.session_state.pop(k, None)


def decide(env, query_text: str, category: str, creation_cost: float):
    policy: DoublyOptimisticPolicy = st.session_state.policy
    policy.creation_cost = creation_cost

    emb = env.embedder.encode_one(query_text)
    query = Query(f"demo-{st.session_state.n_rounds}", category, query_text,
                  gold_response=f"[created response for: {query_text}]", embedding=emb)

    library = st.session_state.library
    decision = policy.act(query, library)
    estimates = policy.last_estimates

    if decision.is_create:
        action = Action(
            action_id=f"NEW-{st.session_state.n_rounds:03d}",
            category=category,
            canonical_query=query_text,
            response_text=query.gold_response,
            embedding=emb,
            created_round=st.session_state.n_rounds,
        )
        library.append(action)
        st.session_state.n_created += 1
        st.session_state.cum_creation += creation_cost
        st.session_state.cum_loss += creation_cost
        chosen = action
        round_cost = creation_cost
    else:
        chosen = library[decision.action_index]
        loss, _ = env.loss_model.realized_loss(
            emb, category, chosen.embedding, chosen.category, env._rng
        )
        st.session_state.cum_mismatch += loss
        st.session_state.cum_loss += loss
        round_cost = loss

    policy.update(query, decision, _FakeResult(round_cost, decision), library)
    st.session_state.n_rounds += 1
    st.session_state.history.insert(0, {
        "round": st.session_state.n_rounds,
        "query": query_text,
        "decision": decision.kind.value,
        "action": chosen.action_id,
        "round_cost": round(round_cost, 3),
    })
    return decision, chosen, estimates, query


class _FakeResult:
    """Minimal stand-in for StepResult so the policy can learn in the demo."""

    def __init__(self, mismatch_loss, decision):
        self.mismatch_loss = 0.0 if decision.is_create else mismatch_loss


def main():
    st.set_page_config(page_title="GenAction create-or-reuse demo", page_icon="🤖")
    env, faq = load_engine()
    init_state(env)

    st.title("🤖 GenAction — create-or-reuse decision agent")
    st.caption(
        "A cost-aware agent decides whether to **reuse** a stored response or "
        "**create** a new reusable one. Backend: "
        f"`{env.embedder.backend}`."
    )

    with st.sidebar:
        st.header("Settings")
        creation_cost = st.slider("Creation cost  c", 0.0, 1.0, 0.20, 0.05,
                                  help="Higher c => the agent reuses more.")
        st.markdown("---")
        st.metric("Library size", len(st.session_state.library))
        st.metric("Cumulative cost", f"{st.session_state.cum_loss:.2f}")
        st.metric("Actions created", st.session_state.n_created)
        st.metric("Queries handled", st.session_state.n_rounds)
        if st.button("Reset session"):
            reset_session()
            st.rerun()

    st.subheader("Ask the agent")
    cols = st.columns(len(EXAMPLES))
    for col, (text, cat) in zip(cols, EXAMPLES):
        if col.button(text, use_container_width=True):
            st.session_state["query_text"] = text
            st.session_state["category"] = cat

    query_text = st.text_input("Customer query",
                               value=st.session_state.get("query_text",
                                                           "How do I cancel my subscription?"))
    category = st.selectbox(
        "Query category (used only to score the realised mismatch loss)",
        CATEGORIES,
        index=CATEGORIES.index(st.session_state.get("category", "subscription_cancel")),
    )

    if st.button("Decide", type="primary"):
        decision, chosen, estimates, query = decide(env, query_text, category, creation_cost)

        if decision.is_create:
            st.error(f"Decision: **CREATE** a new action  (pays c = {creation_cost:g})")
            st.write("The best reusable candidate was too risky/costly to trust, "
                     "so the agent created a new reusable action.")
        else:
            st.success(f"Decision: **REUSE**  (action `{chosen.action_id}`)")
            st.markdown(f"**Canonical query:** {chosen.canonical_query}")
            st.markdown(f"**Stored response:** {chosen.response_text}")
            st.markdown(f"**Category:** `{chosen.category}`")

        # show the confidence-bound reasoning for the top candidates
        if estimates:
            rows = []
            for e in sorted(estimates, key=lambda x: x.lcb)[:5]:
                a = st.session_state.library[e.index]
                rows.append({
                    "action": a.action_id,
                    "category": a.category,
                    "prior(dist^p)": round(e.prior, 3),
                    "mu": round(e.mu, 3),
                    "LCB": round(e.lcb, 3),
                    "UCB": round(e.ucb, 3),
                    "eff_n": round(e.eff_n, 2),
                })
            st.markdown("**Top candidates by LCB** "
                        f"(CREATE is triggered when the best UCB > c = {creation_cost:g}):")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if st.session_state.history:
        st.subheader("Session history")
        st.dataframe(pd.DataFrame(st.session_state.history),
                     hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
