"""
Streamlit clinical QA chat interface.

Features:
  - Chat interface with conversation history
  - Source citations panel (right column)
  - Hallucination confidence meter per response
  - MCQ mode for USMLE-style questions
  - PDF download of responses
  - System health status
"""

import io
import time
from typing import Optional

import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical QA System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        use_rag = st.toggle("Enable RAG Retrieval", value=True)
        use_halluc = st.toggle("Hallucination Detection", value=True)
        mcq_mode = st.toggle("MCQ Mode (A/B/C/D)", value=False)

        st.divider()
        st.subheader("System Health")
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=3)
            health = resp.json()
            status_color = "🟢" if health["status"] == "healthy" else "🟡"
            st.write(f"{status_color} **{health['status'].capitalize()}**")
            st.write(f"Corpus size: **{health['vector_store_count']:,}** chunks")
            st.write(f"Model: **{'Loaded' if health['model_loaded'] else 'Not loaded'}**")
        except Exception:
            st.write("🔴 API not reachable")

        st.divider()
        st.caption("LJMU MSc ML&AI — Amit Choudhary")
        st.caption("Student No: 1187247")

    return use_rag, use_halluc, mcq_mode


# ── Hallucination meter ────────────────────────────────────────────────────────
def render_hallucination_meter(hallucination_rate: float):
    confidence = 1.0 - hallucination_rate
    color = (
        "green" if confidence >= 0.8
        else "orange" if confidence >= 0.5
        else "red"
    )
    label = (
        "High Confidence" if confidence >= 0.8
        else "Medium Confidence" if confidence >= 0.5
        else "Low Confidence — Review Carefully"
    )
    st.markdown(
        f"""
        <div style="margin: 8px 0;">
            <span style="font-size:0.85em; color:{color}; font-weight:bold;">
                🔬 {label} ({confidence:.0%} grounded)
            </span>
            <div style="background:#eee; border-radius:4px; height:8px; margin-top:4px;">
                <div style="background:{color}; width:{confidence*100:.0f}%;
                     height:8px; border-radius:4px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Citations panel ────────────────────────────────────────────────────────────
def render_citations(citations: list[dict]):
    if not citations:
        st.info("No citations available.")
        return

    for i, cit in enumerate(citations, start=1):
        with st.expander(f"[{i}] {cit.get('title', cit.get('source', 'Source'))[:60]}"):
            if cit.get("pmid"):
                st.write(f"**PMID:** {cit['pmid']}")
            if cit.get("year"):
                st.write(f"**Year:** {cit['year']}")
            st.write(f"**Relevance:** {cit.get('relevance', 0):.3f}")
            st.write(f"**Source ID:** `{cit.get('source', '')[:80]}`")


# ── PDF export ─────────────────────────────────────────────────────────────────
def generate_pdf(question: str, answer: str, citations: list[dict]) -> bytes:
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Clinical QA Response", ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.ln(4)
        pdf.multi_cell(0, 7, f"Question: {question}")
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Answer:", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 6, answer[:3000])
        if citations:
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "References:", ln=True)
            pdf.set_font("Helvetica", size=9)
            for i, cit in enumerate(citations, start=1):
                ref = f"[{i}] {cit.get('title', cit.get('source', ''))} ({cit.get('year', '')})"
                if cit.get("pmid"):
                    ref += f" PMID: {cit['pmid']}"
                pdf.multi_cell(0, 5, ref[:120])
        return pdf.output(dest="S").encode("latin-1")
    except Exception as e:
        return f"PDF generation failed: {e}".encode()


# ── MCQ input form ─────────────────────────────────────────────────────────────
def render_mcq_form() -> Optional[dict[str, str]]:
    st.subheader("MCQ Options")
    cols = st.columns(2)
    options = {}
    for i, letter in enumerate(["A", "B", "C", "D"]):
        col = cols[i % 2]
        val = col.text_input(f"Option {letter}", key=f"opt_{letter}", placeholder=f"Option {letter}")
        if val.strip():
            options[letter] = val.strip()
    return options if len(options) == 4 else None


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    st.title("🏥 Clinical QA System")
    st.caption("Hallucination-Resistant Medical Question Answering via RAG + Fine-tuned LLMs")

    use_rag, use_halluc, mcq_mode = render_sidebar()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "hallucination_rate" in msg:
                render_hallucination_meter(msg["hallucination_rate"])

    # MCQ options input
    mcq_options = None
    if mcq_mode:
        with st.expander("Enter MCQ Options", expanded=True):
            mcq_options = render_mcq_form()

    # Chat input
    if prompt := st.chat_input("Ask a clinical question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating answer…"):
                try:
                    payload = {
                        "question": prompt,
                        "use_rag": use_rag,
                        "use_hallucination_filter": use_halluc,
                    }
                    if mcq_options:
                        payload["mcq_options"] = mcq_options

                    resp = requests.post(
                        f"{API_BASE}/query",
                        json=payload,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                    answer = result.get("answer", "No answer generated.")
                    halluc_rate = result.get("hallucination_rate", 0.0)
                    citations = result.get("citations", [])
                    resp_time = result.get("response_time_s", 0.0)
                    mcq_pred = result.get("mcq_prediction")

                    # Two-column layout: answer | citations
                    col_ans, col_cit = st.columns([2, 1])

                    with col_ans:
                        if mcq_pred:
                            st.success(f"**Predicted Answer: {mcq_pred}**")
                        st.markdown(answer)
                        render_hallucination_meter(halluc_rate)
                        st.caption(
                            f"⏱ {resp_time:.2f}s | "
                            f"📚 {result.get('sources_retrieved', 0)} sources | "
                            f"🔁 {result.get('iterations', 1)} iterations"
                        )

                        # PDF download
                        pdf_bytes = generate_pdf(prompt, answer, citations)
                        st.download_button(
                            "📄 Download as PDF",
                            data=pdf_bytes,
                            file_name="clinical_qa_response.pdf",
                            mime="application/pdf",
                        )

                    with col_cit:
                        st.subheader("📚 Sources")
                        render_citations(citations)

                    # Store in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "hallucination_rate": halluc_rate,
                        "citations": citations,
                    })

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API. Is `uvicorn src.api.main:app` running?")
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
