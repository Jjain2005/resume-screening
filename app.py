import os
import json
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from modules.knowledge_base.taxonomy import SkillTaxonomy
from modules.resume_processing.extractor import ResumeExtractor
from modules.resume_processing.models import ProcessedResume, SectionType
from modules.jd_processing.parser import JobDescriptionParser
from modules.jd_processing.models import (
    JobDescription,
    RequirementItem,
    RequirementType,
    RequirementPriority,
    MatchingLogic
)
from modules.matching_engine.hybrid_engine import HybridMatchingEngine
from modules.matching_engine.models import MatchClassification, EvidenceStrength
from modules.scoring.evaluator import CandidateEvaluator, ScoringWeights, Recommendation
from modules.reporting.reporter import ScreeningReportGenerator
from modules.llm import GroqClientManager, GroqContextualEvaluator, DEFAULT_GROQ_MODEL

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAXONOMY_PATH = os.path.join(BASE_DIR, "data", "knowledge_base", "skills_taxonomy.json")
TEST_RESUMES_DIR = os.path.join(BASE_DIR, "data", "test_resumes")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

st.set_page_config(
    page_title="Intelligent Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Standard Recommendation Categories
# -------------------------------------------------------------
REC_SELECTION = "Recommended for Selection"
REC_MANUAL_REVIEW = "Manual Review Required"
REC_NOT_RECOMMENDED = "Not Recommended"

def normalize_recommendation(rec_val: Any) -> str:
    """Robustly maps any recommendation value or enum to the 3 standard categories."""
    if hasattr(rec_val, "value"):
        rec_str = str(rec_val.value)
    else:
        rec_str = str(rec_val)

    rec_lower = rec_str.lower()
    if "selection" in rec_lower or "interview" in rec_lower or "strongly" in rec_lower:
        return REC_SELECTION
    elif "review" in rec_lower or "borderline" in rec_lower or "potential" in rec_lower:
        return REC_MANUAL_REVIEW
    else:
        return REC_NOT_RECOMMENDED

# -------------------------------------------------------------
# Caching Core Engine Components
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Skill Knowledge Base and Hybrid Engine...")
def load_heavy_engines():
    taxonomy = SkillTaxonomy(TAXONOMY_PATH)
    matching_engine = HybridMatchingEngine(taxonomy)
    return taxonomy, matching_engine

taxonomy, matching_engine = load_heavy_engines()
resume_extractor = ResumeExtractor()

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "confirmed_jd" not in st.session_state:
    st.session_state.confirmed_jd = None

if "extracted_requirements" not in st.session_state:
    st.session_state.extracted_requirements = []

if "custom_jd_title" not in st.session_state:
    st.session_state.custom_jd_title = ""

if "custom_jd_text" not in st.session_state:
    st.session_state.custom_jd_text = ""

if "screening_reports" not in st.session_state:
    st.session_state.screening_reports = []

if "comparison_df" not in st.session_state:
    st.session_state.comparison_df = pd.DataFrame()

# -------------------------------------------------------------
# Sidebar: Settings & Configuration
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Engine Settings")
    st.caption("Universal evaluation engine: criteria and weights depend dynamically on the uploaded Job Description.")

    with st.expander("🤖 Groq AI Analysis (Optional)", expanded=True):
        groq_manager = GroqClientManager()
        use_groq = st.toggle(
            "Enable Groq AI",
            value=groq_manager.is_available,
            help="Uses Groq LLM to deeply parse Job Descriptions and perform evidence-grounded requirement evaluation validated by Python."
        )
        groq_api_key_input = st.text_input(
            "Groq API Key",
            value=groq_manager.api_key or os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Get a free key at console.groq.com/keys or configure in .env"
        )
        available_models = groq_manager.get_available_models()
        default_idx = available_models.index(groq_manager.model) if groq_manager.model in available_models else 0
        groq_model_choice = st.selectbox(
            "Groq Model",
            options=available_models,
            index=default_idx
        )
        if groq_api_key_input:
            groq_manager.update_credentials(groq_api_key_input, groq_model_choice)

        if use_groq:
            if groq_manager.is_available:
                st.caption(f"🟢 Connected to `{groq_manager.model}`")
            else:
                st.warning("Enter a valid Groq API Key to enable AI analysis.")

        groq_evaluator = GroqContextualEvaluator(groq_manager)

    with st.expander("Matching Thresholds", expanded=False):
        fuzzy_th = st.slider("Fuzzy Match Threshold (%)", 70, 95, 85, 1)
        sem_th = st.slider("Semantic Match Threshold (Cosine)", 0.55, 0.85, 0.76, 0.02)
        sem_part_th = st.slider("Partial Semantic Threshold", 0.40, 0.65, 0.65, 0.02)

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base Info")
    st.write(f"**Categories:** {len(taxonomy.categories)}")
    st.write(f"**Canonical Skills:** {len(taxonomy.skills)}")
    st.write(f"**Indexed Terms / Aliases:** {len(taxonomy.get_all_lookup_terms())}")
    if taxonomy.unknown_skills:
        st.caption(f"Flagged {len(taxonomy.unknown_skills)} unknown terms for review.")

jd_parser = JobDescriptionParser(taxonomy, groq_manager=groq_manager if use_groq else None)
evaluator = CandidateEvaluator()

# -------------------------------------------------------------
# Main Navigation Tabs (3 Core Workflow Sections)
# -------------------------------------------------------------
st.title("Universal Intelligent Resume Screening")
st.markdown("Dynamic Job Description-driven candidate evaluation with evidence grounding and explainable scoring.")

tab_jd, tab_screening, tab_reports = st.tabs([
    "1. Job Description & Criteria",
    "2. Resume Screening & Processing",
    "3. Candidate Comparison & Reports"
])

# =============================================================
# SECTION 1: JOB DESCRIPTION
# =============================================================
with tab_jd:
    st.header("Step 1: Define & Confirm Job Description")
    st.markdown("Enter your **Job Position Title** and **Job Description**. Evaluation criteria and relative weights will be dynamically generated.")

    with st.expander("💡 Optional: Load a Pre-defined Sample JD Template", expanded=False):
        sample_options = [
            "-- Select to populate --",
            "Senior AI & Machine Learning Engineer",
            "Senior Backend & DevOps Engineer",
            "Frontend Web Developer",
            "Junior Data Science Intern (Skills & Projects Only)",
            "Growth Marketing & Sales Specialist"
        ]
        sample_pick = st.selectbox("Sample Roles", sample_options, index=0)
        if sample_pick != "-- Select to populate --":
            if sample_pick == "Senior AI & Machine Learning Engineer":
                st.session_state.custom_jd_title = "Senior AI & Machine Learning Engineer"
                st.session_state.custom_jd_text = """About the Role:
We are looking for a Senior AI/ML Engineer to lead our NLP initiatives and build scalable inference services.

Requirements:
• 4+ years of professional experience in Python development
• Strong expertise in Natural Language Processing (NLP) or Deep Learning
• Hands-on proficiency with PyTorch or TensorFlow frameworks
• Experience deploying microservices with Docker and Kubernetes
• Relational database experience with PostgreSQL
• Bachelor's or Master's in Computer Science or related field

Preferred Qualifications:
• Experience developing high-throughput APIs with FastAPI
• CI/CD pipeline automation with GitHub Actions
• Experience in Scikit-Learn predictive modeling"""
            elif sample_pick == "Senior Backend & DevOps Engineer":
                st.session_state.custom_jd_title = "Senior Backend & DevOps Engineer"
                st.session_state.custom_jd_text = """Requirements:
• 4+ years experience in Python backend development
• Microservices architecture with FastAPI or Django
• Containerization and orchestration using Docker and Kubernetes
• Relational database schema design and optimization using PostgreSQL
• CI/CD automation and Git version control

Preferred:
• Experience with Redis caching
• Linux systems administration"""
            elif sample_pick == "Frontend Web Developer":
                st.session_state.custom_jd_title = "Frontend Web Developer"
                st.session_state.custom_jd_text = """Requirements:
• 3+ years experience building web applications using React and TypeScript
• HTML5, CSS3, and modern JavaScript standards
• State management and component architecture
• Version control with Git

Preferred:
• Experience working with Agile Scrum methodologies and JIRA
• Familiarity with Webpack or Vite"""
            elif sample_pick == "Junior Data Science Intern (Skills & Projects Only)":
                st.session_state.custom_jd_title = "Data Science Intern"
                st.session_state.custom_jd_text = """About the Role:
Looking for an enthusiastic Data Science Intern to work on predictive models and exploratory data analysis.

Requirements:
• Hands-on programming proficiency in Python
• Experience with Pandas and NumPy for data manipulation
• Applied Machine Learning projects using Scikit-Learn
• Data visualization with Matplotlib or Seaborn
• SQL fundamentals

Nice to Have:
• Familiarity with Git version control
• Jupyter Notebook experimentation"""
            elif sample_pick == "Growth Marketing & Sales Specialist":
                st.session_state.custom_jd_title = "Growth Marketing & Sales Specialist"
                st.session_state.custom_jd_text = """Requirements:
• 3+ years experience in Search Engine Optimization (SEO) and technical audits
• Hands-on Google Ads PPC campaign management
• B2B Sales pipeline management using Salesforce CRM
• Lead Generation and prospecting

Preferred:
• Content marketing strategy
• Analytics tracking and conversion rate optimization"""

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        job_title_input = st.text_input(
            "📌 Job Position / Title",
            value=st.session_state.custom_jd_title,
            placeholder="e.g. Senior Machine Learning Engineer, Marketing Lead, Data Science Intern",
            help="Enter the exact job title for this screening session."
        )
    with col_t2:
        if st.session_state.confirmed_jd:
            st.success(f"Active Role: **{st.session_state.confirmed_jd.title}** ({len(st.session_state.confirmed_jd.requirements)} criteria)")

    input_method = st.radio("Choose Input Method", ["Type / Paste Job Description", "Upload Job Description File (PDF/TXT)"], horizontal=True)

    if input_method == "Type / Paste Job Description":
        jd_body_text = st.text_area(
            "Job Description Content",
            value=st.session_state.custom_jd_text,
            height=240,
            placeholder="Paste your Job Description here..."
        )
    else:
        uploaded_jd = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"])
        if uploaded_jd:
            if uploaded_jd.type == "application/pdf":
                extracted = resume_extractor.extract_from_pdf(uploaded_jd, uploaded_jd.name)
                jd_body_text = extracted.cleaned_text
            else:
                jd_body_text = uploaded_jd.read().decode("utf-8")
            st.session_state.custom_jd_text = jd_body_text
            st.success(f"Loaded content from {uploaded_jd.name}")
        else:
            jd_body_text = st.session_state.custom_jd_text

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        extract_clicked = st.button("🔍 Analyze JD & Generate Criteria", type="primary")

    if extract_clicked:
        if not jd_body_text.strip():
            st.error("Please provide Job Description text or upload a file first.")
        else:
            active_title = job_title_input.strip() if job_title_input.strip() else "Custom Job Position"
            st.session_state.custom_jd_title = active_title
            st.session_state.custom_jd_text = jd_body_text

            with st.spinner("Analyzing Job Description and dynamically constructing evaluation strategy..."):
                if use_groq and groq_manager.is_available:
                    if hasattr(groq_evaluator, "tracker"):
                        groq_evaluator.tracker.record_jd_analysis_call()
                parsed_jd = jd_parser.parse(
                    jd_body_text,
                    jd_id="custom_jd",
                    default_title=active_title,
                    use_llm=use_groq
                )
                st.session_state.confirmed_jd = parsed_jd
                st.session_state.extracted_requirements = [r.to_dict() for r in parsed_jd.requirements]
                st.session_state.screening_reports = []
                st.success(f"Generated {len(parsed_jd.requirements)} criteria with dynamic weights for '{parsed_jd.title}'!")

    # Display Dynamic Evaluation Strategy Matrix
    if st.session_state.confirmed_jd:
        active_jd = st.session_state.confirmed_jd
        st.markdown("---")

        # ── Shared Evaluation Strategy Card (Reused across all candidates) ──
        strategy = getattr(active_jd, "evaluation_strategy", None)
        if strategy:
            with st.container(border=True):
                st.subheader(f"🎯 Shared JD Evaluation Strategy: {strategy.job_domain}")
                st.caption("Generated once and reused across every candidate evaluation for 1-to-many screening consistency.")
                if strategy.role_summary:
                    st.info(f"**Mission & Role Summary:** {strategy.role_summary}")
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    st.markdown("**Essential Practical Capabilities:**")
                    for cap in strategy.essential_capabilities:
                        st.markdown(f"- {cap}")
                    st.markdown("**Relevant Core Skills:**")
                    st.markdown(", ".join([f"`{s}`" for s in strategy.relevant_skills]) if strategy.relevant_skills else "—")
                with s_c2:
                    st.markdown("**Expected Practical Implementations:**")
                    for ep in strategy.expected_practical_knowledge:
                        st.markdown(f"- {ep}")
                    st.markdown("**Desirable Qualifications & Extras:**")
                    for dq in strategy.desirable_qualifications:
                        st.markdown(f"- {dq}")

        st.subheader("📋 Dynamic Evaluation Strategy Matrix")
        st.caption("Criteria and weights are automatically generated from the Job Description. Python validates and normalizes all weights.")

        # Matrix metrics
        matrix_val = active_jd.validate_matrix()
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Total Criteria", matrix_val["total_requirements"])
        m_col2.metric("Mandatory", matrix_val["mandatory_count"])
        m_col3.metric("Preferred", matrix_val["preferred_count"])
        m_col4.metric("Optional", matrix_val["optional_count"])
        has_exp = matrix_val["has_experience_requirement"]
        m_col5.metric("Min Experience", "Specified in JD" if has_exp else "None Required")

        # Table display
        matrix_rows = []
        for r in active_jd.requirements:
            logic_badge = r.matching_logic.value
            if r.compound_components:
                logic_badge += f" ({', '.join(r.compound_components)})"
            matrix_rows.append({
                "ID": r.id,
                "Priority": r.priority.value,
                "Type": r.req_type.value,
                "Logic": logic_badge,
                "Dynamic Weight (%)": f"{r.weight * 100:.1f}%",
                "Requirement": r.text,
                "Accepted Equivalents": ", ".join(r.accepted_equivalents) if r.accepted_equivalents else "—"
            })

        st.dataframe(
            pd.DataFrame(matrix_rows),
            column_config={
                "ID": st.column_config.TextColumn("ID", width="small"),
                "Priority": st.column_config.TextColumn("Priority", width="small"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Logic": st.column_config.TextColumn("Logic", width="medium"),
                "Dynamic Weight (%)": st.column_config.TextColumn("Weight", width="small"),
                "Requirement": st.column_config.TextColumn("Requirement Description", width="large"),
                "Accepted Equivalents": st.column_config.TextColumn("Equivalents", width="medium")
            },
            hide_index=True,
            width="stretch"
        )
        st.success("✔️ Ready for screening! Proceed to Step 2 to upload resumes.")

# =============================================================
# SECTION 2: RESUME SCREENING
# =============================================================
with tab_screening:
    st.header("Step 2: Upload Resumes & Run Screening")

    if not st.session_state.confirmed_jd:
        st.warning("⚠️ Please define and confirm a Job Description in Step 1 first.")
    else:
        st.info(f"Target Position: **{st.session_state.confirmed_jd.title}** ({len(st.session_state.confirmed_jd.requirements)} criteria)")

        st.markdown("### 📤 Upload Candidate Resumes")
        st.caption("Upload PDF resumes to screen against the confirmed Job Description.")

        uploaded_files = st.file_uploader(
            "Drop or select PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or multiple PDF resumes from your computer."
        )

        if uploaded_files:
            st.success(f"Selected **{len(uploaded_files)}** candidate resume(s):")
            files_summary = [{"Filename": f.name, "File Size (KB)": round(f.size / 1024, 1)} for f in uploaded_files]
            st.dataframe(pd.DataFrame(files_summary), width="stretch")

        with st.expander("Optional: Include Pre-Packaged Synthetic Test Resumes", expanded=False):
            if os.path.exists(TEST_RESUMES_DIR):
                test_files = [f for f in os.listdir(TEST_RESUMES_DIR) if f.endswith(".pdf")]
                selected_samples = st.multiselect("Select sample resumes to include (empty by default)", test_files, default=[])
            else:
                selected_samples = []

        st.markdown("")
        if st.button("🚀 Start Universal Screening", type="primary"):
            resumes_to_process = []

            if uploaded_files:
                for f in uploaded_files:
                    resumes_to_process.append((f.name, f))

            if selected_samples:
                for fname in selected_samples:
                    fpath = os.path.join(TEST_RESUMES_DIR, fname)
                    resumes_to_process.append((fname, open(fpath, "rb")))

            if not resumes_to_process:
                st.error("Please upload at least one candidate PDF resume to screen.")
            else:
                progress_bar = st.progress(0.0)
                status_box = st.empty()

                matching_engine.cache_jd(st.session_state.confirmed_jd)

                # Phase 1: Extract and pre-match all resumes deterministically
                candidate_preps = []
                for idx, (fname, f_obj) in enumerate(resumes_to_process):
                    status_box.markdown(f"**Extracting & Pre-matching ({idx+1}/{len(resumes_to_process)}):** `{fname}`...")
                    proc_resume = resume_extractor.extract_from_pdf(f_obj, filename=fname)

                    matching_engine.fuzzy_threshold = fuzzy_th
                    matching_engine.semantic_threshold = sem_th
                    matching_engine.semantic_partial_threshold = sem_part_th
                    matches = matching_engine.match_resume_to_jd(proc_resume, st.session_state.confirmed_jd)

                    eval_result = evaluator.evaluate(
                        matches,
                        resume=proc_resume,
                        jd=st.session_state.confirmed_jd
                    )
                    candidate_preps.append((proc_resume, matches, eval_result))
                    progress_bar.progress((idx + 1) / (len(resumes_to_process) * 2))

                # Phase 2: Independent candidate evaluations
                reports = []
                if use_groq and groq_manager.is_available:
                    status_box.markdown(f"**Running Independent LLM Evaluations ({len(candidate_preps)} candidate(s), controlled concurrency)...**")
                    def _on_cand_done(done, total, c_fname):
                        status_box.markdown(f"**AI Screening ({done}/{total}):** Completed `{c_fname}`")
                        progress_bar.progress(0.5 + ((done / max(1, total)) * 0.5))

                    batch_results = groq_evaluator.evaluate_candidates_batch(
                        jd=st.session_state.confirmed_jd,
                        candidates=candidate_preps,
                        max_workers=2,
                        progress_callback=_on_cand_done
                    )

                    for i, (proc_resume, final_eval, llm_eval) in enumerate(batch_results):
                        matches = candidate_preps[i][1]
                        rep = ScreeningReportGenerator.generate_candidate_report(
                            proc_resume, final_eval, matches, st.session_state.confirmed_jd, llm_evaluation=llm_eval
                        )
                        rep["evaluation_summary"]["recommendation"] = normalize_recommendation(rep["evaluation_summary"]["recommendation"])
                        reports.append(rep)
                else:
                    for proc_resume, matches, eval_result in candidate_preps:
                        rep = ScreeningReportGenerator.generate_candidate_report(
                            proc_resume, eval_result, matches, st.session_state.confirmed_jd, llm_evaluation=None
                        )
                        rep["evaluation_summary"]["recommendation"] = normalize_recommendation(rep["evaluation_summary"]["recommendation"])
                        reports.append(rep)
                    progress_bar.progress(1.0)

                st.session_state.screening_reports = reports
                comparison_df = ScreeningReportGenerator.generate_comparison_dataframe(reports)
                if not comparison_df.empty and "Recommendation" in comparison_df.columns:
                    comparison_df["Recommendation"] = comparison_df["Recommendation"].apply(normalize_recommendation)
                st.session_state.comparison_df = comparison_df

                call_stats = groq_evaluator.get_call_stats(len(reports))
                st.session_state.groq_call_stats = call_stats
                llm_note = f" (Groq Total API Calls: {call_stats['total_calls']} | Cache Hits: {call_stats['cache_hits']})" if use_groq and groq_manager.is_available else ""
                status_box.success(f"🎉 Successfully screened {len(reports)} candidate(s){llm_note}! View results in Step 3.")

# =============================================================
# SECTION 3: CANDIDATE COMPARISON & REPORTS
# =============================================================
with tab_reports:
    st.header("Step 3: Candidate Comparison & Explainable Reports")

    if not st.session_state.screening_reports:
        st.info("No candidates screened yet. Upload your resumes and click 'Start Universal Screening' in Step 2.")
    else:
        reports = st.session_state.screening_reports
        df = st.session_state.comparison_df

        for r in reports:
            r["evaluation_summary"]["recommendation"] = normalize_recommendation(r["evaluation_summary"].get("recommendation", ""))

        if not df.empty and "Recommendation" in df.columns:
            df["Recommendation"] = df["Recommendation"].apply(normalize_recommendation)

        # ── API Usage & Reliability Visibility Card ────────────────────────
        call_stats = st.session_state.get("groq_call_stats")
        if call_stats:
            with st.container(border=True):
                st.subheader("⚡ Groq API Usage & Reliability Visibility")
                st.caption("Live monitoring of API calls, individual resume evaluations, cache hits, retries, and errors.")
                u1, u2, u3, u4, u5, u6, u7 = st.columns(7)
                u1.metric("🔍 JD Analysis", call_stats.get("jd_analysis_calls", 0))
                u2.metric("📄 Resume Calls", call_stats.get("resume_eval_calls", 0))
                u3.metric("✅ Success", call_stats.get("successful_evaluations", 0))
                u4.metric("❌ Failures", call_stats.get("failed_evaluations", 0))
                u5.metric("🔄 Retries", call_stats.get("retry_count", 0))
                u6.metric("⚡ Cache Hits", call_stats.get("cache_hits", 0))
                u7.metric("📊 Total API Usage", call_stats.get("total_calls", 0))

        sel_count = sum(1 for r in reports if r["evaluation_summary"]["recommendation"] == REC_SELECTION)
        rev_count = sum(1 for r in reports if r["evaluation_summary"]["recommendation"] == REC_MANUAL_REVIEW)
        not_count = sum(1 for r in reports if r["evaluation_summary"]["recommendation"] == REC_NOT_RECOMMENDED)

        rec_col1, rec_col2, rec_col3 = st.columns(3)
        rec_col1.metric("🟢 Recommended for Selection", f"{sel_count} candidates")
        rec_col2.metric("🟡 Manual Review Required", f"{rev_count} candidates")
        rec_col3.metric("🔴 Not Recommended", f"{not_count} candidates")

        st.subheader("📊 Candidate Comparison Overview")

        table_rows = []
        for r in reports:
            ev = r["evaluation_summary"]
            proc = r.get("processing_status", {})
            exp = r.get("verified_experience", {})
            rec = normalize_recommendation(ev.get("recommendation", ""))
            if rec == REC_SELECTION:
                badge = "🟢 Recommended for Selection"
            elif rec == REC_MANUAL_REVIEW:
                badge = "🟡 Manual Review Required"
            else:
                badge = "🔴 Not Recommended"

            tenure_val = exp.get("verified_years", ev.get("verified_tenure_years", 0.0))
            tenure_display = f"{tenure_val:.1f} yrs" if tenure_val > 0 else "0.0 yrs"

            llm_sc = ev.get("llm_score")
            llm_sc_str = f"{llm_sc:.1f}" if llm_sc is not None else "N/A"
            llm_sugg_str = ev.get("llm_suggested_recommendation") or "N/A"

            table_rows.append({
                "Selection Status": badge,
                "Candidate Name": r["candidate_name"],
                "Final Score": ev["overall_score"],
                "Mandatory Coverage": f"{ev.get('essential_coverage_pct', 0.0):.1f}%",
                "Practical Depth": f"{ev.get('practical_evidence_score', 0.0):.1f}%",
                "Project Depth": f"{ev.get('project_depth_score', 0.0):.1f}%",
                "Preferred Coverage": f"{ev.get('preferred_coverage_pct', 0.0):.1f}%",
                "Verified Tenure": tenure_display,
                "AI Validated Assessment": llm_sc_str,
                "Decision Reason": ev.get("decision_rationale", ""),
                "Conf": f"{int(proc.get('extraction_confidence', 1.0) * 100)}%",
                "Filename": r["filename"]
            })

        display_df = pd.DataFrame(table_rows)
        if not display_df.empty:
            display_df = display_df.sort_values(by="Final Score", ascending=False)

        st.dataframe(
            display_df,
            column_config={
                "Selection Status": st.column_config.TextColumn("Selection Status", width="medium"),
                "Candidate Name": st.column_config.TextColumn("Candidate Name", width="medium"),
                "Final Score": st.column_config.ProgressColumn("Final Score (0-100)", format="%.1f", min_value=0.0, max_value=100.0),
                "Mandatory Coverage": st.column_config.TextColumn("Mandatory", width="small"),
                "Practical Depth": st.column_config.TextColumn("Practical", width="small"),
                "Project Depth": st.column_config.TextColumn("Projects", width="small"),
                "Preferred Coverage": st.column_config.TextColumn("Preferred", width="small"),
                "Verified Tenure": st.column_config.TextColumn("Tenure", width="small"),
                "AI Validated Assessment": st.column_config.TextColumn("AI Validated", width="small"),
                "Decision Reason": st.column_config.TextColumn("Decision Rationale", width="large"),
                "Conf": st.column_config.TextColumn("Conf"),
                "Filename": st.column_config.TextColumn("Resume File")
            },
            hide_index=True,
            width="stretch"
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_bytes = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Comparison (CSV)",
                data=csv_bytes,
                file_name="candidate_comparison.csv",
                mime="text/csv"
            )
        with col_dl2:
            json_bytes = json.dumps(reports, indent=2).encode('utf-8')
            st.download_button(
                label="📥 Download All Detailed Reports (JSON)",
                data=json_bytes,
                file_name="screening_reports.json",
                mime="application/json"
            )

        st.markdown("---")
        st.subheader("📋 Candidate Decision Reasons & Detailed Breakdown")
        st.caption("Review the exact rationale, strengths, gaps, and timeline for each candidate below.")

        for idx, cand_report in enumerate(reports):
            ev_summary = cand_report["evaluation_summary"]
            proc_status = cand_report.get("processing_status", {})
            exp_info = cand_report.get("verified_experience", {})
            cand_name = cand_report["candidate_name"]
            rec = normalize_recommendation(ev_summary.get("recommendation", ""))

            rec_icon = {
                REC_SELECTION: "🟢",
                REC_MANUAL_REVIEW: "🟡",
                REC_NOT_RECOMMENDED: "🔴"
            }.get(rec, "⚪")

            with st.container(border=True):
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    st.markdown(f"### {rec_icon} {cand_name} — **{rec}**")
                    st.caption(f"📄 Resume: `{cand_report['filename']}`")
                with col_h2:
                    st.metric("Overall Score", f"{ev_summary['overall_score']:.1f}/100")

                st.markdown("#### 🎯 Why was this decision made?")
                decision_text = ev_summary.get("decision_rationale", "No rationale recorded.")
                if rec == REC_SELECTION:
                    st.success(f"**Selection Rationale:** {decision_text}")
                elif rec == REC_MANUAL_REVIEW:
                    st.warning(f"**Manual Review Reason:** {decision_text}")
                else:
                    st.error(f"**Not Recommended Reason:** {decision_text}")

                llm_info = cand_report.get("llm_evaluation") or {}
                if llm_info and llm_info.get("llm_applied"):
                    groq_sugg = llm_info.get("llm_suggested_recommendation") or llm_info.get("recommendation", "")
                    st.info(f"🤖 **Groq AI Qualitative Assessment:** Suggested **{groq_sugg}** (Python-Validated Score: `{llm_info.get('overall_score', 0):.1f}/100`)")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Mandatory Match", f"{ev_summary.get('essential_coverage_pct', 0.0):.1f}%")
                m2.metric("Practical Depth", f"{ev_summary.get('practical_evidence_score', 0.0):.1f}%")
                m3.metric("Project Depth", f"{ev_summary.get('project_depth_score', 0.0):.1f}%")
                m4.metric("Preferred Match", f"{ev_summary.get('preferred_coverage_pct', 0.0):.1f}%")
                llm_sc_val = ev_summary.get('llm_score')
                llm_disp = f"{llm_sc_val:.1f}/100" if llm_sc_val is not None else "N/A"
                m5.metric("🤖 AI Validated Score", llm_disp)

                if ev_summary.get("score_breakdown_formula"):
                    st.caption(f"📐 **Evaluation Breakdown:** `{ev_summary['score_breakdown_formula']}`")

                if proc_status.get("is_low_confidence"):
                    st.warning(
                        f"⚠️ **Low Extraction Confidence ({int(proc_status.get('extraction_confidence', 0.0)*100)}%):** "
                        f"Resume text volume was minimal or scanned ({'; '.join(proc_status.get('confidence_reasons', []))}). "
                        f"Routed to Manual Review to avoid accidental rejection."
                    )

                transferable = ev_summary.get("transferable_skills", [])
                if transferable:
                    st.info(
                        "🔄 **Transferable Skills Identified:** " +
                        "; ".join([f"`{t['required']}` ➜ candidate has `{t['candidate_has']}`" for t in transferable])
                    )

                if exp_info.get("timeline_summary") and exp_info["timeline_summary"] != "Unverified":
                    st.markdown(f"📅 **Verified Employment Timeline:** {exp_info['timeline_summary']}")
                    periods = exp_info.get("employment_periods", [])
                    if periods:
                        p_cols = st.columns(min(len(periods), 4))
                        for p_idx, p in enumerate(periods):
                            with p_cols[p_idx % len(p_cols)]:
                                st.caption(f"**Period {p_idx+1}:** {p['start']:.1f} – {p['end']:.1f} ({p['duration_years']} yrs)")

                if cand_report.get("llm_evaluation") and cand_report["llm_evaluation"].get("llm_applied"):
                    llm_info = cand_report["llm_evaluation"]
                    col_st, col_wk = st.columns(2)
                    with col_st:
                        st.markdown("##### 🌟 Key Strengths")
                        strengths = llm_info.get("strengths", [])
                        if strengths:
                            for s in strengths:
                                st.markdown(f"- {s}")
                        else:
                            st.caption("No specific strengths highlighted.")

                    with col_wk:
                        st.markdown("##### ⚠️ Key Gaps / Areas to Review")
                        weaknesses = llm_info.get("weaknesses", [])
                        if weaknesses:
                            for w in weaknesses:
                                st.markdown(f"- {w}")
                        else:
                            st.caption("No critical gaps identified.")

                llm_eval_obj = cand_report.get("llm_evaluation") or {}
                skills_ass = llm_eval_obj.get("skills_assessment") or {}
                if skills_ass and any(skills_ass.get(k) for k in ["direct_evidence", "reasonable_inferences", "genuine_gaps", "unknown_information"]):
                    with st.expander("🧠 Qualitative Evidence & Capability Breakdown", expanded=False):
                        qc1, qc2 = st.columns(2)
                        with qc1:
                            st.markdown("🎯 **Direct Evidence (Demonstrated in Experience/Projects):**")
                            dev = skills_ass.get("direct_evidence", [])
                            if dev:
                                for d in dev:
                                    st.markdown(f"- {d}")
                            else:
                                st.caption("No explicit direct evidence isolated.")

                            st.markdown("💡 **Reasonable Inferences (Related Tech Stack):**")
                            rinf = skills_ass.get("reasonable_inferences", [])
                            if rinf:
                                for r in rinf:
                                    st.markdown(f"- {r}")
                            else:
                                st.caption("None recorded.")

                        with qc2:
                            st.markdown("⚠️ **Genuine Gaps (Critical Capabilities Missing):**")
                            gaps = skills_ass.get("genuine_gaps", [])
                            if gaps:
                                for g in gaps:
                                    st.markdown(f"- {g}")
                            else:
                                st.caption("No critical competency gaps identified.")

                            st.markdown("❓ **Unknown Information (Unmentioned Soft Skills / Behavioral):**")
                            unks = skills_ass.get("unknown_information", [])
                            if unks:
                                for u in unks:
                                    st.markdown(f"- {u}")
                            else:
                                st.caption("No unknown items recorded.")

                with st.expander("🔎 View Raw Requirement-by-Requirement Evidence", expanded=False):
                    badge_colors = {
                        MatchClassification.EXACT_MATCH.value: "green",
                        MatchClassification.STRONG_SEMANTIC_MATCH.value: "blue",
                        MatchClassification.FUZZY_MATCH.value: "violet",
                        MatchClassification.RELATED_SKILL_NOT_EQUIVALENT.value: "orange",
                        MatchClassification.PARTIAL_EVIDENCE.value: "yellow",
                        MatchClassification.NEEDS_MANUAL_REVIEW.value: "orange",
                        MatchClassification.NOT_FOUND.value: "red"
                    }
                    for item in cand_report["detailed_requirements"]:
                        badge_color = badge_colors.get(item["classification"], "gray")
                        st.markdown(
                            f"**:{badge_color}[{item['classification']}]** | "
                            f"**{item['priority']}** (Weight: {item.get('weight', 0)*100:.1f}%): {item['requirement_text']}"
                        )
                        if item["evidence_sentence"]:
                            st.markdown(f"> *\"{item['evidence_sentence']}\"*")
                        else:
                            st.caption("No supporting evidence found in resume.")
                        st.divider()
