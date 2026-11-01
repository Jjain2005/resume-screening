import os
import json
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from modules.knowledge_base.taxonomy import SkillTaxonomy
from modules.resume_processing.extractor import ResumeExtractor
from modules.resume_processing.models import ProcessedResume, SectionType
from modules.jd_processing.parser import JobDescriptionParser
from modules.jd_processing.models import JobDescription, RequirementItem, RequirementType, RequirementPriority
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
    page_title="Resume Screening System",
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
jd_parser = JobDescriptionParser(taxonomy)

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
    st.caption("Adjust candidate scoring profile and weights.")

    role_mode = st.radio(
        "🎯 Hiring Role Profile",
        options=[
            "🎓 Intern / Fresher (0% Experience Weight — Skills & Projects Focus)",
            "💼 Experienced Professional (Balanced 5-Factor Scoring)"
        ],
        index=0,
        help="Intern mode gives 0% weight to working years, evaluating candidates purely on essential skills, project depth, and practical code demonstration."
    )

    is_intern = "Intern" in role_mode
    default_w_ess = 0.25 if is_intern else 0.30
    default_w_llm = 0.30 if is_intern else 0.20
    default_w_prac = 0.20 if is_intern else 0.15
    default_w_proj = 0.15 if is_intern else 0.10
    default_w_pref = 0.10 if is_intern else 0.10
    default_w_exp = 0.00 if is_intern else 0.15

    with st.expander("Scoring Weights (Tunable)", expanded=True):
        st.caption(f"Active Profile: **{'🎓 Intern / Fresher' if is_intern else '💼 Experienced'}**")
        w_ess = st.slider("Essential Skills Weight", 0.05, 0.70, default_w_ess, 0.05, key=f"w_ess_{is_intern}")
        w_llm = st.slider(
            "Groq AI Assessment Weight",
            0.0, 0.60, default_w_llm, 0.05,
            key=f"w_llm_{is_intern}",
            help="Holistic LLM qualitative score evaluating candidate depth and contextual evidence."
        )
        w_prac = st.slider("Practical Evidence Weight", 0.0, 0.50, default_w_prac, 0.05, key=f"w_prac_{is_intern}")
        w_proj = st.slider("Project Implementation Depth", 0.0, 0.40, default_w_proj, 0.05, key=f"w_proj_{is_intern}")
        w_pref = st.slider("Preferred Skills Weight", 0.0, 0.40, default_w_pref, 0.05, key=f"w_pref_{is_intern}")
        w_exp = st.slider(
            "Relevant Experience Duration",
            0.0, 0.40, default_w_exp, 0.05,
            key=f"w_exp_{is_intern}",
            help="Set to 0.0 for interns/freshers where working years do not apply."
        )

        total_weight = w_ess + w_llm + w_prac + w_proj + w_pref + w_exp
        if abs(total_weight - 1.0) > 0.02:
            st.warning(f"Weights sum to {total_weight:.2f} (Recommended: 1.00)")
        else:
            st.success("Weights normalized to 1.00")

    with st.expander("Matching Thresholds", expanded=False):
        fuzzy_th = st.slider("Fuzzy Match Threshold (%)", 70, 95, 85, 1)
        sem_th = st.slider("Semantic Match Threshold (Cosine)", 0.55, 0.85, 0.70, 0.02)
        sem_part_th = st.slider("Partial Semantic Threshold", 0.40, 0.65, 0.52, 0.02)

    with st.expander("🤖 Groq AI Evaluation (Optional)", expanded=True):
        groq_manager = GroqClientManager()
        use_groq = st.toggle(
            "Enable Groq Deep Analysis",
            value=groq_manager.is_available,
            help="Enables LLM qualitative evaluation of project context, internships, candidate strengths, and gaps without hallucination."
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

    scoring_weights = ScoringWeights(
        essential_skills_weight=w_ess / max(total_weight, 0.001),
        preferred_skills_weight=w_pref / max(total_weight, 0.001),
        practical_evidence_weight=w_prac / max(total_weight, 0.001),
        relevant_experience_weight=w_exp / max(total_weight, 0.001),
        project_depth_weight=w_proj / max(total_weight, 0.001),
        llm_evaluation_weight=w_llm / max(total_weight, 0.001)
    )
    evaluator = CandidateEvaluator(scoring_weights)

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base Info")
    st.write(f"**Categories:** {len(taxonomy.categories)}")
    st.write(f"**Canonical Skills:** {len(taxonomy.skills)}")
    st.write(f"**Indexed Terms / Aliases:** {len(taxonomy.get_all_lookup_terms())}")
    if taxonomy.unknown_skills:
        st.caption(f"Flagged {len(taxonomy.unknown_skills)} unknown terms for review.")

# -------------------------------------------------------------
# Main Navigation Tabs (3 Core Workflow Sections)
# -------------------------------------------------------------
st.title("Intelligent Resume Screening Prototype")
st.markdown("Explainable, evidence-grounded candidate matching against Job Descriptions.")

tab_jd, tab_screening, tab_reports = st.tabs([
    "1. Job Description & Requirements",
    "2. Resume Screening & Processing",
    "3. Candidate Comparison & Reports"
])

# =============================================================
# SECTION 1: JOB DESCRIPTION
# =============================================================
with tab_jd:
    st.header("Step 1: Define & Confirm Job Description")
    st.markdown("Enter your **Job Position Title** and **Job Description**. You can paste text or upload a document.")

    # Optional quick template loader (collapsed so it doesn't get in the way of custom JDs)
    with st.expander("💡 Optional: Load a Pre-defined Sample JD Template", expanded=False):
        sample_options = [
            "-- Select to populate --",
            "Senior AI & Machine Learning Engineer",
            "Senior Backend & DevOps Engineer",
            "Frontend Web Developer",
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

    # 1. Job Title Input
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        job_title_input = st.text_input(
            "📌 Job Position / Title",
            value=st.session_state.custom_jd_title,
            placeholder="e.g. Senior Machine Learning Engineer, Marketing Lead, Accountant",
            help="Enter the exact job title for this screening session."
        )
    with col_t2:
        if st.session_state.confirmed_jd:
            st.success(f"Active Confirmed Role: **{st.session_state.confirmed_jd.title}**")

    # 2. JD Text or File Upload
    input_method = st.radio("Choose Input Method", ["Type / Paste Job Description", "Upload Job Description File (PDF/TXT)"], horizontal=True)

    if input_method == "Type / Paste Job Description":
        jd_body_text = st.text_area(
            "Job Description Content",
            value=st.session_state.custom_jd_text,
            height=240,
            placeholder="Paste your Job Description here...\n\nExample:\nRequirements:\n• 3+ years experience with Python and SQL\n• Experience with Docker and APIs\n\nPreferred:\n• Familiarity with AWS and CI/CD"
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
        extract_clicked = st.button("🔍 Extract Requirements", type="primary")

    if extract_clicked:
        if not jd_body_text.strip():
            st.error("Please provide Job Description text or upload a file first.")
        else:
            active_title = job_title_input.strip() if job_title_input.strip() else "Custom Job Position"
            st.session_state.custom_jd_title = active_title
            st.session_state.custom_jd_text = jd_body_text

            parsed_jd = jd_parser.parse(
                jd_body_text,
                jd_id="custom_jd",
                default_title=active_title
            )
            st.session_state.extracted_requirements = [r.to_dict() for r in parsed_jd.requirements]
            st.success(f"Extracted {len(parsed_jd.requirements)} requirements for position: '{active_title}'")

    # 3. HR Review, Edit & Add Requirements
    if st.session_state.extracted_requirements:
        st.markdown("---")
        st.subheader("📋 Structured Requirements (Review & Edit)")
        st.info("You can edit priorities, add new requirements, or delete rows before confirming.")

        # Manual Requirement Adder
        with st.expander("➕ Add New Custom Requirement", expanded=False):
            c_add1, c_add2, c_add3, c_add4 = st.columns([3, 1, 1, 1])
            with c_add1:
                new_req_text = st.text_input("Requirement Text", placeholder="e.g. 3+ years experience with Kubernetes clusters")
            with c_add2:
                new_req_prio = st.selectbox("Priority", ["Required", "Preferred"])
            with c_add3:
                new_req_type = st.selectbox("Type", ["Skill", "Experience", "Education", "Certification", "Responsibility"])
            with c_add4:
                st.write("")
                st.write("")
                if st.button("Add to List"):
                    if new_req_text.strip():
                        canon = taxonomy.resolve_canonical(new_req_text.strip())
                        new_item = {
                            "id": f"req_custom_{len(st.session_state.extracted_requirements)+1}",
                            "text": new_req_text.strip(),
                            "priority": new_req_prio,
                            "req_type": new_req_type,
                            "canonical_skill": canon,
                            "weight": 1.5 if new_req_prio == "Required" else 1.0
                        }
                        st.session_state.extracted_requirements.append(new_item)
                        st.rerun()

        req_df = pd.DataFrame(st.session_state.extracted_requirements)
        edited_df = st.data_editor(
            req_df[["id", "priority", "req_type", "canonical_skill", "text"]],
            column_config={
                "id": st.column_config.TextColumn("ID", disabled=True),
                "priority": st.column_config.SelectboxColumn("Priority", options=["Required", "Preferred"], required=True),
                "req_type": st.column_config.SelectboxColumn("Type", options=["Skill", "Experience", "Education", "Certification", "Responsibility"]),
                "canonical_skill": st.column_config.TextColumn("Mapped Skill (Taxonomy)"),
                "text": st.column_config.TextColumn("Requirement Description", width="large")
            },
            num_rows="dynamic",
            width="stretch",
            key="req_editor"
        )

        st.markdown("")
        col_conf1, col_conf2 = st.columns([2, 3])
        with col_conf1:
            if st.button("✅ Confirm Job Description for Screening", type="primary"):
                active_title = job_title_input.strip() if job_title_input.strip() else "Job Position"
                confirmed_reqs = []
                for _, row in edited_df.iterrows():
                    item = RequirementItem(
                        id=str(row["id"]),
                        text=str(row["text"]),
                        req_type=RequirementType(row["req_type"]),
                        priority=RequirementPriority(row["priority"]),
                        canonical_skill=str(row["canonical_skill"]) if pd.notna(row["canonical_skill"]) and str(row["canonical_skill"]).strip() else None,
                        weight=1.5 if row["priority"] == "Required" else 1.0
                    )
                    confirmed_reqs.append(item)

                st.session_state.confirmed_jd = JobDescription(
                    id="confirmed_jd",
                    title=active_title,
                    raw_text=jd_body_text,
                    requirements=confirmed_reqs
                )
                st.session_state.screening_reports = []  # Clear previous screening on new JD
                st.success(f"🎉 Job Description '{active_title}' confirmed with {len(confirmed_reqs)} requirements! Go to Step 2 to screen resumes.")

        if st.session_state.confirmed_jd:
            st.success(f"✔️ Active Confirmed Position: **{st.session_state.confirmed_jd.title}** ({len(st.session_state.confirmed_jd.requirements)} criteria confirmed)")

# =============================================================
# SECTION 2: RESUME SCREENING
# =============================================================
with tab_screening:
    st.header("Step 2: Upload Resumes & Run Screening")

    if not st.session_state.confirmed_jd:
        st.warning("⚠️ Please confirm a Job Description in Step 1 first.")
    else:
        st.info(f"Target Position: **{st.session_state.confirmed_jd.title}** ({len(st.session_state.confirmed_jd.requirements)} requirements confirmed)")

        st.markdown("### 📤 Upload Candidate Resumes")
        st.caption("Upload multiple PDF resumes to screen against the confirmed Job Description.")

        uploaded_files = st.file_uploader(
            "Drop or select PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or multiple PDF resumes from your computer."
        )

        # Show uploaded resume files summary
        if uploaded_files:
            st.success(f"Selected **{len(uploaded_files)}** candidate resume(s) for screening:")
            files_summary = [{"Filename": f.name, "File Size (KB)": round(f.size / 1024, 1)} for f in uploaded_files]
            st.dataframe(pd.DataFrame(files_summary), width="stretch")

        # Optional collapsed expander only if user explicitly wants pre-packaged test resumes
        with st.expander("Optional: Include Pre-Packaged Synthetic Test Resumes", expanded=False):
            if os.path.exists(TEST_RESUMES_DIR):
                test_files = [f for f in os.listdir(TEST_RESUMES_DIR) if f.endswith(".pdf")]
                selected_samples = st.multiselect("Select sample resumes to include (empty by default)", test_files, default=[])
            else:
                selected_samples = []

        st.markdown("")
        if st.button("🚀 Start Screening Resumes", type="primary"):
            resumes_to_process = []

            # 1. User uploaded files
            if uploaded_files:
                for f in uploaded_files:
                    resumes_to_process.append((f.name, f))

            # 2. Selected sample files (if any were chosen)
            if selected_samples:
                for fname in selected_samples:
                    fpath = os.path.join(TEST_RESUMES_DIR, fname)
                    resumes_to_process.append((fname, open(fpath, "rb")))

            if not resumes_to_process:
                st.error("Please upload at least one candidate PDF resume to screen.")
            else:
                progress_bar = st.progress(0.0)
                status_box = st.empty()

                # Fast caching: precompute JD embeddings once for the entire screening batch
                matching_engine.cache_jd(st.session_state.confirmed_jd)
                groq_evaluator.reset_call_counter()

                reports = []
                for idx, (fname, f_obj) in enumerate(resumes_to_process):
                    status_box.markdown(f"**Screening ({idx+1}/{len(resumes_to_process)}):** `{fname}`...")

                    # 1. Extract PDF Text, Validate Confidence & Parse Timeline
                    proc_resume = resume_extractor.extract_from_pdf(f_obj, filename=fname)

                    # 2. Run Vectorized Hybrid Matching Engine
                    matching_engine.fuzzy_threshold = fuzzy_th
                    matching_engine.semantic_threshold = sem_th
                    matching_engine.semantic_partial_threshold = sem_part_th
                    matches = matching_engine.match_resume_to_jd(proc_resume, st.session_state.confirmed_jd)

                    # 3. Evaluate & Score (5-factor scoring + tenure + extraction safeguarding)
                    eval_result = evaluator.evaluate(
                        matches,
                        resume=proc_resume,
                        jd=st.session_state.confirmed_jd
                    )

                    # 3b. Optional Groq Contextual Evaluation (Integrated into Matrix with 30% Weight)
                    llm_eval = None
                    if use_groq and groq_manager.is_available:
                        with st.spinner(f"Running Groq AI contextual analysis for {fname}..."):
                            llm_eval = groq_evaluator.evaluate_candidate(
                                jd=st.session_state.confirmed_jd,
                                resume=proc_resume,
                                deterministic_matches=matches,
                                deterministic_eval=eval_result
                            )
                            if llm_eval and llm_eval.get("llm_applied"):
                                # Re-calculate candidate evaluation with Groq LLM integrated into the 30% matrix!
                                eval_result = evaluator.evaluate(
                                    matches,
                                    resume=proc_resume,
                                    jd=st.session_state.confirmed_jd,
                                    llm_score=llm_eval.get("overall_score"),
                                    llm_recommendation=llm_eval.get("llm_suggested_recommendation"),
                                    llm_synthesis=llm_eval.get("executive_summary")
                                )

                    # 4. Generate Report
                    rep = ScreeningReportGenerator.generate_candidate_report(
                        proc_resume, eval_result, matches, st.session_state.confirmed_jd, llm_evaluation=llm_eval
                    )
                    rep["evaluation_summary"]["recommendation"] = normalize_recommendation(rep["evaluation_summary"]["recommendation"])
                    reports.append(rep)
                    progress_bar.progress((idx + 1) / len(resumes_to_process))

                st.session_state.screening_reports = reports
                comparison_df = ScreeningReportGenerator.generate_comparison_dataframe(reports)
                if not comparison_df.empty and "Recommendation" in comparison_df.columns:
                    comparison_df["Recommendation"] = comparison_df["Recommendation"].apply(normalize_recommendation)
                st.session_state.comparison_df = comparison_df

                call_stats = groq_evaluator.get_call_stats(len(reports))
                st.session_state.groq_call_stats = call_stats
                llm_note = f" (Groq API Calls: {call_stats['total_calls']} | Avg: {call_stats['average_calls_per_candidate']}/cand)" if use_groq and groq_manager.is_available else ""
                status_box.success(f"🎉 Successfully screened {len(reports)} candidate(s){llm_note}! View detailed results in Step 3.")

# =============================================================
# SECTION 3: CANDIDATE COMPARISON & REPORTS
# =============================================================
with tab_reports:
    st.header("Step 3: Candidate Comparison & Explainable Reports")

    if not st.session_state.screening_reports:
        st.info("No candidates screened yet. Upload your resumes and click 'Start Screening' in Step 2.")
    else:
        reports = st.session_state.screening_reports
        df = st.session_state.comparison_df

        # Ensure all reports use the 3 standard recommendation categories
        for r in reports:
            r["evaluation_summary"]["recommendation"] = normalize_recommendation(r["evaluation_summary"].get("recommendation", ""))

        if not df.empty and "Recommendation" in df.columns:
            df["Recommendation"] = df["Recommendation"].apply(normalize_recommendation)

        # Top Executive Summary Metrics
        call_stats = st.session_state.get("groq_call_stats")
        if call_stats and call_stats.get("total_calls", 0) > 0:
            rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
            rec_col4.metric("🤖 Groq API Calls", f"{call_stats['total_calls']} (Avg {call_stats['average_calls_per_candidate']}/cand)")
        else:
            rec_col1, rec_col2, rec_col3 = st.columns(3)

        sel_count = sum(1 for r in reports if r["evaluation_summary"]["recommendation"] == REC_SELECTION)
        rev_count = sum(1 for r in reports if r["evaluation_summary"]["recommendation"] == REC_MANUAL_REVIEW)
        not_count = sum(1 for r in reports if r["evaluation_summary"]["recommendation"] == REC_NOT_RECOMMENDED)

        rec_col1.metric("🟢 Recommended for Selection", f"{sel_count} candidates")
        rec_col2.metric("🟡 Manual Review Required", f"{rev_count} candidates")
        rec_col3.metric("🔴 Not Recommended", f"{not_count} candidates")

        st.subheader("📊 Candidate Comparison Overview")

        # Build prominent comparison table with Selection Status up front
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
            if scoring_weights.relevant_experience_weight <= 0.01:
                tenure_display = "0% Weight (Intern Mode)"
            else:
                tenure_display = f"{tenure_val:.1f} yrs"

            llm_sc = ev.get("llm_score")
            llm_sc_str = f"{llm_sc:.1f}" if llm_sc is not None else "N/A"
            llm_sugg_str = ev.get("llm_suggested_recommendation") or "N/A"

            table_rows.append({
                "Selection Status": badge,
                "Candidate Name": r["candidate_name"],
                "Final Score": ev["overall_score"],
                "Groq LLM (30%)": llm_sc_str,
                "LLM Suggestion": llm_sugg_str,
                "Essential (25%)": f"{ev.get('essential_coverage_pct', 0.0):.1f}%",
                "Practical (20%)": f"{ev.get('practical_evidence_score', 0.0):.1f}%",
                "Projects (15%)": f"{ev.get('project_depth_score', 0.0):.1f}%",
                "Preferred (10%)": f"{ev.get('preferred_coverage_pct', 0.0):.1f}%",
                "Decision Reason": ev.get("decision_rationale", ""),
                "Extraction Conf": f"{int(proc.get('extraction_confidence', 1.0) * 100)}%",
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
                "Groq LLM (30%)": st.column_config.TextColumn("Groq AI (30%)", width="small"),
                "LLM Suggestion": st.column_config.TextColumn("LLM Suggestion", width="medium"),
                "Essential (25%)": st.column_config.TextColumn("Essential (25%)", width="small"),
                "Practical (20%)": st.column_config.TextColumn("Practical (20%)", width="small"),
                "Projects (15%)": st.column_config.TextColumn("Projects (15%)", width="small"),
                "Preferred (10%)": st.column_config.TextColumn("Preferred (10%)", width="small"),
                "Decision Reason": st.column_config.TextColumn("Decision Rationale", width="large"),
                "Extraction Conf": st.column_config.TextColumn("Conf"),
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

        # Candidate selector tabs or clean cards
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
                # Header row
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    st.markdown(f"### {rec_icon} {cand_name} — **{rec}**")
                    st.caption(f"📄 Resume: `{cand_report['filename']}`")
                with col_h2:
                    st.metric("Overall Score", f"{ev_summary['overall_score']:.1f}/100")

                # PROMINENT DECISION REASON CALLOUT
                st.markdown("#### 🎯 Why was this decision made?")
                decision_text = ev_summary.get("decision_rationale", "No rationale recorded.")
                if rec == REC_SELECTION:
                    st.success(f"**Selection Rationale:** {decision_text}")
                elif rec == REC_MANUAL_REVIEW:
                    st.warning(f"**Manual Review Reason:** {decision_text}")
                else:
                    st.error(f"**Not Recommended Reason:** {decision_text}")

                # Groq AI output suggestion banner (if available)
                llm_info = cand_report.get("llm_evaluation") or {}
                if llm_info and llm_info.get("llm_applied"):
                    groq_sugg = llm_info.get("llm_suggested_recommendation") or llm_info.get("recommendation", "")
                    st.info(f"🤖 **Groq AI Qualitative Output:** Suggested **{groq_sugg}** (Qualitative Score: `{llm_info.get('overall_score', 0):.1f}/100` — Weighted 30% in Matrix)")

                # 5 Pillars of the Candidate Matrix
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Essential Match", f"{ev_summary.get('essential_coverage_pct', 0.0):.1f}%", f"{scoring_weights.essential_skills_weight*100:.0f}% Wt")
                llm_sc_val = ev_summary.get('llm_score')
                llm_disp = f"{llm_sc_val:.1f}/100" if llm_sc_val is not None else "N/A"
                m2.metric("🤖 Groq AI Score", llm_disp, f"{scoring_weights.llm_evaluation_weight*100:.0f}% Wt")
                m3.metric("Practical Depth", f"{ev_summary.get('practical_evidence_score', 0.0):.1f}%", f"{scoring_weights.practical_evidence_weight*100:.0f}% Wt")
                m4.metric("Project Depth", f"{ev_summary.get('project_depth_score', 0.0):.1f}%", f"{scoring_weights.project_depth_weight*100:.0f}% Wt")
                m5.metric("Preferred Match", f"{ev_summary.get('preferred_coverage_pct', 0.0):.1f}%", f"{scoring_weights.preferred_skills_weight*100:.0f}% Wt")

                if ev_summary.get("score_breakdown_formula"):
                    st.caption(f"📐 **Matrix Breakdown:** `{ev_summary['score_breakdown_formula']}`")

                # Extraction Warning (if any)
                if proc_status.get("is_low_confidence"):
                    st.warning(
                        f"⚠️ **Low Extraction Confidence ({int(proc_status.get('extraction_confidence', 0.0)*100)}%):** "
                        f"Resume text volume was minimal or scanned ({'; '.join(proc_status.get('confidence_reasons', []))}). "
                        f"Routed to Manual Review to avoid accidental rejection."
                    )

                # Transferable Skills (if any)
                transferable = ev_summary.get("transferable_skills", [])
                if transferable:
                    st.info(
                        "🔄 **Transferable Skills Identified:** " +
                        "; ".join([f"`{t['required']}` ➜ candidate has `{t['candidate_has']}`" for t in transferable])
                    )

                # Verified Employment Timeline (Only highlight if weighted or real internships exist)
                if scoring_weights.relevant_experience_weight > 0.01 or exp_info.get("employment_periods"):
                    if exp_info.get("timeline_summary"):
                        st.markdown(f"📅 **Verified Employment Timeline:** {exp_info['timeline_summary']}")
                        periods = exp_info.get("employment_periods", [])
                        if periods:
                            p_cols = st.columns(min(len(periods), 4))
                            for p_idx, p in enumerate(periods):
                                with p_cols[p_idx % len(p_cols)]:
                                    st.caption(f"**Period {p_idx+1}:** {p['start']:.1f} – {p['end']:.1f} ({p['duration_years']} yrs)")
                else:
                    st.caption("ℹ️ **Intern / Skills-First Evaluation:** Prior work tenure is not required or evaluated for this role.")

                # Groq AI Recruiter Synthesis (if available)
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

                # Granular Requirement Evidence (Collapsed by default so it does not clutter)
                with st.expander("🔎 Optional: View Raw Requirement-by-Requirement Evidence", expanded=False):
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
                            f"**{item['priority']}**: {item['requirement_text']}"
                        )
                        if item["evidence_sentence"]:
                            st.markdown(f"> *\"{item['evidence_sentence']}\"*")
                        else:
                            st.caption("No supporting evidence found in resume.")
                        st.divider()
