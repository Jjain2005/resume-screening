import os
import sys
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from modules.knowledge_base.taxonomy import SkillTaxonomy
from modules.resume_processing.extractor import ResumeExtractor
from modules.jd_processing.parser import JobDescriptionParser
from modules.matching_engine.hybrid_engine import HybridMatchingEngine
from modules.scoring.evaluator import CandidateEvaluator, ScoringWeights
from modules.reporting.reporter import ScreeningReportGenerator

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    taxonomy_path = os.path.join(base_dir, "data", "knowledge_base", "skills_taxonomy.json")
    test_resumes_dir = os.path.join(base_dir, "data", "test_resumes")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing system components...")
    taxonomy = SkillTaxonomy(taxonomy_path)
    engine = HybridMatchingEngine(taxonomy)
    extractor = ResumeExtractor()
    parser = JobDescriptionParser(taxonomy)
    evaluator = CandidateEvaluator(ScoringWeights())

    jd_sample_text = """Senior AI & Machine Learning Engineer
Job Requirements:
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

    print("Parsing Job Description...")
    jd = parser.parse(jd_sample_text, jd_id="sample_ai_jd")

    pdf_files = sorted([f for f in os.listdir(test_resumes_dir) if f.endswith(".pdf")])
    print(f"Screening {len(pdf_files)} test resumes against JD: {jd.title}...")

    reports = []
    for pdf_name in pdf_files:
        pdf_path = os.path.join(test_resumes_dir, pdf_name)
        resume = extractor.extract_from_pdf(pdf_path, filename=pdf_name)
        matches = engine.match_resume_to_jd(resume, jd)
        evaluation = evaluator.evaluate(matches, resume=resume, jd=jd)
        rep = ScreeningReportGenerator.generate_candidate_report(resume, evaluation, matches, jd)
        reports.append(rep)
        print(f"  -> {rep['candidate_name']} ({pdf_name}): Score {evaluation.overall_score:.1f}/100 | {evaluation.recommendation.value}")

    # Generate comparison dataframe and export
    df = ScreeningReportGenerator.generate_comparison_dataframe(reports)
    
    json_path = os.path.join(output_dir, "example_screening_reports.json")
    csv_path = os.path.join(output_dir, "example_candidate_comparison.csv")
    
    ScreeningReportGenerator.export_reports_to_json(reports, json_path)
    ScreeningReportGenerator.export_comparison_to_csv(df, csv_path)

    print("\nSaved Deliverables:")
    print(f"  - Detailed Reports: {json_path}")
    print(f"  - Comparison CSV: {csv_path}")

if __name__ == "__main__":
    main()
