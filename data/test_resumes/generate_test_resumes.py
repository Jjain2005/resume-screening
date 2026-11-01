import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

RESUMES_DIR = os.path.dirname(os.path.abspath(__file__))

def create_resume_pdf(filename: str, name: str, title: str, summary: str, skills: list, experience: list, projects: list, education: list):
    pdf_path = os.path.join(RESUMES_DIR, filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        'HeaderName',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=4
    )
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading3'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#475569'),
        spaceAfter=12
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        borderPadding=(0, 0, 2, 0),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        spaceAfter=4
    )

    story = []
    # Header
    story.append(Paragraph(name, header_style))
    story.append(Paragraph(f"{title} | email: {name.lower().replace(' ', '.')}@example.com | (555) 019-2834", title_style))
    story.append(Spacer(1, 8))

    # Summary
    if summary:
        story.append(Paragraph("Professional Summary", section_heading))
        story.append(Paragraph(summary, body_style))
        story.append(Spacer(1, 6))

    # Skills
    if skills:
        story.append(Paragraph("Technical Skills", section_heading))
        skills_text = " • ".join(skills)
        story.append(Paragraph(skills_text, body_style))
        story.append(Spacer(1, 6))

    # Experience
    if experience:
        story.append(Paragraph("Professional Experience", section_heading))
        for exp in experience:
            story.append(Paragraph(f"<b>{exp['role']}</b> — {exp['company']} ({exp['dates']})", body_style))
            for pt in exp['points']:
                story.append(Paragraph(f"• {pt}", bullet_style))
            story.append(Spacer(1, 4))

    # Projects
    if projects:
        story.append(Paragraph("Projects", section_heading))
        for proj in projects:
            story.append(Paragraph(f"<b>{proj['name']}</b> ({proj['tech']})", body_style))
            for pt in proj['points']:
                story.append(Paragraph(f"• {pt}", bullet_style))
            story.append(Spacer(1, 4))

    # Education
    if education:
        story.append(Paragraph("Education", section_heading))
        for edu in education:
            story.append(Paragraph(f"<b>{edu['degree']}</b>, {edu['institution']} ({edu['year']})", body_style))

    doc.build(story)
    print(f"Generated: {pdf_path}")

def generate_all_test_resumes():
    # 1. Alex Morgan: Strong Match for Senior AI & NLP Engineer
    create_resume_pdf(
        filename="alex_morgan_ai_engineer.pdf",
        name="Alex Morgan",
        title="Senior AI & Machine Learning Engineer",
        summary="Senior AI Engineer with 6+ years of production experience in Natural Language Processing, deep learning architectures, and scalable cloud deployment. Proven track record designing transformer models and low-latency inference pipelines.",
        skills=[
            "Python", "Natural Language Processing (NLP)", "PyTorch", "Transformers",
            "Scikit-Learn", "FastAPI", "Docker", "Kubernetes", "CI/CD", "PostgreSQL"
        ],
        experience=[
            {
                "role": "Lead NLP Engineer",
                "company": "Cognitive AI Systems",
                "dates": "2021 - Present",
                "points": [
                    "Engineered multi-class NLP models to categorize customer sentiment using BERT and transformers, improving classification accuracy to 94%.",
                    "Architected low-latency REST APIs using FastAPI with Pydantic validation serving 5,000 QPS across distributed Kubernetes nodes.",
                    "Packaged backend services into Docker containers and managed cluster workloads with Kubernetes Helm charts.",
                    "Configured automated GitHub Actions workflows for testing, linting, and continuous deployment (CI/CD)."
                ]
            },
            {
                "role": "Machine Learning Engineer",
                "company": "DataSphere Analytics",
                "dates": "2018 - 2021",
                "points": [
                    "Developed and deployed a customer churn prediction pipeline using scikit-learn and Python that reduced churn by 12%.",
                    "Designed 3NF database schemas and optimized indexing in PostgreSQL to improve query times by 40%."
                ]
            }
        ],
        projects=[
            {
                "name": "Clinical Entity Extraction Engine",
                "tech": "PyTorch, Transformers, Spacy, Docker",
                "points": [
                    "Fine-tuned RoBERTa models using PyTorch on AWS GPU clusters to extract biomedical entities from clinical trial documents.",
                    "Delivered 88% F1 score across 15 entity categories and deployed model as an autoscaled microservice."
                ]
            }
        ],
        education=[
            {"degree": "Master of Science in Computer Science", "institution": "Georgia Institute of Technology", "year": "2018"},
            {"degree": "Bachelor of Science in Software Engineering", "institution": "University of Washington", "year": "2016"}
        ]
    )

    # 2. Jordan Lee: Partial Match (Backend & DevOps, Lacks NLP/PyTorch)
    create_resume_pdf(
        filename="jordan_lee_backend_dev.pdf",
        name="Jordan Lee",
        title="Senior Backend & DevOps Engineer",
        summary="Backend Software Engineer with 5 years of experience building resilient microservices in Python and managing cloud infrastructure with Docker and Kubernetes. Strong database optimization skills in PostgreSQL.",
        skills=[
            "Python", "Django", "FastAPI", "Docker", "Kubernetes", "PostgreSQL",
            "CI/CD", "Git", "Redis", "Linux"
        ],
        experience=[
            {
                "role": "Backend Engineer",
                "company": "CloudScale Platforms",
                "dates": "2020 - Present",
                "points": [
                    "Architected high-throughput REST APIs using Python and FastAPI handling 10M daily transactions.",
                    "Managed containerized application deployments using Docker and Kubernetes clusters across three availability zones.",
                    "Designed and maintained relational databases in PostgreSQL with complex partitioned queries.",
                    "Automated integration tests and deployment pipelines using GitHub Actions CI/CD."
                ]
            }
        ],
        projects=[
            {
                "name": "Distributed Task Scheduler",
                "tech": "Python, Redis, Docker",
                "points": [
                    "Built asynchronous task processing system capable of dispatching 50,000 background jobs per hour."
                ]
            }
        ],
        education=[
            {"degree": "Bachelor of Science in Computer Science", "institution": "Purdue University", "year": "2019"}
        ]
    )

    # 3. Taylor Reed: Frontend Developer (Tests "Related Skill - Not Equivalent" like Angular vs React)
    create_resume_pdf(
        filename="taylor_reed_frontend_dev.pdf",
        name="Taylor Reed",
        title="Frontend Web Developer",
        summary="Front-end developer with 4 years of experience building responsive, interactive web applications using Angular, TypeScript, and RxJS. Passionate about component architecture and UI performance.",
        skills=[
            "JavaScript", "TypeScript", "Angular", "Vue.js", "HTML5", "CSS3",
            "Git", "Webpack", "JIRA", "Agile Methodologies"
        ],
        experience=[
            {
                "role": "Frontend Developer",
                "company": "PixelCraft Studio",
                "dates": "2021 - Present",
                "points": [
                    "Built enterprise web dashboards using Angular 14 and RxJS state management, serving 200,000 monthly active users.",
                    "Facilitated bi-weekly sprint planning, daily standups, and retrospectives in Agile Scrum teams.",
                    "Tracked team velocity and managed backlog epics and bug triage workflows in Jira."
                ]
            }
        ],
        projects=[
            {
                "name": "E-Commerce Checkout Portal",
                "tech": "Angular, TypeScript, REST APIs",
                "points": [
                    "Redesigned the multi-step checkout workflow, decreasing cart abandonment rate by 18%."
                ]
            }
        ],
        education=[
            {"degree": "Bachelor of Science in Information Technology", "institution": "Ohio State University", "year": "2020"}
        ]
    )

    # 4. Sam Rivera: Marketing & Sales (Unrelated Domain)
    create_resume_pdf(
        filename="sam_rivera_marketing.pdf",
        name="Sam Rivera",
        title="Growth Marketing & Sales Specialist",
        summary="Results-driven Digital Marketing and B2B Sales Specialist with 5 years experience scaling organic search traffic, running high-ROI PPC campaigns, and managing enterprise CRM pipelines.",
        skills=[
            "Search Engine Optimization (SEO)", "Google Ads", "Content Marketing",
            "Salesforce", "Customer Relationship Management (CRM)", "Lead Generation", "B2B Sales"
        ],
        experience=[
            {
                "role": "Growth Marketing Lead",
                "company": "Apex Media Group",
                "dates": "2020 - Present",
                "points": [
                    "Implemented technical SEO improvements and structured data, raising organic search visibility by 35%.",
                    "Managed a $50k monthly budget for Google Ads campaigns with focus on keyword bidding and conversion rate.",
                    "Managed pipeline of 50+ enterprise accounts in Salesforce and achieved 115% of annual sales quota."
                ]
            }
        ],
        projects=[
            {
                "name": "Inbound Lead Funnel Revamp",
                "tech": "Salesforce CRM, Hubspot",
                "points": [
                    "Automated lead scoring and nurturing workflows, generating 400+ qualified B2B leads per quarter."
                ]
            }
        ],
        education=[
            {"degree": "Bachelor of Arts in Communications and Marketing", "institution": "Boston University", "year": "2019"}
        ]
    )

    # 5. Morgan Vance: Minimal / Scanned-like Resume (Extraction Warning test)
    create_resume_pdf(
        filename="morgan_vance_minimal.pdf",
        name="Morgan Vance",
        title="Software Intern",
        summary="Intern looking for software opportunities.",
        skills=["Python"],
        experience=[],
        projects=[],
        education=[{"degree": "High School Diploma", "institution": "Central High", "year": "2022"}]
    )

if __name__ == "__main__":
    generate_all_test_resumes()
