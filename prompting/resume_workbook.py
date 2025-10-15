import base64
import os
from dataclasses import dataclass
from typing import Optional, cast

import marimo as mo
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

__generated_with = "0.16.5"

app = mo.App(width="large")


class KeywordExtraction(BaseModel):
    """Structured response containing extracted job posting keywords."""

    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of the most relevant keywords, skills, and focus areas "
            "identified in the job posting."
        ),
    )


class ResumeEnhancement(BaseModel):
    """Container for an enhanced resume."""

    enhanced_resume: str = Field(
        ..., description="A refined resume tailored to the target job posting."
    )


class CoverLetterPackage(BaseModel):
    """Bundle containing the generated cover letter and recruiter outreach note."""

    cover_letter: str = Field(
        ..., description="Tailored cover letter ready to share with the employer."
    )
    recruiter_message: str = Field(
        ...,
        description="Concise, friendly note that can be sent directly to the recruiter.",
    )


class ApplicationBundle(BaseModel):
    """Aggregated deliverables returned to the UI."""

    keywords: list[str]
    enhanced_resume: str
    cover_letter: str
    recruiter_message: str


@dataclass
class JobApplicationAgents:
    """LLM agents dedicated to each stage of the workflow."""

    keyword_agent: Agent[KeywordExtraction]
    resume_agent: Agent[ResumeEnhancement]
    cover_letter_agent: Agent[CoverLetterPackage]


def _build_model(model_env: str, default_model: str) -> OpenAIChatModel:
    """Create an OpenAI chat model using environment configuration."""

    provider = os.getenv("OPENAI_PROVIDER", "openai")
    model_name = os.getenv(model_env, default_model)
    return OpenAIChatModel(model_name=model_name, provider=provider)


def _default_model_name() -> str:
    """Resolve a default model name with a sensible fallback."""

    return os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")


def build_agents() -> JobApplicationAgents:
    """Instantiate all specialized agents used in the workflow."""

    default_model = _default_model_name()

    keyword_agent = Agent(
        model=_build_model("KEYWORD_MODEL", default_model),
        name="keyword_extractor",
        instructions=(
            "You extract the most important keywords, phrases, competencies, tools, and "
            "qualification requirements from a job posting. Return between 8 and 15 "
            "items ordered by importance."
        ),
        output_type=KeywordExtraction,
    )

    resume_agent = Agent(
        model=_build_model("RESUME_MODEL", default_model),
        name="resume_enhancer",
        instructions=(
            "Create an updated resume using the supplied job posting context, list of "
            "target keywords, any original resume content, and additional guidance. "
            "Preserve professional tone, ensure all relevant experience is aligned "
            "to the keywords, and format the result as Markdown with clear headings "
            "and bullet points where helpful."
        ),
        output_type=ResumeEnhancement,
    )

    cover_letter_agent = Agent(
        model=_build_model("COVER_LETTER_MODEL", default_model),
        name="cover_letter_specialist",
        instructions=(
            "Craft both a tailored cover letter and a short recruiter outreach message "
            "(120 words max) based on the job posting, enhanced resume, original cover "
            "letter (if available), and any extra instructions. Maintain a confident, "
            "warm, and professional tone."
        ),
        output_type=CoverLetterPackage,
    )

    return JobApplicationAgents(
        keyword_agent=keyword_agent,
        resume_agent=resume_agent,
        cover_letter_agent=cover_letter_agent,
    )


def encode_upload(upload: mo.ui.file, *, label: str) -> Optional[str]:
    """Represent uploaded binary files as base64 strings for the LLM."""

    if not upload.value:
        return None

    file_payload = upload.value[0]
    encoded = base64.b64encode(file_payload.contents).decode("utf-8")
    return (
        f"Uploaded {label}: {file_payload.name}\n"
        f"Base64Contents: {encoded}"
        "\n\nProvide docx output later."
    )


def merge_inputs(
    *,
    file_payload: Optional[str],
    text_value: str,
    label: str,
) -> Optional[str]:
    """Combine text and file uploads into a single string for the agent."""

    text_value = text_value.strip()
    if file_payload and text_value:
        return (
            f"{label} (user provided text):\n{text_value}\n\n"
            f"{label} (uploaded file reference):\n{file_payload}"
        )
    if text_value:
        return f"{label}:\n{text_value}"
    if file_payload:
        return file_payload
    return None


def generate_application_package(
    agents: JobApplicationAgents,
    *,
    job_posting: str,
    resume: Optional[str],
    cover_letter: Optional[str],
    instructions_text: Optional[str],
) -> ApplicationBundle:
    """Run the sequential LLM workflow to build the final deliverables."""

    keyword_result = agents.keyword_agent.run(
        {
            "job_posting": job_posting,
            "goal": (
                "Identify the essential skills, technologies, experience levels, and "
                "cultural cues that matter most in this posting."
            ),
        }
    )
    keywords = keyword_result.output.keywords

    resume_result = agents.resume_agent.run(
        {
            "job_posting": job_posting,
            "target_keywords": keywords,
            "original_resume": resume or "",
            "additional_instructions": instructions_text or "",
        }
    )
    enhanced_resume = resume_result.output.enhanced_resume

    cover_letter_result = agents.cover_letter_agent.run(
        {
            "job_posting": job_posting,
            "enhanced_resume": enhanced_resume,
            "original_cover_letter": cover_letter or "",
            "additional_instructions": instructions_text or "",
            "target_keywords": keywords,
        }
    )

    package = cover_letter_result.output
    return ApplicationBundle(
        keywords=keywords,
        enhanced_resume=enhanced_resume,
        cover_letter=package.cover_letter,
        recruiter_message=package.recruiter_message,
    )


@app.cell
def _():
    custom_css = mo.md(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');

        body {
            background-color: #F9F7F4;
            color: #000000;
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Bebas Neue', Helvetica, Arial, sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #000000;
        }

        .headline-accent {
            border-bottom: 3px solid #9D0208;
            padding-bottom: 12px;
            margin-bottom: 24px;
        }

        .pill {
            display: inline-block;
            background-color: #FDECEC;
            color: #9D0208;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 14px;
            margin: 4px;
            font-weight: 600;
            border: 1px solid #FFB3B3;
            letter-spacing: 0.5px;
        }

        .card {
            background: #FFFFFF;
            border: 1px solid #E7D7C9;
            border-radius: 18px;
            padding: 24px 28px;
            box-shadow: 0px 12px 24px rgba(0, 0, 0, 0.04);
        }

        .cta-button {
            background-color: #E5383B !important;
            border: 2px solid #9D0208 !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            letter-spacing: 1.5px !important;
            padding: 16px 22px !important;
            text-transform: uppercase !important;
        }
        </style>
        """
    )
    custom_css


@app.cell
def _():
    hero = mo.md(
        """
        <section style="display: flex; flex-direction: column; align-items: center; padding: 60px 20px;">
            <div style="text-align: center; max-width: 880px;">
                <h1 style="font-size: 72px; margin-bottom: 12px;" class="headline-accent">
                    Tailored Application Studio
                </h1>
                <p style="font-size: 20px; color: #333; line-height: 1.7;">
                    Upload a job description or paste the text, layer in your resume materials, and let our
                    multi-step workflow craft an optimized resume, cover letter, and recruiter message for you.
                </p>
            </div>
        </section>
        """
    )
    hero


@app.cell
def _():
    job_posting_file = mo.ui.file(
        label="**Job posting** — upload a .docx or .pdf", filetypes=[".docx", ".pdf"], kind="area"
    )
    resume_file = mo.ui.file(
        label="**Resume** — upload a .docx or .pdf", filetypes=[".docx", ".pdf"], kind="area"
    )
    cover_letter_file = mo.ui.file(
        label="**Existing cover letter (optional)**", filetypes=[".docx", ".pdf"], kind="area"
    )

    job_posting_text = mo.ui.text_area(
        label="**Or paste the job posting text**", placeholder="Paste the job description here..."
    )
    resume_text = mo.ui.text_area(
        label="**Or paste your resume**", placeholder="Paste your current resume..."
    )
    cover_letter_text = mo.ui.text_area(
        label="**Or paste your cover letter (optional)**", placeholder="Paste an existing cover letter..."
    )
    extra_instructions = mo.ui.text_area(
        label="**Additional instructions for the agent**",
        placeholder="Highlight accomplishments, mention constraints, or describe target tone...",
    )

    inputs_layout = mo.vstack(
        [
            mo.hstack([job_posting_file, job_posting_text], gap="24px"),
            mo.hstack([resume_file, resume_text], gap="24px"),
            mo.hstack([cover_letter_file, cover_letter_text], gap="24px"),
            extra_instructions,
        ],
        gap="32px",
    ).style({"padding": "0 20px"})

    inputs_layout


@app.cell
def _(job_posting_file, job_posting_text, resume_file, resume_text, cover_letter_file, cover_letter_text):
    job_posting_payload = encode_upload(job_posting_file, label="job posting")
    resume_payload = encode_upload(resume_file, label="resume")
    cover_letter_payload = encode_upload(cover_letter_file, label="cover letter")

    merged_job_posting = merge_inputs(
        file_payload=job_posting_payload,
        text_value=job_posting_text.value,
        label="Job posting",
    )
    merged_resume = merge_inputs(
        file_payload=resume_payload,
        text_value=resume_text.value,
        label="Resume",
    )
    merged_cover_letter = merge_inputs(
        file_payload=cover_letter_payload,
        text_value=cover_letter_text.value,
        label="Cover letter",
    )

    return (
        merged_job_posting,
        merged_resume,
        merged_cover_letter,
    )


@app.cell
def _():
    get_generation_state, set_generation_state = mo.state(
        cast(Optional[ApplicationBundle], None)
    )
    get_error_state, set_error_state = mo.state(cast(Optional[str], None))
    get_agents, _ = mo.state(build_agents())

    return (
        get_agents,
        get_error_state,
        get_generation_state,
        set_error_state,
        set_generation_state,
    )


@app.cell
def _(
    get_agents,
    get_error_state,
    extra_instructions,
    get_generation_state,
    merged_cover_letter,
    merged_job_posting,
    merged_resume,
    set_error_state,
    set_generation_state,
):
    def handle_generate(counter: int | None) -> int:
        set_error_state(None)
        if not merged_job_posting:
            set_error_state("Please provide a job posting via upload or text.")
            set_generation_state(None)
            return (counter or 0) + 1

        try:
            package = generate_application_package(
                get_agents(),
                job_posting=merged_job_posting,
                resume=merged_resume,
                cover_letter=merged_cover_letter,
                instructions_text=extra_instructions.value,
            )
        except Exception as exc:  # pragma: no cover - UI surfaced for the user
            set_error_state(str(exc))
            set_generation_state(None)
        else:
            set_generation_state(package)
        return (counter or 0) + 1

    generate_button = mo.ui.button(
        label="Generate tailored resume & cover letter",
        kind="success",
        value=0,
        on_click=handle_generate,
        full_width=True,
    ).style({"margin": "0 20px"})

    status_callout = mo.md(
        f"""
        :::info
        **Models in use**  
        Keywords → `{os.getenv('KEYWORD_MODEL', os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-4o-mini'))}`  
        Resume → `{os.getenv('RESUME_MODEL', os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-4o-mini'))}`  
        Cover letter → `{os.getenv('COVER_LETTER_MODEL', os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-4o-mini'))}`
        :::
        """
    ).style({"margin": "20px"})

    layout = mo.vstack([generate_button, status_callout], gap="16px")
    layout


@app.cell
def _(get_error_state, get_generation_state):
    if get_error_state():
        mo.md(
            f"""
            :::danger
            **Generation failed**  
            {get_error_state()}
            :::
            """
        )
    elif get_generation_state():
        package = get_generation_state()
        keywords_html = "".join(
            f'<span class="pill">{keyword}</span>' for keyword in package.keywords
        )

        mo.vstack(
            [
                mo.md(
                    f"""
                    <section class="card">
                        <h2>Keyword spotlight</h2>
                        <div>{keywords_html}</div>
                    </section>
                    """
                ),
                mo.md(
                    f"""
                    <section class="card">
                        <h2>Enhanced resume draft</h2>
                        {mo.md(package.enhanced_resume).text}
                    </section>
                    """
                ),
                mo.md(
                    f"""
                    <section class="card">
                        <h2>Cover letter</h2>
                        {mo.md(package.cover_letter).text}
                    </section>
                    """
                ),
                mo.md(
                    f"""
                    <section class="card">
                        <h2>Recruiter message</h2>
                        <p style="font-size: 16px; line-height: 1.6; color: #222;">{package.recruiter_message}</p>
                    </section>
                    """
                ),
            ],
            gap="28px",
        ).style({"padding": "20px"})
    else:
        mo.md(
            """
            :::neutral
            Upload a job posting and resume materials, then click the button above to start the workflow.
            :::
            """
        ).style({"margin": "20px"})


if __name__ == "__main__":
    app.run()
