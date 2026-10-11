"""
One-off generator for the bundled sample document used by the demo seed
script and the frontend's "Try a sample" button. Run this manually if you
ever want to regenerate the PDF (e.g. to change its content):

    python3 sample_data/generate_sample_pdf.py

The output is checked into sample_data/rag_overview.pdf and
frontend/public/sample/rag_overview.pdf directly -- this script does not
run as part of the app.
"""
import shutil
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

OUTPUT_PATH = Path(__file__).parent / "rag_overview.pdf"
FRONTEND_COPY_PATH = Path(__file__).parent.parent.parent / "frontend" / "public" / "sample" / "rag_overview.pdf"

styles = getSampleStyleSheet()
heading = ParagraphStyle("SampleHeading", parent=styles["Heading1"], spaceAfter=10)
body = ParagraphStyle("SampleBody", parent=styles["Normal"], fontSize=11, leading=16, spaceAfter=10)

story = []

story.append(Paragraph("Retrieval-Augmented Generation: An Overview", styles["Title"]))
story.append(Spacer(1, 16))

story.append(Paragraph("What is Retrieval-Augmented Generation?", heading))
story.append(Paragraph(
    "Retrieval-Augmented Generation, or RAG, is an architecture that combines a retrieval "
    "system with a text-generation model. Instead of relying only on knowledge encoded in the "
    "model's parameters during training, a RAG system first retrieves relevant passages from an "
    "external knowledge source -- such as a document collection or a database -- and then "
    "provides those passages to the generation model as context. The two core components are "
    "the retriever, which finds relevant evidence for a given query, and the generator, which "
    "produces a natural-language answer conditioned on that evidence.",
    body,
))
story.append(Paragraph(
    "This architecture was introduced as a way to give language models access to information "
    "that changes over time or is too specific to have been memorized during training, without "
    "requiring the model itself to be retrained.",
    body,
))

story.append(Paragraph("Why Do Language Models Hallucinate?", heading))
story.append(Paragraph(
    "Large language models generate text by predicting the most statistically likely next "
    "token, not by consulting a verified source of facts. When a model is asked about something "
    "it was not trained on, or something it recalls imperfectly, it can still produce a fluent, "
    "confident-sounding answer that is factually wrong. This behavior is commonly called "
    "hallucination: the model states something with no grounding in any actual source, but "
    "phrased no differently than a true statement would be.",
    body,
))
story.append(Paragraph(
    "Hallucination is considered especially risky in applications where users are likely to "
    "trust an answer at face value, such as medical, legal, or educational tools, because "
    "nothing about the surface form of a hallucinated sentence distinguishes it from a correct "
    "one.",
    body,
))

story.append(PageBreak())

story.append(Paragraph("How RAG Reduces Hallucination", heading))
story.append(Paragraph(
    "By grounding a generated answer in retrieved evidence, RAG reduces -- but does not "
    "eliminate -- hallucination. The model is given real, retrieved text to draw from, which "
    "makes it more likely to produce an answer that is actually supported by a source. However, "
    "a generated answer can still introduce claims that go beyond what was retrieved, "
    "especially when the model paraphrases loosely, combines multiple retrieved facts "
    "incorrectly, or fills a gap in the retrieved evidence with something plausible-sounding "
    "instead of leaving it unanswered.",
    body,
))

story.append(Paragraph("Common Causes of RAG Failures", heading))
story.append(Paragraph(
    "RAG systems can fail to produce a faithful answer for several distinct reasons. "
    "Retrieval failure occurs when nothing relevant to the question was retrieved at all. "
    "Insufficient context occurs when retrieval found something relevant, but not enough "
    "information to fully answer the question. Context misinterpretation occurs when relevant "
    "evidence was retrieved, but the generation model drew the wrong conclusion from it. "
    "Unsupported generation occurs when the model introduces a claim that has no basis in the "
    "retrieved evidence at all, effectively hallucinating on top of a grounded answer. "
    "Conflicting evidence occurs when different retrieved passages disagree with each other, "
    "leaving the model to reconcile -- or fail to reconcile -- a contradiction.",
    body,
))

story.append(PageBreak())

story.append(Paragraph("Evaluating Faithfulness in RAG Outputs", heading))
story.append(Paragraph(
    "Because hallucination in RAG systems can happen at the claim level rather than the whole "
    "answer, evaluation is usually done by breaking a generated answer into individual claims "
    "and checking each one against the retrieved evidence. Several families of methods exist "
    "for this. Lexical overlap methods, such as ROUGE-L, measure how much wording a claim "
    "shares with the evidence. Natural language inference methods use a model trained to judge "
    "whether one piece of text logically entails another, applied here to test whether the "
    "evidence entails the claim. LLM-judge methods prompt a separate language model to directly "
    "verdict each claim as supported, partially supported, or unsupported. RAGAS faithfulness "
    "follows a related but distinct methodology, asking for a strictly binary judgment per "
    "claim. Self-consistency methods take a different approach entirely: instead of checking a "
    "claim against retrieved evidence, they check whether the same claim reappears when the "
    "model is asked the same question again independently, on the theory that hallucinated "
    "claims tend to be less stable across independent generations than genuinely grounded ones.",
    body,
))
story.append(Paragraph(
    "No single method perfectly predicts human judgment of faithfulness, which is why "
    "comparing multiple methods against real human-labeled examples -- rather than trusting "
    "any one method in isolation -- is considered good practice when building a faithfulness "
    "evaluation system.",
    body,
))

doc = SimpleDocTemplate(str(OUTPUT_PATH), pagesize=letter, topMargin=0.9 * inch, bottomMargin=0.9 * inch)
doc.build(story)
print(f"Wrote {OUTPUT_PATH}")

FRONTEND_COPY_PATH.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(OUTPUT_PATH, FRONTEND_COPY_PATH)
print(f"Copied to {FRONTEND_COPY_PATH}")
