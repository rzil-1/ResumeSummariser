from phi.agent import Agent, RunResponse
from phi.model.ollama import Ollama
import PyPDF2

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def Summarise(file_path, role, company):
    agent = Agent(
        model = Ollama(id = "llama3.2"),
    )
    resume = extract_text_from_pdf(file_path)
    prompt = (
        "You are an expert resume assistant and career advisor."
        "Your task is to analyze the user's resume in relation to the target company and role."
        f"The user's resume content is: {resume}"
        f"The target company is: {company}"
        f"The target role is: {role}"
        "Based on the provided resume and the requirements for the specified company and role, divide your analysis into three distinct sections:"
        "1. Skills aligned with the company's and role's requirements:"
        "   - List the technical and soft skills the user currently possesses that are directly relevant to the target role and company requirements."
        "2. Skills present but less relevant to the company's and role's requirements:"
        "   - List the skills the user currently possesses that are less critical or not directly aligned with the specific needs of the target role and company."
        "3. Skills to be acquired:"
        "   - List the skills the user lacks but are essential or highly desirable for excelling in the target role and company."
        "   - Provide actionable suggestions on how the user can acquire these skills, such as courses, certifications, projects, or other resources."
        "Additional Considerations:"
        "- Ensure the analysis is tailored to the user's industry, experience level, and career goals."
        "- If the resume content is insufficient or unclear, mention assumptions made and provide guidance for improvement."
        "- Highlight any unique insights or recommendations that can enhance the user's alignment with the target role and company."
    )

    run: RunResponse = agent.run(prompt)
    return run.content, resume

def HardQuestions(resume, summary, role, company):
    agent = Agent(
        model = Ollama(id = "llama3.2"),
    )
    prompt = (
        f"You are an interviewer at the company {company}."
        f"The target role being applied for by the user is: {role}."
        f"The user's resume content is: {resume}."
        f"The summary of the skills the user possesses is: {summary}."
        "Your task is to generate ten highly detailed and in-depth questions that focus specifically on the skills the user already possesses, which align with the company's requirements for this role."
        "These questions should assess the user's depth of understanding, practical experience, problem-solving ability, and ability to apply these skills effectively in real-world scenarios."
        "Ensure that the questions are relevant to the target role and challenge the user to demonstrate their competency in these areas."
        "The questions should cover technical and soft skills, where applicable, and include both scenario-based and theoretical questions."
        "Make sure you include few questions on deep concepts of related skills."
    )
    run: RunResponse = agent.run(prompt)
    return run.content

def EasyQuestions(resume, summary, role, company):
    agent = Agent(
        model=Ollama(id="llama3.2"),
    )
    prompt = (
        f"You are an interviewer at the company {company}."
        f"The target role being applied for by the user is: {role}."
        f"The user's resume content is: {resume}."
        f"The summary of the skills the user possesses is: {summary}."
        "Your task is to generate ten surface-level questions based on:"
        "1. Skills the user possesses but are not critical or highly required by the company for this role."
        "2. Skills the user does not currently have but are required or highly desirable for this role."
        "For first category:"
        "For each category:"
        " - Create straightforward and introductory questions to assess the user’s basic understanding and familiarity with these skills."
        " - Ensure the questions are simple, allowing the user to demonstrate awareness or readiness to learn, rather than in-depth expertise."
        " - Keep the questions more technical oriented"
        "Keep the questions easy and relevant to the context of the user's profile and the target role."
    )
    run: RunResponse = agent.run(prompt)
    return run.content

summary, resume = Summarise(r"C:\Users\manda\OneDrive\Desktop\Kunaal_Joshi-resume.pdf", "associate", "Boston Consulting Group")
print(EasyQuestions(resume, summary, 'associate', 'Boston Consulting Group'))

