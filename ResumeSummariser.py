from phi.agent import Agent, RunResponse
from phi.model.google import Gemini
import PyPDF2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''.join([page.extract_text() for page in reader.pages])
    return text

class ResumeBuilder:
    def __init__(self, file_path, role, company):
        """Initialize with resume content, role, and company."""
        self.resume = extract_text_from_pdf(file_path)
        self.role = role
        self.company = company
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=os.getenv('GEMINI_API_KEY'))
        )

    def analyze_keywords(self):
        """Analyze keywords in the 'Projects' section of the resume."""
        prompt = (
            "You are a highly skilled and experienced keyword analysis expert specializing in resumes and professional documents."
            f"You are provided with a resume. The content of the resume is as follows: {self.resume}."
            "Your task is to analyze the 'Projects' section of the given resume thoroughly."
            "For each project mentioned in the 'Projects' section, identify all relevant keywords, including but not limited to:"
            "1. Technical terms (e.g., programming languages, frameworks, tools)."
            "2. Domain-specific jargon (e.g., finance, healthcare, AI-related terminology)."
            "3. Other significant descriptors or action verbs."
            "The output should be in the following format:"
            "Project Name: [Comma-separated list of identified keywords for this project]"
            "Ensure the output is accurate, concise, and captures the essence of the key elements from each project."
        )
        run: RunResponse = self.agent.run(prompt)
        return run.content

    def analyze_depth(self, keyword_analysis):
        """Decompose keywords into detailed sub-concepts."""
        prompt = (
            f"For each of the keywords specified for every project in {keyword_analysis},"
            "Your task is to:"
            "1. For every Keyword, decompose it into specific, detailed sub-concepts or components, ensuring each sub-concept is well-defined and clearly explains its role or significance."
            "2. Include technical, functional, and contextual sub-concepts where applicable, considering the nuances of the domain or field."
            "3. Ensure each sub-concept is comprehensive, granular, and captures all relevant dimensions."
            "The output should follow this structured format:"
            "Project1:"
            "Keyword1: Sub-concept1, Sub-concept2, Sub-concept3, ..."
            "Keyword2: Sub-concept1, Sub-concept2, Sub-concept3, ..."
            "Project2:"
            "Keyword1: Sub-concept1, Sub-concept2, Sub-concept3, ..."
            "Keyword2: Sub-concept1, Sub-concept2, Sub-concept3, ..."
            "Make sure the decomposition is exhaustive and provides a detailed understanding of each topic and its constituent elements. Include examples, if relevant, to clarify complex sub-concepts."
        )
        run: RunResponse = self.agent.run(prompt)
        return run.content

    def analyze_skills(self):
        """Analyze skills in the resume relative to the company and role."""
        prompt = (
            "You are a highly experienced resume assistant and career advisor with expertise in tailoring resumes for specific companies and roles."
            "Your task is to conduct a detailed analysis of the user's resume in relation to the target company and role."
            f"The user's resume content is as follows: {self.resume}."
            f"The target company is: {self.company}."
            f"The target role is: {self.role}."
            "Your analysis must be divided into three clear and distinct sections, focusing on skills alignment and gaps in relation to the requirements of the specified company and role:"
            "1. Skills aligned with the company's and role's requirements:"
            "   - Identify and list all technical, functional, and soft skills currently present in the user's resume that are directly relevant to the requirements of the target role and company."
            "   - Ensure these skills are contextually aligned with the industry, domain, and specific job responsibilities."
            "2. Skills present but less relevant to the company's and role's requirements:"
            "   - Identify and list the skills mentioned in the resume that, while useful, are less critical or not directly aligned with the specific requirements of the target role and company."
            "   - Focus on skills that could be peripheral or secondary in the context of the job."
            "3. Skills to be acquired:"
            "   - Identify and list the key skills missing from the user's resume that are essential or highly desirable for excelling in the specified role and company."
            "   - Include any industry-specific certifications, tools, or methodologies that are commonly expected or advantageous for the role."
            "   - Highlight areas for upskilling, including both technical and soft skills, where applicable."
            "The output should be concise and in the following format, without any additional explanations:"
            "Skills aligned with both: [List of skills]."
            "Skills present but less relevant: [List of skills]."
            "Skills to be acquired: [List of skills]."
            "Ensure the analysis is thorough, precise, and directly tailored to the target role and company."
        )
        run: RunResponse = self.agent.run(prompt)
        return run.content

    def generate_interview_questions(self):
        """Generate interview questions based on keyword, depth, and skill analyses."""
        keywords = self.analyze_keywords()
        depth_analysis = self.analyze_depth(keywords)
        skill_analysis = self.analyze_skills()
        prompt = (
            f"You are an experienced and highly skilled interviewer representing the company: {self.company}."
            f"The user's resume content is provided as follows: {self.resume}."
            f"The target role for the user is: {self.role}."
            f"The {depth_analysis} provides detailed insights into the relevant concepts and sub-concepts that need to be focused on during the interview."
            f"The {skill_analysis} outlines the necessity and relevance of each skill, along with their associated subtopics, based on the requirements of the company and the target role."
            "Your task is to generate 10 well-structured interview questions, ensuring the following:"
            "1. **Question Depth**:"
            "   - The first two questions should be medium-level, focusing on moderately detailed subtopics."
            "   - All remaining questions must be in-depth, highly specific, and focus on niche subtopics critical to the role and company requirements."
            "2. **Subtopic-Centric Questions Only**: All questions must be built exclusively on the subtopics derived from the keywords in the skill analysis. No question should directly reference the keyword itself."
            "3. **Weightage on Technical and Functional Skills**:"
            "   - Focus primarily on technical and functional skills, crafting detailed questions to evaluate the user's depth of knowledge, expertise, and practical application."
            "   - Soft skills may be given minimal weightage or omitted entirely unless explicitly relevant to the role."
            "4. **Skill-Relevance Focus**:"
            "   - For skills strongly aligned with the role and company, prioritize creating detailed, highly specific questions exploring deeper subtopics."
            "   - For moderately aligned skills, use subtopics to create questions that evaluate foundational understanding and applicability."
            "5. **Practical Application Emphasis**: Frame questions to test the user's problem-solving abilities, hands-on experience, and practical application of concepts, especially for technical and functional skills."
            "6. **Avoid Generic Questions**: Ensure each question is uniquely tailored, leveraging subtopics to reflect the specific requirements of the target role and company."
            "7. **Structured Format**: The output should follow this structured format:"
            "   1. [Medium-level question using a moderately detailed subtopic]"
            "   2. [Medium-level question using another moderately detailed subtopic]"
            "   3. [Highly specific and niche question on a critical subtopic]"
            "   ..."
            "   10. [Highly specific and niche question on another critical subtopic]"
            "Ensure all questions are logically sequenced, focus mainly on technical and functional skills, and are aligned with the depth required to evaluate the user's fit for the role and company."
        )

        run: RunResponse = self.agent.run(prompt)
        return run.content

if __name__ == "__main__":
    builder = ResumeBuilder(
        r"C:\Users\manda\OneDrive\Desktop\Kunaal_Joshi-resume.pdf",
        "associate",
        "Boston Consulting Group"
    )
    questions = builder.generate_interview_questions()
    print(questions)