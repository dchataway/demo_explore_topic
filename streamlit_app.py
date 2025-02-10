import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from openai import OpenAI
import re

# Ask user for their OpenAI API key via `st.text_input`.
# or via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["OPENAI_API_KEY"]
perplexity_api_key = st.secrets["PERPLEXITY_API_KEY"]
openai_model = "gpt-4o-mini"
perplexity_model = "sonar-reasoning-pro"

def exploration_chat(topic, company_name, perplexity_model = perplexity_model, openai_api_key=openai_api_key, perplexity_api_key=perplexity_api_key, openai_model = openai_model):

    llm = ChatOpenAI(model=openai_model, temperature=0, api_key = openai_api_key)
    topic_summarized = llm.invoke(f"Summarize this R&D search topic into a single sentence: {topic}")
    
    # Use system prompt to provide directions regarding style, tone and language (but note that search doesn't attend to the system prompt)
    # Sources:
    # https://bigbangpartnership.co.uk/what-if-technique/
    # https://www.linkedin.com/advice/0/how-can-what-technique-help-you-generate-innovative-ytdae
    # https://medium.com/@QuestionClass/how-asking-what-if-questions-lead-to-business-breakthroughs-b2481c1b6efb
    system_message = '''You are a helpful AI assistant, tasked with exploring an R&D topic for a corporate director of innovation.  
    
    Rules:
    1. Provide only the final answer. It is important that you do not include any explanation on the steps below.
    2. Do not show the intermediate steps information.
    3. Make sure you structure the What If questions with this regex format (r"###\s*\d+\.\s*\*\*(What if.*?)\*\*", re.DOTALL).
    4. Make sure you structure the feasibility questions underneath each what-if question with this regex format (r"-\s*\*\*Feasibility Questions:\*\*\s*(.*?)\n\n", re.DOTALL) and split them with "\n- ".
      
    Steps:
    1. Define the challenge: The first step is to define the challenge clearly and concisely. What is the problem, opportunity, or goal that the user wants to address? What are the main criteria, constraints, and stakeholders involved? Write down your challenge as a simple statement or question that summarizes the essence of what you want to achieve."
    2. Question assumptions and apply divergent thinking: The second step is to question your assumptions and generate "what if" questions that explore different scenarios, perspectives, and alternatives. You can use various prompts, such as "What if we could...", "What if we had...", "What if we changed...", or "What if we eliminated..." to stimulate your curiosity and creativity. You can also use categories, such as customers, competitors, technology, environment, or resources, to guide your questioning. Write down as many "what if" questions as you can, without judging or filtering them."
    3. Evaluate your what-if questions: The third step is to evaluate your ideas and select the most relevant ones for further exploration. You can use various criteria, such as originality, or impact, to assess your ideas. Choose a few ideas that have the most potential to solve your challenge or meet your goal."
    4. Ask technical feasibility questions: For each what-if question, to identify a set of technical or scientific questions necessary to assess the feasiblity of the idea. Remember that you don't need to address these questions yourself, just merely ask them for future analysis. Consider the critical components of the ideas that are not yet proven via current science/engineering or the state of the art."   

    Tips:
    1. Ask Challenging Questions: Challenge conventional thinking by asking bold and thought-provoking ‘What If?’ questions. Don’t be afraid to explore radical ideas or consider scenarios that seem impossible.
    2. Explore Different Perspectives: Create a judgment-free environment where all ideas are welcomed. To stimulate your imagination, try looking at the problem from different perspectives. Put yourself in the shoes of your customers, employees, or even completely unrelated industries. This can help you uncover unique insights and generate innovative ideas.
    3. Embrace Uncertainty: The ‘What If?’ technique thrives on uncertainty. Embrace the unknown and be open to exploring uncharted territories. This is where breakthrough ideas often emerge.
    4. Combine Ideas: Don’t limit yourself to one ‘What If?’ question. Combine different ideas and scenarios to create even more innovative possibilities. Sometimes, the most groundbreaking ideas come from the intersection of seemingly unrelated concepts.
    5. Be Technical and an Expert: Ask detailed and esoteric technical questions that are highly specific to the ideas and scientific in nature. Remember that you are speaking to an innovation expert.
    '''

    user_prompt = "Apply divergent thinking to explore an R&D topic for a company. Your task is to brainstorm various 'What-if' ideas and then for each idea, identify technical feasibility questions to be assessed later. You will first need to plan how you will complete the task, then search for relevant information (ex: company products, regulations, markets, competitors) and then conduct the brainstorming actions. You will output the final results in a structured, tree-like format."
    human_message = f"Instructions: {user_prompt}. \n Inputs: 1) Company: {company_name}, 2) Topic and details: {topic}." 

    messages = [
        {
            "role": "system",
            "content": (system_message),
        },
        {
            "role": "user",
            "content": (human_message),
        },
    ]
    client = OpenAI(api_key = perplexity_api_key, base_url="https://api.perplexity.ai")
    
    # chat completion without streaming
    response = client.chat.completions.create(
        model= perplexity_model,
        messages=messages,
    )
    
    return response, topic_summarized

def feedback(company_name, topic_summarized, tree, user_request, perplexity_model = perplexity_model, openai_api_key=openai_api_key, perplexity_api_key=perplexity_api_key, openai_model = openai_model):
    '''
    https://python.langchain.com/docs/integrations/chat/openai/#tool-calling
    OpenAI has a tool calling (we use "tool calling" and "function calling" interchangeably here) API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool. 
    tool-calling is extremely useful for building tool-using chains and agents, and for getting structured outputs from models more generally.

    Output does not execute the tools 
    
    AIMessage(content='', additional_kwargs={'tool_calls': 
    '''
    
    # Define LangChain tools
    @tool
    def search_information(user_request: str, topic_summarized:str):
        """Use this to search for recent information if given a request from a user for additional What If or Technical Assessment questions. Never use this tool if the user is just asking for clarification or minor formatting updates. """
        
        client = OpenAI(api_key = "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", base_url="https://api.perplexity.ai")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant who is trained to search for information if a user asks for additions or modifications to the tree.",
            },
            {
                "role": "user",
                "content": f"Given a topic for an R&D exploration exercise and provided user feedback, search for information that will help address the user's request. \n Topic summary: {topic_summarized}. \n User Feedback: {user_request}.",
            },
        ]
        
        # chat completion without streaming
        response = client.chat.completions.create(
            model= perplexity_model,
            messages=messages,
        )
    
        return response.choices[0].message.content

    # Whenever we invoke `llm_with_tool`, these tool definitions are passed to the model.
    llm = ChatOpenAI(model=openai_model, temperature=0, api_key = openai_api_key)
    llm_with_tools = llm.bind_tools([search_information])
    
    system_message = "You are a helpful AI assistant. Determine how best to response to the user request. You have access to a `search_information` tool that should only be used if the user is asking for new information (not to be used for question-answer or clarification requests)."
    
    user_prompt = "You will be given a draft of an exploratory tree-of-thoughts exercise for a strategic R&D topic of a company. You have access to a tool `search_information` that allows you to find more information if needed. "
    human_message = f"Instructions: {user_prompt}. \n Inputs: 1) Company: {company_name}, \n 2) Topic and details: {topic_summarized}, \n 3) Drafted Exploratory Tree: {tree} ,\n 4) User Request: {user_request}." 
    
    response = llm_with_tools.invoke([
    	("system", system_message), 
    	("human", human_message),
    ])

    # ✅ Manually Execute Tool Calls
    if "tool_calls" in response.additional_kwargs:
        tool_results = []
        for tool_call in response.additional_kwargs["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = eval(tool_call["function"]["arguments"])  # Convert JSON string to dict
            tool_result = search_information.invoke(tool_args)
            tool_results.append(tool_result)
        
        ## Update the tree based on the tool information 
        system_message = '''You are a helpful AI assistant, tasked with exploring an R&D topic for a corporate director of innovation.  
            
            Rules:
            1. Provide only the final answer. It is important that you do not include any explanation on the steps below.
            2. Do not show the intermediate steps information.
            3. You must follow the same format and structure of the originally provided exploratory tree! Including:
            3i. Make sure you structure the What If questions with this regex format (r"###\s*\d+\.\s*\*\*(What if.*?)\*\*", re.DOTALL).
            3ii. Make sure you structure the feasibility questions underneath each what-if question with this regex format (r"-\s*\*\*Feasibility Questions:\*\*\s*(.*?)\n\n", re.DOTALL) and split them with "\n- ".
              
            Steps:
            1. Define the challenge: The first step is to define the challenge clearly and concisely. What is the problem, opportunity, or goal that the user wants to address? What are the main criteria, constraints, and stakeholders involved? Write down your challenge as a simple statement or question that summarizes the essence of what you want to achieve."
            2. Question assumptions and apply divergent thinking: The second step is to question your assumptions and generate "what if" questions that explore different scenarios, perspectives, and alternatives. You can use various prompts, such as "What if we could...", "What if we had...", "What if we changed...", or "What if we eliminated..." to stimulate your curiosity and creativity. You can also use categories, such as customers, competitors, technology, environment, or resources, to guide your questioning. Write down as many "what if" questions as you can, without judging or filtering them. Do not generate more than 20."
            3. Evaluate your what-if questions: The third step is to evaluate your ideas and select the most relevant ones for further exploration. You can use various criteria, such as originality, or impact, to assess your ideas. Choose a few ideas that have the most potential to solve your challenge or meet your goal."
            4. Ask technical feasibility questions: For each what-if question, to identify a set of no more than 5 technical or scientific questions necessary to assess the feasiblity of the idea. Remember that you don't need to address these questions yourself, just merely ask them for future analysis. Consider the critical components of the ideas that are not yet proven via current science/engineering or the state of the art."   

            Tips:
            1. Ask Challenging Questions: Challenge conventional thinking by asking bold and thought-provoking ‘What If?’ questions. Don’t be afraid to explore radical ideas or consider scenarios that seem impossible.
            2. Explore Different Perspectives: Create a judgment-free environment where all ideas are welcomed. To stimulate your imagination, try looking at the problem from different perspectives. Put yourself in the shoes of your customers, employees, or even completely unrelated industries. This can help you uncover unique insights and generate innovative ideas.
            3. Embrace Uncertainty: The ‘What If?’ technique thrives on uncertainty. Embrace the unknown and be open to exploring uncharted territories. This is where breakthrough ideas often emerge.
            4. Combine Ideas: Don’t limit yourself to one ‘What If?’ question. Combine different ideas and scenarios to create even more innovative possibilities. Sometimes, the most groundbreaking ideas come from the intersection of seemingly unrelated concepts.
            5. Be Technical and an Expert: Ask detailed and esoteric technical questions that are highly specific to the ideas and scientific in nature. Remember that you are speaking to an innovation expert.
            '''

        user_prompt = "You will be given a draft of an exploratory tree-of-thoughts exercise for a strategic R&D topic of a company. Your task is to update, modify or add to the tree in response to a provided User Request and corresponding information from tools (if provided). You will first need to plan how you will complete the task. The final results must remain in a similar structured, tree-like format."
        human_message = f"Original Instructions: {user_prompt}. \n Inputs: 1) Company: {company_name}, \n 2) Topic and details: {topic_summarized}, \n 3) Drafted Exploratory Tree: {tree} ,\n 4) User Request: {user_request}, \n 5) Tool Information: {str(tool_results)}. Update the provided tree accordingly. Do not remove information unless explicitly asked. " 
        
        llm = ChatOpenAI(model=openai_model, temperature=0, api_key = openai_api_key)
        
        response = llm.invoke([
        	("system", system_message), 
        	("human", human_message),
        ])
        
    else:
        pass
        #print("NO TOOL USED")

    return response


def parse_perplexity_output(response: str, citations: str = None):
    '''
    Parses the output into a user-legible format.
    '''
    output = []  # List to accumulate output

    # Extract content inside <think> tags
    explanation_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."

    # Extract content after </think>
    content_match = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    if not content_match:
        content_match = re.search(r"(.*?)", response, re.DOTALL)
    content = content_match.group(1).strip()

    # Find all "What if" questions
    what_if_pattern = re.compile(r"###\s*\d+\.\s*\*\*(What if.*?)\*\*", re.DOTALL)
    what_if_matches = what_if_pattern.findall(content)

    # Find feasibility questions
    feasibility_pattern = re.compile(r"-\s*\*\*Feasibility Questions:\*\*\s*(.*?)\n\n", re.DOTALL)
    feasibility_matches = feasibility_pattern.findall(content)

    # Structure results
    structured_tree = []
    for i, what_if in enumerate(what_if_matches):
        feasibility_questions = []
        if i < len(feasibility_matches):
            clean_text = feasibility_matches[i].strip()  # Remove leading/trailing spaces
            if clean_text.startswith("- "):  # Ensure we don't get an extra empty entry
                clean_text = clean_text[2:]
            feasibility_questions = [q.strip() for q in clean_text.split("\n- ") if q.strip()]
        structured_tree.append((what_if.strip(), feasibility_questions))

    # Append structured tree to output
    for i, (what_if, questions) in enumerate(structured_tree, start=1):
        output.append(f"{i}. {what_if}")
        for q in questions:
            output.append(f"   - {q}")
        output.append("\n" + "-" * 50)  # Separator for readability

    # Append Explanation Section
    output.append("\n### Appendix 1: Explanation")
    output.append(explanation)

    # Append Citations Section
    output.append("\n### Appendix 2: Citations")
    if citations:
        for i, citation in enumerate(citations, start=1):
            output.append(f"{i}. {citation}")

    return "\n".join(output)

################## CONTENT ########################

# Show title and description.
st.title("💬 R&D Exploration Tool ")
st.write(
    "This is a simple demo of a tool that explores an open-ended R&D topic for a company. It applies divergent thinking to consider 1) what-if questions and 2) corresponding technical assessment questions."
    "To use this app, you need to provide your company name and details (of any length) about the R&D topic. You may also provide feedback afterwards for revisions and additions."
    " Your information is not saved or stored."
)

# Initialize session state variables
if "company_name" not in st.session_state:
    st.session_state.company_name = None
if "topic" not in st.session_state:
    st.session_state.topic = None
if "result_parsed" not in st.session_state:
    st.session_state.result_parsed = None  # Ensure this is defined before feedback starts

# Step 1: Ask for company name
if st.session_state.company_name is None:
    company_name = st.text_input("Provide your company name:", key="company_name_input")
    if company_name:
        st.session_state.company_name = company_name
        # st.experimental_rerun() # Rerun to hide the first input

# Step 2: Ask for topic once company name is provided
if st.session_state.company_name and st.session_state.topic is None:
    topic = st.text_input("Provide details about the topic (paste in plain text):", key="topic_input")
    if topic:
        st.session_state.topic = topic

# Once both inputs are collected, generate response
if st.session_state.company_name and st.session_state.topic:
    st.write("Give me a minute to think...")
    result, topic_summary = exploration_chat(st.session_state.topic, st.session_state.company_name)
    st.session_state.result_parsed = parse_perplexity_output(result.choices[0].message.content, result.citations)  # Store in session state

    with st.chat_message("assistant"):
        st.write(st.session_state.result_parsed)  # Use the stored result

# FEEDBACK
# Initialize session state variables
if "feedback_iteration" not in st.session_state:
    st.session_state.feedback_iteration = 0
if "feedback_messages" not in st.session_state:
    st.session_state.feedback_messages = []

# Show feedback input **only if result_parsed is written**
if st.session_state.result_parsed:
    # Display all assistant responses so far
    for message in st.session_state.feedback_messages:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

    # If feedback iteration is less than 5, allow user input for feedback
    if st.session_state.feedback_iteration < 5:
        user_input = st.text_input(
            f"Enter feedback (or 'Done' to exit): ",
            key=f"feedback_input_{st.session_state.feedback_iteration+3}"
        )

        if user_input:
            if user_input.strip().lower() == "done":
                st.session_state.feedback_iteration = 5  # End feedback process
            else:
                # Process feedback without rerunning the app
                st.write("Give me a minute to think...")
                result_feedback = feedback(st.session_state.company_name, topic_summary, st.session_state.result_parsed, user_input)

                # Store assistant’s response in session state
                st.session_state.feedback_messages.append({"role": "assistant", "content": result_feedback.content})

                # Increment feedback iteration
                st.session_state.feedback_iteration += 1

                # Display the assistant's feedback immediately
                with st.chat_message("assistant"):
                    st.markdown(result_feedback.content)
                
        else:
            st.write("Waiting for your feedback...")

# If max iterations reached, stop input
if st.session_state.feedback_iteration >= 5:
    st.write("No more input allowed. Please start a new session.")
