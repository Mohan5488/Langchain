import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

# Load API Key
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

# Initialize LLM
llm = ChatGroq(
    api_key=api_key,
    model_name="llama3-8b-8192",
    temperature=0.7
)

def generate_social_post(topic, no_words, platform, tone):
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    prompt = PromptTemplate(
        input_variables=["topic", "no_words", "platform", "tone"],
        template=f"Write a {tone} social media post for {platform} about '{topic}' within {no_words} words. Make it engaging and appropriate for the platform."
    )
    print(prompt.format(topic=topic, no_words=no_words, platform=platform, tone=tone))
    
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    response = chain.run(topic=topic, no_words=no_words, platform=platform, tone=tone)
    return response

st.title("📱 Social Media Post Generator")

topic = st.text_area("Enter a topic for your post")
col1, col2 = st.columns([5,5])
col3, = st.columns([5])

with col1:
    no_words = st.text_input("Number of words")
with col2:
    platform = st.selectbox(
        "Choose Platform",
        ['LinkedIn', 'Twitter', 'Instagram', 'Facebook', 'YouTube', 'WhatsApp']
    )
with col3:
    tone = st.selectbox("Select Tone", 
                        ["Professional", "Casual", "Inspirational", "Funny", "Marketing"])

submit = st.button("🚀 Generate Post")
if submit:
    if topic and no_words and platform and tone:
        post = generate_social_post(topic, no_words, platform, tone)
        st.subheader(f"Generated {platform} Post:")
        st.write(post)
    else:
        st.error("Please fill in all fields.")
