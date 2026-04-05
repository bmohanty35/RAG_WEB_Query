import streamlit as st
from backend import run_rag
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="RAG Web QA", layout="wide")

# Title
st.title("🌐 RAG Application (Website Q&A)")

# Highlight box
st.markdown(
    """
    <div style="background-color:#f0f8ff; padding:15px; border-radius:10px;">
        <h4>💡 What You Can Do</h4>
        <ul>
            <li>🔍 Query any <b>research article</b>, blog, or website</li>
            <li>📄 Extract key insights from long documents</li>
            <li>❓ Ask specific questions and get precise answers</li>
            <li>⚡ Avoid reading full pages manually</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# Examples
st.markdown("### 📌 Example Use Cases")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    🔬 **Research Articles**
    - Summarize findings  
    - Extract methodology  
    - Understand results  
    """)

    st.info("""
    📚 **Documentation**
    - Ask about APIs  
    - Get code explanations  
    """)

with col2:
    st.info("""
    📰 **Blogs & News**
    - Quick summaries  
    - Follow-up questions  
    """)

    st.info("""
    🏢 **Company Websites**
    - Services overview  
    - Business insights  
    """)

st.write("---")

# Inputs
st.subheader("🔗 Input")

url = st.text_input("Enter Website URL")
query = st.text_input("Enter your question")

# Button
if st.button("🚀 Get Answer"):
    if not url or not query:
        st.warning("⚠️ Please provide both URL and question")
    else:
        with st.spinner("⏳ Fetching and analyzing content..."):
            try:
                answer = run_rag(url, query)

                st.success("✅ Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.write("---")
st.caption("Built with LangChain + Gemini + Streamlit 🚀")